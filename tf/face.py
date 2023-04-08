import os
import json
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import temperflow.option as opts
opts.set_opts(jit=True, debug=False)
from temperflow.block import SmoothedReLU
from temperflow.bijector import SplineCoupling, BlockAutoRegSpline
from temperflow.distribution import TransformedDistribution
from temperflow.flow import TemperFlow
from pretrained.face import Face_Generator, FaceLatentEnergy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Hyper-parameters
img_size = 64
input_dim = img_size**2 * 3
latent_dim = 100
base_channels = 64
count_bins = 32
layer_width = 256

# Paths
cache_path = "cache"
model_path = "model"
os.makedirs(cache_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

np.random.seed(123)
tf.random.set_seed(123)
# Generator
generator = Face_Generator(latent_dim=latent_dim, base_channels=base_channels)
generator.load_params(f"pretrained/face-gmmvae-generator.npz")
# Flow
flow = SplineCoupling(
    dim=latent_dim, depth=16, count_bins=32, bound=5.0,
    hlayers=[256, 256])
bijdistr = TransformedDistribution(dim=latent_dim, bijector=flow)
bijdistr.load_params(f"{model_path}/face-gmmvae-flow.npz")

# Energy function for sampling
energy = FaceLatentEnergy(bijdistr)



# Generated images
np.random.seed(123)
tf.random.set_seed(123)
zacc = bijdistr.sample(1000)
fig = plt.figure(figsize=(12, 12))
sub = fig.add_subplot(111)
xsamp = generator(zacc[:100], apply_tanh=True) * 0.5 + 0.5
img = xsamp.numpy().transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3). \
        transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
plt.imshow(img)
plt.axis("off")
plt.show()
plt.savefig(f"{model_path}/face_gen.png", bbox_inches="tight")



# TemperFlow
np.random.seed(123)
tf.random.set_seed(123)
bij_tf = BlockAutoRegSpline(
    dim=latent_dim, block_size=10, count_bins=count_bins, bound=6.0,
    hlayers=[layer_width, layer_width], nonlinearity=SmoothedReLU)
bijdistr_tf = TransformedDistribution(dim=latent_dim, bijector=bij_tf)
temper = TemperFlow(energy, bijdistr_tf)

t1 = time.time()
trace = temper.train_flow(beta0=0.25, kl_decay=0.8, nepoch=(2501, 2001, 101),
                          bs=2500, lr=0.001, max_betas=100, verbose=2,
                          ref_cutoff=None, checkpoint=f"{cache_path}/face")
t2 = time.time()
print(f"{t2 - t1} seconds")
bijdistr_tf.save_params(f"{model_path}/face-l2.npz")
with open(f"{model_path}/face-l2.json", mode="w") as f:
    json.dump(trace, f, indent=2)

# bijdistr_tf.load_params(f"{model_path}/face-l2.npz")
# with open(f"{model_path}/face-l2.json", mode="r") as f:
#     trace = json.load(f)

# Generated images
np.random.seed(123)
tf.random.set_seed(123)
fig = plt.figure(figsize=(12, 12))
sub = fig.add_subplot(111)
zsamp = bijdistr_tf.sample(100)
xsamp = generator(zsamp, apply_tanh=True) * 0.5 + 0.5
img = xsamp.numpy().transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3). \
        transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
plt.imshow(img)
plt.axis("off")
plt.show()
plt.savefig(f"{model_path}/face_l2_gen.png", bbox_inches="tight")



# Measure transport via KL
np.random.seed(123)
tf.random.set_seed(123)
bij_kl = BlockAutoRegSpline(
    dim=latent_dim, block_size=10, count_bins=count_bins, bound=6.0,
    hlayers=[layer_width, layer_width], nonlinearity=SmoothedReLU)
bijdistr_kl = TransformedDistribution(dim=latent_dim, bijector=bij_kl)
flow_kl = TemperFlow(energy, bijdistr_kl)

t1 = time.time()
losses = flow_kl.train_one_beta(beta=1.0, fn="kl", bs=2000, nepoch=5001, lr=0.001, verbose=True)
t2 = time.time()
print(f"{t2 - t1} seconds")
bijdistr_kl.save_params(f"{model_path}/face-kl.npz")
# bijdistr_kl.load_params(f"{model_path}/face-kl.npz")

# Generated images
np.random.seed(123)
tf.random.set_seed(123)
fig = plt.figure(figsize=(12, 12))
sub = fig.add_subplot(111)
zsamp = bijdistr_kl.sample(100)
xsamp = generator(zsamp, apply_tanh=True) * 0.5 + 0.5
img = xsamp.numpy().transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3). \
        transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
plt.imshow(img)
plt.axis("off")
plt.show()
plt.savefig(f"{model_path}/face_kl_gen.png", bbox_inches="tight")



# MCMC
np.random.seed(123)
tf.random.set_seed(123)
num_results = 1
num_chains = 100

# Kernel for M-H
kernel_mh = tfp.mcmc.RandomWalkMetropolis(
    target_log_prob_fn=energy.log_pdf,
    new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=0.01)
)
# Kernel for HMC
kernel_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=energy.log_pdf,
    step_size=0.01,
    num_leapfrog_steps=5
)

# Kernel for parallel tempering
# Inverse temperature parameters
# inv_temp = 0.5**tf.range(5, dtype=np.float32)
inv_temp = tf.exp(tf.linspace(0.0, math.log(0.1), num=5))
step_size = 0.01 / tf.sqrt(inv_temp)
step_size = tf.reshape(step_size, shape=(5, 1))
kernel_pt = tfp.mcmc.ReplicaExchangeMC(
    target_log_prob_fn=energy.log_pdf,
    inverse_temperatures=inv_temp,
    make_kernel_fn=lambda target_log_prob_fn: tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size))
)
# Initial state of the chain
init_state = tf.random.normal(shape=(num_chains, latent_dim))

@tf.function
def run_mcmc(kernel, num_results, init_state, trace=False):
    trace_fn = (lambda current_state, kernel_results: kernel_results) if trace else None
    res = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=kernel,
        num_burnin_steps=1000,
        num_steps_between_results=1,  # Thinning
        trace_fn=trace_fn,
        seed=123)
    sample, trace_res = res if trace else (res, None)
    return sample, trace_res

# Run MCMC algorithms
t1 = time.time()
mh, trace_mh = run_mcmc(kernel_mh, num_results, init_state, trace=True)
t2 = time.time()
mh = tf.transpose(mh, perm=[1, 0, 2])
print(f"{t2 - t1} seconds")
print(mh.shape)
print(tf.reduce_mean(tf.cast(trace_mh.is_accepted, dtype=tf.float32)))
zsamp_mh = tf.reshape(mh, (-1, latent_dim))

t1 = time.time()
hmc, trace_hmc = run_mcmc(kernel_hmc, num_results, init_state, trace=True)
t2 = time.time()
hmc = tf.transpose(hmc, perm=[1, 0, 2])
print(f"{t2 - t1} seconds")
print(hmc.shape)
print(tf.reduce_mean(tf.cast(trace_hmc.is_accepted, dtype=tf.float32)))
zsamp_hmc = tf.reshape(hmc, (-1, latent_dim))

pts = []
t1 = time.time()
for i in range(100):
    if i % 10 == 0:
        print(i)
    init_state = tf.random.normal(shape=(latent_dim,))
    pt, _ = run_mcmc(kernel_pt, num_results, init_state, trace=False)
    pts.append(pt)
t2 = time.time()
pt = tf.stack(pts, axis=0)
pt = tf.transpose(pt, perm=[1, 0, 2])
print(f"{t2 - t1} seconds")
print(pt.shape)
zsamp_pt = tf.reshape(pt, (-1, latent_dim))

# Plot the generated images
fig = plt.figure(figsize=(18, 6))
sub = fig.add_subplot(131)
xsamp = generator(zsamp_mh, apply_tanh=True) * 0.5 + 0.5
img = xsamp.numpy().transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3). \
        transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
plt.imshow(img)
plt.axis("off")
sub = fig.add_subplot(132)
xsamp = generator(zsamp_hmc, apply_tanh=True) * 0.5 + 0.5
img = xsamp.numpy().transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3). \
        transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
plt.imshow(img)
plt.axis("off")
sub = fig.add_subplot(133)
xsamp = generator(zsamp_pt, apply_tanh=True) * 0.5 + 0.5
img = xsamp.numpy().transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3). \
        transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
plt.imshow(img)
plt.axis("off")
plt.show()

# Save images
methods = ["mh", "hmc", "pt"]
zsamps = [zsamp_mh, zsamp_hmc, zsamp_pt]
for method, zsamp in zip(methods, zsamps):
    fig = plt.figure(figsize=(12, 12))
    sub = fig.add_subplot(111)
    xsamp = generator(zsamp, apply_tanh=True) * 0.5 + 0.5
    img = xsamp.numpy().transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3). \
            transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    plt.savefig(f"{model_path}/face_{method}_gen.png", bbox_inches="tight")
