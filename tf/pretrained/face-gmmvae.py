import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from pretrained.face import Face_Encoder, Face_Generator
from temperflow.scheduler import one_cycle_lr
from temperflow.bijector import SplineCoupling
from temperflow.distribution import TransformedDistribution

import matplotlib.pyplot as plt
import seaborn as sns

# Paths
cache_path = "cache"
model_path = "model"
os.makedirs(cache_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Hyper-parameters
img_size = 64
input_dim = img_size**2 * 3
latent_dim = 100
base_channels = 64
bs = 512

# Load data
np.random.seed(123)
tf.random.set_seed(123)

dat = np.load("pretrained/face.npz")
face_ind = dat["face_ind"]
label_ind = dat["label_ind"]
loader = tf.data.Dataset.from_tensor_slices((face_ind, label_ind)).\
    shuffle(buffer_size=face_ind.shape[0], reshuffle_each_iteration=True).\
    batch(bs)

# Encoder
ncomp = 4
encoder = Face_Encoder(latent_dim=latent_dim, img_size=img_size, ncomp=ncomp, base_channels=base_channels)
encoder.load_params(f"pretrained/face-gmmvae-encoder-pytorch.npz")
# Generator
generator = Face_Generator(latent_dim=latent_dim, base_channels=base_channels)
generator.load_params(f"pretrained/face-gmmvae-generator-pytorch.npz")

# Optimizer
nepoch = 300  # nepoch = 1000 if training from scratch
lr = 0.001
total_steps = nepoch * len(loader)
opt = tf.keras.optimizers.Adam(learning_rate=lr)
params = encoder.trainable_variables + generator.trainable_variables
losses = []
# scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch * len(face_loader), cycle_momentum=False)

# Model parameters
tau2 = 0.1
shift = 1.0
scale = 0.5
post_sd = 0.02
vae_losses = []

mix = tfd.Categorical(probs=tf.constant([10.0, 5.0, 10.0, 10.0]) / 35.0)
loc = np.eye(ncomp, latent_dim, dtype=np.float32)
loc[:ncomp, :ncomp] -= 0.5
loc *= (shift / 0.5)
comp = tfd.Independent(
    distribution=tfd.Normal(
        loc=tf.constant(loc),
        scale=scale * tf.ones(shape=(ncomp, latent_dim))),
    reinterpreted_batch_ndims=1)
prior = tfd.MixtureSameFamily(
    mixture_distribution=mix, components_distribution=comp)

@tf.function(jit_compile=True)
def vae_loss(x, y, seed):
    mean, logvar = encoder(x, y, training=True)
    post = tfd.Independent(
        distribution=tfd.Normal(
            loc=mean, scale=post_sd * tf.ones_like(mean)),
        reinterpreted_batch_ndims=1)
    postz = post.sample(seed=seed)
    kl = tf.math.reduce_mean(post.log_prob(postz) - prior.log_prob(postz))

    # use Bernoulli loss function
    # loglik = xlogp + (1-x)log1p
    #        = x(fz + lpg1p) + log1p - xlog1p
    # fz = generator(postz, apply_sigmoid=False)
    # logp = F.logsigmoid(fz)
    # log1p = logp - fz
    # loglik = torch.mean(x * fz + log1p)
    # loss0 = -loglik * input_dim + kl

    fz = generator(postz, apply_tanh=True, training=True)
    loss = tf.math.reduce_mean(0.5 / tau2 * tf.math.square(fz - x)) * input_dim + kl
    return loss

@tf.function
def train_one_batch(x, y, seed):
    opt.learning_rate = one_cycle_lr(
        opt.iterations, max_lr=tf.constant(lr), total_steps=tf.constant(total_steps))
    with tf.GradientTape() as tape:
        loss = vae_loss(x, y, seed)
    grad = tape.gradient(loss, params)
    opt.apply_gradients(zip(grad, params))
    return loss

t1 = time.time()
for i in range(nepoch):
    tt1 = time.time()
    for batch, (x, y) in loader.enumerate():
        x = tf.constant(x)
        y = tf.constant(y)

        seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
        loss = train_one_batch(x, y, seed)
        loss = loss.numpy()
        vae_losses.append(loss)

        if batch % 50 == 0:
            print(f"epoch {i}, batch {batch}, loss_vae = {loss}")
    tt2 = time.time()
    print(f"epoch {i} finished in {tt2 - tt1} seconds")
t2 = time.time()
print(f"VAE finished in {t2 - t1} seconds")

# Save model
# encoder.save_params(f"{model_path}/face-gmmvae-encoder.npz")
# generator.save_params(f"{model_path}/face-gmmvae-generator.npz")

# Load model (exported from the PyTorch version)
encoder.load_params(f"{model_path}/face-gmmvae-encoder.npz")
generator.load_params(f"{model_path}/face-gmmvae-generator.npz")

def view_data(loader):
    x, y = next(loader.as_numpy_iterator())
    ximg = x[:100] * 0.5 + 0.5
    img = ximg.transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3).\
        transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
    plt.imshow(img)
    plt.axis("off")
    return x, y

def view_recon(x, y):
    xsamp = tf.constant(x[:100])
    ysamp = tf.constant(y[:100])
    mean, logvar = encoder(xsamp, ysamp)
    postz = mean + post_sd * tf.random.normal(shape=mean.shape)
    fz = generator(postz, apply_tanh=True) * 0.5 + 0.5
    img = fz.numpy().transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3). \
        transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
    plt.imshow(img)
    plt.axis("off")
    return fz

def view_gen(distr):
    zsamp = distr.sample(100)
    xsamp = generator(zsamp, apply_tanh=True) * 0.5 + 0.5
    img = xsamp.numpy().transpose(0, 2, 3, 1).reshape(10, img_size * 10, img_size, 3). \
        transpose(1, 0, 2, 3).reshape(img_size * 10, img_size * 10, 3)
    plt.imshow(img)
    plt.axis("off")

def view_pair(zsamp, head=6, bound=2):
    d = pd.DataFrame(zsamp[:, :head].numpy())
    g = sns.pairplot(d, plot_kws={"alpha": 0.2, "s": 10})
    g.set(xlim=(-bound, bound), ylim=(-bound, bound))

# Reconstructed images
np.random.seed(123)
tf.random.set_seed(123)
fig = plt.figure(figsize=(16, 8))
sub = fig.add_subplot(121)
x, y = view_data(loader)
sub = fig.add_subplot(122)
fz = view_recon(x, y)
plt.show()

# Generated images based on prior distribution
np.random.seed(123)
tf.random.set_seed(123)
fig = plt.figure(figsize=(8, 8))
view_gen(prior)
plt.show()



# Flow
np.random.seed(123)
tf.random.set_seed(123)
face_loader = tf.data.Dataset.from_tensor_slices((face_ind, label_ind)).\
    shuffle(buffer_size=face_ind.shape[0], reshuffle_each_iteration=True).\
    batch(bs)
flow = SplineCoupling(
    dim=latent_dim, depth=16, count_bins=32, bound=5.0,
    hlayers=[256, 256])
bijdistr = TransformedDistribution(dim=latent_dim, bijector=flow)

nepoch = 30
lr = 0.001
total_steps = nepoch * len(face_loader)
opt = tf.keras.optimizers.Adam(learning_rate=lr)
params = flow.trainable_variables
loss_flow = []

@tf.function(jit_compile=True)
def flow_loss(x, y, seed):
    mean, logvar = encoder(x, y, training=True)
    post = tfd.Independent(
        distribution=tfd.Normal(
            loc=mean, scale=post_sd * tf.ones_like(mean)),
        reinterpreted_batch_ndims=1)
    postz = post.sample(seed=seed)
    loss = -tf.math.reduce_mean(bijdistr.log_prob(postz))
    return loss

@tf.function
def flow_train_one_batch(x, y, seed):
    opt.learning_rate = one_cycle_lr(
        opt.iterations, max_lr=tf.constant(lr), total_steps=tf.constant(total_steps))
    with tf.GradientTape() as tape:
        loss = flow_loss(x, y, seed)
    grad = tape.gradient(loss, params)
    opt.apply_gradients(zip(grad, params))
    return loss

t1 = time.time()
for i in range(nepoch):
    for batch, (x, y) in enumerate(face_loader):
        x = tf.constant(x)
        y = tf.constant(y)

        seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
        loss = flow_train_one_batch(x, y, seed)
        loss = loss.numpy()
        loss_flow.append(loss)

        if batch % 10 == 0:
            print(f"epoch {i}, batch {batch}, loss = {loss:.3f}")
t2 = time.time()
print(f"finished in {t2 - t1} seconds")

# Save model
bijdistr.save_params(f"{model_path}/face-gmmvae-flow.npz")

# Load model
bijdistr.load_params(f"{model_path}/face-gmmvae-flow.npz")

# Generated images
np.random.seed(123)
tf.random.set_seed(123)
fig = plt.figure(figsize=(8, 8))
view_gen(bijdistr)
plt.show()

def true_sample(bs):
    loader = tf.data.Dataset.from_tensor_slices((face_ind, label_ind)). \
        shuffle(buffer_size=face_ind.shape[0], reshuffle_each_iteration=True). \
        batch(bs)
    x, y = next(loader.as_numpy_iterator())
    x = tf.constant(x)
    y = tf.constant(y)

    mean, logvar = encoder(x, y)
    post = tfd.Independent(
        distribution=tfd.Normal(
            loc=mean, scale=post_sd * tf.ones_like(mean)),
        reinterpreted_batch_ndims=1)
    postz = post.sample()
    return postz

# Marginal distributions
fig = plt.figure(figsize=(12, 6))
zacc = true_sample(5000)
gdat = pd.DataFrame(zacc.numpy(), columns=[f"x{i}" for i in range(zacc.shape[1])])
for i in range(4, zacc.shape[1]):
    sns.kdeplot(data=gdat, x=f"x{i}")
plt.show()

# Pairwise scatterplots
np.random.seed(123)
tf.random.set_seed(123)
# Prior distribution
fig = plt.figure(figsize=(9, 9))
view_pair(prior.sample(1000), head=6, bound=3)
plt.show()
# Aggregated posterior
fig = plt.figure(figsize=(9, 9))
view_pair(true_sample(1000), head=6, bound=3)
plt.show()
# Flow
fig = plt.figure(figsize=(9, 9))
view_pair(bijdistr.sample(1000), head=6, bound=3)
plt.show()
