import time
import os
import math
import json
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
import tensorflow_probability as tfp
tfd = tfp.distributions

import temperflow.option as opts
opts.set_opts(jit=True, debug=False)
from temperflow.bijector import AutoRegSpline
from temperflow.distribution import TransformedDistribution
from temperflow.flow import TemperFlow

from utils import get_distr

# Paths
cache_path = "cache"
model_path = "model"
cache_prefix = f"{cache_path}/clayton-is"
model_prefix = f"{model_path}/clayton-is"
os.makedirs(cache_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# The distribution to be sampled from
np.random.seed(123)
tf.random.set_seed(123)
dim = 8
energy = get_distr("clayton", dim=dim)

# Bijector
np.random.seed(123)
tf.random.set_seed(123)
bijector = AutoRegSpline(
    dim=dim, perm_dim=2, count_bins=32, bound=6.0, hlayers=[64, 64])
bijdistr = TransformedDistribution(dim=dim, bijector=bijector)
temper = TemperFlow(energy, bijdistr)

# Compiling model and warmup
t1 = time.time()
temper.train_one_beta(beta=0.1, fn="kl", bs=2000, nepoch=11, lr=0.0001, verbose=True)
temper.update_ref()
temper.train_one_beta(beta=0.1, fn="l2", ref=True, bs=2000, nepoch=11, lr=0.0001, verbose=True)
t2 = time.time()
print(f"{t2 - t1} seconds")

# Main training procedure
t1 = time.time()
trace = temper.train_flow(beta0=0.1, kl_decay=0.7, nepoch=(1001, 1001, 101),
                          bs=2000, lr=0.001, max_betas=100, verbose=2,
                          checkpoint=cache_prefix, recover=False)
t2 = time.time()
print(f"{t2 - t1} seconds")

# Save model and traces
bijdistr.save_params(f"{model_prefix}-bijdistr.npz")
with open(f"{model_prefix}-losses.json", mode="w") as f:
    json.dump(trace, f, indent=2)

# Restore model and traces
bijdistr.load_params(f"{model_prefix}-bijdistr.npz")
with open(f"{model_prefix}-losses.json", mode="r") as f:
    trace = json.load(f)

# Extract betas
betas = [math.exp(lbeta) for lbeta in trace["logbeta"]]
nbeta = len(betas)

def impl_samp_est(energy, beta, bijdistr, nsamp=1000):
    x, logh = bijdistr.sample(nsamp, log_pdf=True)
    E = energy.energy(x)
    log_ratio = -beta * E - logh
    logz_est = tf.math.reduce_logsumexp(log_ratio) - math.log(nsamp)
    z_est = tf.math.exp(logz_est)
    sd_est = tf.math.reduce_std(tf.math.exp(log_ratio)) * math.sqrt(nsamp / (nsamp - 1.0))
    return logz_est, z_est, sd_est / math.sqrt(nsamp)

# Use learned distribution flow to estimate the normalizing constant
# of p_beta(x), which is proportional to exp(-beta * E(x))
np.random.seed(123)
tf.random.set_seed(123)
logz_est = np.full(shape=(nbeta, nbeta), fill_value=np.nan)
z_est = np.full(shape=(nbeta, nbeta), fill_value=np.nan)
sd_est = np.full(shape=(nbeta, nbeta), fill_value=np.nan)
for i in range(nbeta):
    for j in range(i + 1):
        bijdistr.load_params(f"{cache_prefix}_{i}.npz")
        logz, z, sd = impl_samp_est(energy, betas[j], bijdistr, nsamp=10000)
        print(f"i = {i}, j = {j}, logz = {logz:.3f}, z = {z:.3g}, sd = {sd:.3g}")
        logz_est[i, j] = logz.numpy()
        z_est[i, j] = z.numpy()
        sd_est[i, j] = sd.numpy()

fig = plt.figure(figsize=(9, 8))
plt.plot(range(nbeta), betas)
plt.tick_params(labelsize=15)
plt.ylim(0.0, 1.0)
plt.xlabel("k", fontdict=dict(size=24))
plt.ylabel(r"$\beta$", fontdict=dict(size=24))
plt.title(r"$\beta_k$ selected by TemperFlow", fontdict=dict(size=24))
fig.tight_layout()
plt.show()
plt.savefig(f"{model_path}/is_beta.pdf")

fig = plt.figure(figsize=(9, 8))
ax = sns.heatmap(logz_est)
plt.tick_params(labelsize=15)
# Color bar font size - https://stackoverflow.com/a/53095480
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.xlabel("s", fontdict=dict(size=24))
plt.ylabel("k", fontdict=dict(size=24))
plt.title(r"Estimate of $\log(Z_{\beta_s})$ based on proposal $\hat{r}_k$", fontdict=dict(size=24))
fig.tight_layout()
plt.show()
plt.savefig(f"{model_path}/is_logz.pdf")

fig = plt.figure(figsize=(9, 8))
ax = sns.heatmap(np.log(sd_est / z_est))
plt.tick_params(labelsize=15)
# Color bar font size - https://stackoverflow.com/a/53095480
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.xlabel("s", fontdict=dict(size=24))
plt.ylabel("k", fontdict=dict(size=24))
plt.title(r"$\log[CV(\hat{r}_k,\beta_s)]$", fontdict=dict(size=24))
fig.tight_layout()
plt.show()
plt.savefig(f"{model_path}/is_logcv.pdf")
