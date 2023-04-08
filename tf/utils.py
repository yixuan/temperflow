# Utility functions for simulation
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from density import GaussianCircle, GaussianGrid, GaussianCross, ClaytonNormalMix

# Get the distribution to sample from
def get_distr(data, dim=None):
    if data == "circle":
        energy = GaussianCircle(modes=8, radius=5.0, scale=0.5)
    elif data == "grid":
        energy = GaussianGrid(num_on_edge=5, width=10.0, scale=0.3)
    elif data == "cross":
        energy = GaussianCross(scale=1.0, rho=0.9)
    elif data == "clayton":
        energy = ClaytonNormalMix(dim=dim, mix_dim=8, theta=2.0)
    else:
        raise NotImplementedError

    return energy

# Conditionally run experiments
def cond_run_experiment(descr, cond, fun):
    print(f"\n** {descr} **")
    if cond:
        fun()
    else:
        print("** Skipped...\n")

# Visualize 2D energy function
def vis_energy(energy, xlim=(-8.0, 8.0), ylim=(-8.0, 8.0)):
    fig = plt.figure(figsize=(20, 10))
    sub = fig.add_subplot(121)
    ngrid = 100
    z1 = np.linspace(*xlim, num=ngrid)
    z2 = np.linspace(*ylim, num=ngrid)
    zv1, zv2 = np.meshgrid(z1, z2)
    grids = np.hstack((zv1.reshape(-1, 1), zv2.reshape(-1, 1))).astype(np.float32)
    grids = tf.constant(grids)
    fn = -energy(grids, beta=1.0)
    plt.contourf(zv1, zv2, fn.numpy().reshape(ngrid, ngrid), levels=15)
    plt.colorbar()
    sub = fig.add_subplot(122)
    samp = energy.sample(1000)
    dat = pd.DataFrame(samp.numpy(), columns=["x1", "x2"])
    sns.scatterplot(data=dat, x="x1", y="x2")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.show()

# Visualize traces
def vis_trace(trace):
    losses = [trace["loss_init"]] + trace["loss"]
    nbeta = min(30, len(losses))
    fig = plt.figure(figsize=(20, 10))
    for i in range(nbeta):
        sub = fig.add_subplot(5, 6, i + 1)
        iter = np.arange(len(losses[i])).reshape((-1, 1))
        lossesi = np.array(losses[i]).reshape((-1, 1))
        dat = pd.DataFrame(np.hstack((iter, lossesi)), columns=["iter", "loss"])
        sns.lineplot(data=dat, x="iter", y="loss")
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    plt.show()

# Rejection sampling
def rej_sampling(energy, distr, nsamp=1000, bs=5000, maxit=100, bound_factor=1.5, logz=False, verbose=True):
    # p(x) <= M * q(x)
    xacc = []
    logr = []
    nacc = 0
    is_tfd = isinstance(distr, tfd.Distribution)
    for i in range(maxit):
        print(i) if verbose else None
        # Ordinary Tensorflow distribution
        if is_tfd:
            xprop = distr.sample(bs)
            logp = distr.log_prob(xprop)
        # TransformedDistribution
        else:
            seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
            xprop, logp = distr.sample(bs, log_pdf=True, seed=seed)
        e = energy(xprop, 1.0)
        log_ratio = -e - logp
        logr.append(log_ratio)
        if is_tfd:
            logM = tf.math.reduce_max(log_ratio)
        else:
            logM = tf.math.reduce_logsumexp(log_ratio, axis=0) - math.log(bs)  # logz
        logM = logM + math.log(bound_factor)
        log_acc_prob = -e - logp - logM
        acc = (tf.math.log(tf.random.uniform([bs])) < log_acc_prob).numpy()
        nacc += np.sum(acc)
        xprop = xprop.numpy()
        xacc.append(xprop[acc, :])
        if nacc >= nsamp:
            break
    xacc = np.vstack(xacc)
    print(xacc.shape) if verbose else None
    xacc = xacc[:nsamp, :]
    if logz:
        logr = tf.concat(logr, axis=0)
        logz = tf.math.reduce_logsumexp(logr, axis=0) - math.log(logr.shape[0])
        return xacc, logz.numpy()
    return xacc

# Visualize sampling results
def vis_sampling_res(bijdistr, trace, xacc, nsamp=5000,
                     xlim=(-8.0, 8.0), ylim=(-8.0, 8.0)):
    fig = plt.figure(figsize=(25, 5))
    # Plot 1: Estimated log density
    sub = fig.add_subplot(151)
    ngrid = 100
    z1 = np.linspace(*xlim, num=ngrid)
    z2 = np.linspace(*ylim, num=ngrid)
    zv1, zv2 = np.meshgrid(z1, z2)
    grids = np.hstack((zv1.reshape(-1, 1), zv2.reshape(-1, 1))).astype(np.float32)
    grids = tf.constant(grids)
    logden = bijdistr.log_prob(grids)
    logden = tf.clip_by_value(logden, -30.0, 10.0)
    plt.contourf(zv1, zv2, logden.numpy().reshape(ngrid, ngrid), levels=15)
    plt.colorbar()
    # Plot 2: Curve of log(beta)
    sub = fig.add_subplot(152)
    sub.plot(trace["logbeta"])
    # Plot 3: Curve of KL divergence
    sub = fig.add_subplot(153)
    sub.plot(trace["kl"])
    # Plot 4: scatterplot of generated points
    sub = fig.add_subplot(154)
    seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
    xsamp = bijdistr.sample(nsamp, seed=seed)
    xsamp = pd.DataFrame(data=xsamp.numpy(), columns=["X1", "X2"])
    sns.scatterplot(data=xsamp, x="X1", y="X2", s=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    # Plot 4: scatterplot of generated points after rejection sampling
    sub = fig.add_subplot(155)
    dat = pd.DataFrame(data=xacc[:nsamp, :], columns=["X1", "X2"])
    sns.scatterplot(data=dat, x="X1", y="X2", s=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.show()

def vis_sampling_res_simple(bijdistr, nsamp=5000, xlim=(-8.0, 8.0), ylim=(-8.0, 8.0), xlab="", ylab=""):
    fig = plt.figure(figsize=(5, 11.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.05, 1.0])
    # Plot 1: Estimated log density
    sub = plt.subplot(gs[0])
    ngrid = 100
    z1 = np.linspace(*xlim, num=ngrid)
    z2 = np.linspace(*ylim, num=ngrid)
    zv1, zv2 = np.meshgrid(z1, z2)
    grids = np.hstack((zv1.reshape(-1, 1), zv2.reshape(-1, 1))).astype(np.float32)
    grids = tf.constant(grids)
    logden = bijdistr.log_prob(grids)
    logden = tf.clip_by_value(logden, -30.0, 10.0)
    plt.contourf(zv1, zv2, logden.numpy().reshape(ngrid, ngrid), levels=15)
    # plt.colorbar()
    plt.title("Estimated Log-density", fontdict=dict(size=18))
    # Plot 2: scatterplot of generated points
    sub = plt.subplot(gs[1])
    seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
    xsamp = bijdistr.sample(nsamp, seed=seed).numpy()
    plt.scatter(xsamp[:, 0], xsamp[:, 1], s=3)
    # xsamp = pd.DataFrame(data=xsamp.numpy(), columns=["X1", "X2"])
    # sns.scatterplot(data=xsamp, x="X1", y="X2", s=5)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel(xlab, fontdict=dict(size=18))
    plt.ylabel(ylab)
    plt.title("Randomly Generated Points", fontdict=dict(size=18))
    plt.show()
