import time
import math
import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.distributions as D
from pyro.nn import DenseNN

from pretrained.mnist import MNIST_Encoder, MNIST_Generator
from temperflow.bijector import BijectorAutoRegSpline
from temperflow.flow import Energy, TemperFlow

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

cuda = torch.device("cuda")
cpu = torch.device("cpu")
device = cuda if torch.cuda.is_available() else cpu
print(device)


np.random.seed(123)
torch.manual_seed(123)

input_dim = 784
latent_dim = 32
base_channels = 64

# Paths
cache_path = "cache"
model_path = "model"

# Generator
generator = MNIST_Generator(latent_dim=latent_dim, base_channels=base_channels)
generator = generator.to(device=device)
generator = torch.jit.script(generator)
generator.load_state_dict(torch.load(f"pretrained/fmnist-vae-rd-generator.pt"))
generator.eval()
# Discriminator
discr = DenseNN(input_dim=latent_dim, hidden_dims=[256, 256], param_dims=[1],
                nonlinearity=nn.ReLU(inplace=True))
discr = discr.to(device=device)
discr.load_state_dict(torch.load(f"pretrained/fmnist-vae-rd-discr-z.pt"))
discr.eval()

# Energy function for sampling
class MNISTLatentEnergy(Energy):
    def __init__(self, generator, discriminator):
        super(MNISTLatentEnergy, self).__init__()
        self.generator = generator
        self.discr = discriminator
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.discr.parameters():
            param.requires_grad = False

    def energy(self, x, **kwargs):
        z = x
        logpz = -0.5 * torch.sum(torch.square(z), dim=1) - 0.5 * math.log(2.0 * math.pi) * z.shape[1]
        dz = self.discr(z).squeeze()
        return -logpz - dz

    def log_pdf(self, x, args):
        z = torch.tensor(x, device=device)
        with torch.no_grad():
            logp = -self.energy(z)
        return logp.cpu().numpy()

    def log_pdf_grad(self, x, args):
        z = torch.tensor(x, requires_grad=True, device=device)
        Gz = self.generator(z, apply_sigmoid=True)
        dGz = self.discr(Gz).squeeze()
        logp = -0.5 * torch.sum(torch.square(z)) + torch.sum(dGz)
        logp.backward()
        return z.grad.cpu().numpy()

energy = MNISTLatentEnergy(generator, discr)



# Rejection sampling
np.random.seed(123)
torch.random.manual_seed(123)
proposal = D.Independent(
    D.StudentT(
        df=5,
        loc=torch.zeros(latent_dim, device=device),
        scale=torch.ones(latent_dim, device=device)),
    reinterpreted_batch_ndims=1
)
# p(x) <= M * q(x)
# Also estimate the partition function
# z = integrate exp(-E(x)) dx
#   = E_q exp(-E(x) - logq(x))
# log(z) = log_sum_exp(-E(x) - logq(x)) - log(n), x ~ q(x)
zacc = []
logz = []
with torch.no_grad():
    bs = 50000
    niter = 100
    for i in range(niter):
        zprop = proposal.sample((bs,))
        logq = proposal.log_prob(zprop)
        e = energy(zprop, 1.0)
        log_ratio = -e - logq
        logz.append(log_ratio.cpu().numpy())
        logM = torch.max(log_ratio)
        logM = logM + math.log(1.2)
        log_acc_prob = -e - logq - logM
        acc = (torch.rand(bs, device=device).log() < log_acc_prob).cpu().numpy()
        print(np.sum(acc))
        zprop = zprop.cpu().numpy()
        zacc.append(zprop[acc, :])
zacc = np.vstack(zacc)
logz = np.hstack(logz)
logz = logsumexp(logz) - math.log(logz.size)
print(zacc.shape)
print(logz)

# Plot the generated images
np.random.seed(123)
torch.random.manual_seed(123)
with torch.no_grad():
    xacc = generator(torch.tensor(zacc[:100], device=device), apply_sigmoid=True).cpu().numpy()
fig = plt.figure(figsize=(8, 8))
sub = fig.add_subplot(111)
pic = xacc.reshape(10, 280, 28).transpose(1, 0, 2).reshape(280, 280)
pic = np.clip(pic, 0.0, 1.0)
sub.axes.get_xaxis().set_visible(False)
sub.axes.get_yaxis().set_visible(False)
sub.imshow(pic, cmap="gray")
plt.show()
plt.savefig(f"{model_path}/fmnist_rd_gen.pdf", bbox_inches="tight")



# TemperFlow
np.random.seed(123)
torch.random.manual_seed(123)
bijector = BijectorAutoRegSpline(
    dim=latent_dim, count_bins=64, bound=8.0, hlayers=[256, 256],
    nonlinearity=nn.ReLU(inplace=True), device=device
)
temper = TemperFlow(energy, bijector)

t1 = time.time()
trace = temper.train_flow(beta0=0.1, kl_decay=0.7, nepoch=(2001, 2001, 2001),
                          bs=2000, lr=0.001, max_betas=100, verbose=2)
t2 = time.time()
print(f"{t2 - t1} seconds")
torch.save(temper.state_dict(), f"{model_path}/fmnist-rd-l2.pt")
# temper.load_state_dict(torch.load(f"{model_path}/fmnist-rd-l2.pt"))

# Plot the generated images
np.random.seed(123)
torch.random.manual_seed(123)
with torch.no_grad():
    zsamp = bijector.distr.sample((100,))
    xsamp = generator(zsamp, apply_sigmoid=True).cpu().numpy()
fig = plt.figure(figsize=(8, 8))
sub = fig.add_subplot(111)
pic = xsamp.reshape(10, 280, 28).transpose(1, 0, 2).reshape(280, 280)
pic = np.clip(pic, 0.0, 1.0)
sub.axes.get_xaxis().set_visible(False)
sub.axes.get_yaxis().set_visible(False)
sub.imshow(pic, cmap="gray")
plt.show()
plt.savefig(f"{model_path}/fmnist_rd_l2_gen.pdf", bbox_inches="tight")



# Measure transport via KL
np.random.seed(123)
torch.random.manual_seed(123)
bij_kl = BijectorAutoRegSpline(
    dim=latent_dim, perm_dim=2, count_bins=64, bound=8.0, hlayers=[256, 256],
    nonlinearity=nn.ReLU(inplace=True), device=device
)
flow_kl = TemperFlow(energy, bij_kl)

t1 = time.time()
losses = flow_kl.train_one_beta(beta=1.0, fn="kl", ref_bij=None, bs=2000, nepoch=5001, lr=0.001, verbose=True)
t2 = time.time()
print(f"{t2 - t1} seconds")
torch.save(flow_kl.state_dict(), f"{model_path}/fmnist-rd-kl.pt")
# flow_kl.load_state_dict(torch.load(f"{model_path}/fmnist-rd-kl.pt"))

# Plot the generated images
np.random.seed(123)
torch.random.manual_seed(123)
with torch.no_grad():
    zsamp = bij_kl.distr.sample((100,))
    xsamp = generator(zsamp, apply_sigmoid=True).cpu().numpy()
fig = plt.figure(figsize=(8, 8))
sub = fig.add_subplot(111)
pic = xsamp.reshape(10, 280, 28).transpose(1, 0, 2).reshape(280, 280)
pic = np.clip(pic, 0.0, 1.0)
sub.axes.get_xaxis().set_visible(False)
sub.axes.get_yaxis().set_visible(False)
sub.imshow(pic, cmap="gray")
plt.show()
plt.savefig(f"{model_path}/fmnist_rd_kl_gen.pdf", bbox_inches="tight")
