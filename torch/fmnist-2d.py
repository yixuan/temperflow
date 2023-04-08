import time
import math
import numpy as np
from scipy.special import logsumexp
import pandas as pd
import torch
import torch.distributions as D

from torchvision import datasets
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from pretrained.mnist import MNIST_Generator, MNIST_Discriminator
from temperflow.bijector import Bijector2D
from temperflow.flow import Energy, TemperFlow

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import seaborn as sns

cuda = torch.device("cuda")
cpu = torch.device("cpu")
device = cuda if torch.cuda.is_available() else cpu
print(device)


np.random.seed(123)
torch.manual_seed(123)

input_dim = 784
latent_dim = 2
base_channels = 64

# Paths
cache_path = "cache"
model_path = "model"

# Generator
generator = MNIST_Generator(latent_dim=latent_dim, base_channels=base_channels)
generator = generator.to(device=device)
generator.load_state_dict(torch.load(f"pretrained/fmnist-vae-2d-generator.pt"))
generator.eval()
# Discriminator
discr = MNIST_Discriminator(base_channels=base_channels // 2)
discr = discr.to(device=device)
discr.load_state_dict(torch.load(f"pretrained/fmnist-vae-2d-discr.pt"))
discr.eval()

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
        Gz = self.generator(z, apply_sigmoid=True)
        dGz = self.discr(Gz).squeeze()
        return -logpz - dGz

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
    niter = 10
    for i in range(niter):
        zprop = proposal.sample((bs,))
        logq = proposal.log_prob(zprop)  # .clip_(-30.0, 10.0)
        e = energy(zprop, 1.0)
        log_ratio = -e - logq
        logz.append(log_ratio.cpu().numpy())
        logM = torch.max(log_ratio)
        logM = logM + math.log(2.0)
        log_acc_prob = -e - logq - logM
        acc = (torch.rand(bs, device=device).log() < log_acc_prob).cpu().numpy()
        print(np.sum(acc))
        zprop = zprop.cpu().numpy()
        zacc.append(zprop[acc, :])
zacc = np.vstack(zacc)
logz = np.hstack(logz)
logz = logsumexp(logz) - math.log(logz.size)
print(zacc.shape)

# Testing data
np.random.seed(123)
torch.manual_seed(123)
fmnist_test = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
loader = DataLoader(fmnist_test, batch_size=100, shuffle=True)
x, _ = next(iter(loader))
fig = plt.figure(figsize=(8, 8))
sub = fig.add_subplot(111)
xsamp = x.view(-1, input_dim).to(device=device)
img = make_grid(xsamp.view(-1, 1, 28, 28), nrow=10)[0, :, :]
plt.imshow(img.detach().cpu().numpy(), cmap="gray")
plt.axis("off")
plt.show()
plt.savefig(f"{model_path}/fmnist100.pdf", bbox_inches="tight")
fig = plt.figure(figsize=(12, 5))
sub = fig.add_subplot(111)
xsamp = x[:60].view(-1, input_dim).to(device=device)
img = make_grid(xsamp.view(-1, 1, 28, 28), nrow=12)[0, :, :]
plt.imshow(img.detach().cpu().numpy(), cmap="gray")
plt.axis("off")
plt.show()
plt.savefig(f"{model_path}/fmnist60.pdf", bbox_inches="tight")
with torch.no_grad():
    xacc = generator(torch.tensor(zacc[:60], device=device), apply_sigmoid=True)
fig = plt.figure(figsize=(12, 5))
sub = fig.add_subplot(111)
img = make_grid(xacc.view(-1, 1, 28, 28), nrow=12)[0, :, :]
plt.imshow(img.detach().cpu().numpy(), cmap="gray")
plt.axis("off")
plt.show()
plt.savefig(f"{model_path}/fmnist_2d_gen60.pdf", bbox_inches="tight")

# Visualize the energy function
ngrid = 100
z1 = np.linspace(-4.0, 4.0, num=ngrid, dtype=np.float32)
zv1, zv2 = np.meshgrid(z1, z1)
grids = np.hstack((zv1.reshape(-1, 1), zv2.reshape(-1, 1)))
grids = torch.tensor(grids, device=device)
with torch.no_grad():
    efun = energy.energy(grids).cpu().numpy().reshape(ngrid, ngrid) + logz
# efun = np.exp(-efun)
fig = plt.figure(figsize=(16, 8))
sub = fig.add_subplot(121)
plt.contourf(zv1, zv2, -efun, levels=25)
# plt.colorbar()
sub = fig.add_subplot(122)
dat = pd.DataFrame(zacc, columns=["Z1", "Z2"])
sns.scatterplot(data=dat, x="Z1", y="Z2", s=1)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

# Save data
dendat = np.hstack((grids.cpu().numpy(), -efun.reshape(-1, 1)))
dendat = pd.DataFrame(dendat, columns=["Z1", "Z2", "density"])
dendat.to_csv(f"{model_path}/fmnist-2d-logp.csv", index=False)
sdat = dat[:10000]
sdat.to_csv(f"{model_path}/fmnist-2d-scatter.csv", index=False)



# Measure transport via KL
np.random.seed(123)
torch.random.manual_seed(123)
bij_kl = Bijector2D(count_bins=50, bound=None, hlayers=[32, 32], device=device)
flow_kl = TemperFlow(energy, bij_kl)

t1 = time.time()
losses = flow_kl.train_one_beta(beta=1.0, fn="kl", bs=2000, nepoch=5001, lr=0.001, verbose=True)
t2 = time.time()
print(f"{t2 - t1} seconds")
torch.save(flow_kl.state_dict(), f"{model_path}/fmnist-2d-kl.pt")
# flow_kl.load_state_dict(torch.load(f"{model_path}/fmnist-2d-kl.pt"))

# Scatterplot
np.random.seed(123)
torch.random.manual_seed(123)
# fig = plt.figure(figsize=(8, 5))
# plt.plot(losses)
fig = plt.figure(figsize=(16, 8))
sub = fig.add_subplot(121)
logden = bij_kl.distr.log_prob(grids)  # .clip_(-30.0, 10.0)
plt.contourf(zv1, zv2, logden.detach().view(ngrid, ngrid).cpu().numpy(), levels=25)
# plt.colorbar()
sub = fig.add_subplot(122)
xsamp = bij_kl.distr.sample((20000,)).detach().cpu().numpy()
xsamp = pd.DataFrame(data=xsamp, columns=["Z1", "Z2"])
sns.scatterplot(data=xsamp, x="Z1", y="Z2", s=1)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

# Save data
dendat = np.hstack((grids.cpu().numpy(), logden.detach().cpu().numpy().reshape(-1, 1)))
dendat = pd.DataFrame(dendat, columns=["Z1", "Z2", "density"])
dendat.to_csv(f"{model_path}/fmnist-2d-logp-kl.csv", index=False)
sdat = xsamp[:10000]
sdat.to_csv(f"{model_path}/fmnist-2d-scatter-kl.csv", index=False)



# TemperFlow
np.random.seed(123)
torch.random.manual_seed(123)
bijector = Bijector2D(count_bins=50, bound=None, hlayers=[32, 32], device=device)
temper = TemperFlow(energy, bijector)

t1 = time.time()
trace = temper.train_flow(beta0=0.1, kl_decay=0.6, nepoch=(2001, 2001, 2001),
                          bs=2000, lr=0.001, max_betas=100, verbose=2)
t2 = time.time()
print(f"{t2 - t1} seconds")
torch.save(temper.state_dict(), f"{model_path}/fmnist-2d-l2.pt")
# temper.load_state_dict(torch.load(f"{model_path}/fmnist-2d-l2.pt"))

# Scatterplot
np.random.seed(123)
torch.random.manual_seed(123)
fig = plt.figure(figsize=(16, 8))
sub = fig.add_subplot(121)
logden = bijector.distr.log_prob(grids)
plt.contourf(zv1, zv2, logden.detach().view(ngrid, ngrid).cpu().numpy(), levels=25)
# plt.colorbar()
sub = fig.add_subplot(122)
xsamp = bijector.distr.sample((20000,)).detach().cpu().numpy()
xsamp = pd.DataFrame(data=xsamp, columns=["Z1", "Z2"])
sns.scatterplot(data=xsamp, x="Z1", y="Z2", s=1)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

# Save data
dendat = np.hstack((grids.cpu().numpy(), logden.detach().cpu().numpy().reshape(-1, 1)))
dendat = pd.DataFrame(dendat, columns=["Z1", "Z2", "density"])
dendat.to_csv(f"{model_path}/fmnist-2d-logp-l2.csv", index=False)
sdat = xsamp[:10000]
sdat.to_csv(f"{model_path}/fmnist-2d-scatter-l2.csv", index=False)
