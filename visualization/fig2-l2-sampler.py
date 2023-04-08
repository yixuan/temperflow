import math
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import pyro.distributions as dist

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)
torch.manual_seed(123)

# f(x) = log|exp(x) - 1|
# For x > 0, f(x) = x + log(1 - exp(-x))
# For x < 0, f(x) = log(1 - exp(x))
# Therefore, f(x) = max(x, 0) + log(1 - exp(-|x|))
def log_expm1(x):
    return torch.relu(x) + torch.log(1.000001 - torch.exp(-torch.abs(x)))

# Mixture normal
mix = D.MixtureSameFamily(
    mixture_distribution=D.Categorical(probs=torch.tensor([0.7, 0.3])),
    component_distribution=D.Normal(loc=torch.tensor([-1.0, 8.0]),
                                    scale=torch.tensor([1.0, 0.5]))
)
mix2 = D.MixtureSameFamily(
    mixture_distribution=D.Categorical(probs=torch.tensor([0.9, 0.1])),
    component_distribution=D.Normal(loc=torch.tensor([-1.0, 8.0]),
                                    scale=torch.tensor([1.0, 0.5]))
)

# Plot density function
np.random.seed(123)
torch.manual_seed(123)
fig = plt.figure(figsize=(10, 5))
sub = fig.add_subplot(111)
x0 = torch.linspace(-7.0, 12.0, 300)
den1 = torch.exp(mix.log_prob(x0))
den2 = torch.exp(mix2.log_prob(x0))
dat = pd.DataFrame({"x": x0.numpy(), "den1": den1.numpy(), "den2": den2.numpy()})
sns.lineplot(data=dat, x="x", y="den1", label="Target distribution")
sns.lineplot(data=dat, x="x", y="den2", label="Initial distribution")
plt.title("Density Function")
plt.show()

def copy_spline(spline):
    newsp = dist.transforms.Spline(1, count_bins=spline.count_bins, bound=spline.bound)
    for pnew, pold in zip(newsp.parameters(), spline.parameters()):
        pnew.data = pold.data.clone()
    return newsp

def flow_to_distr(flow):
    base = D.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
    distr = D.TransformedDistribution(
        base_distribution=base,
        transforms=flow)
    return distr

def fit_distr_oracle(flow, target_logpdf, bs=2000, nepoch=2000, lr=0.01):
    distr = flow_to_distr(flow)
    # First stage: fit a tempered distribution
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=nepoch, cycle_momentum=False)
    for i in range(nepoch):
        theta = distr.rsample((bs,))
        loss = distr.log_prob(theta) - 0.1 * target_logpdf(theta)
        loss = torch.mean(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if i % 50 == 0:
            print("epoch {}, loss = {}".format(i, loss.item()))
    # Second stage: fit the target
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=nepoch, cycle_momentum=False)
    for i in range(nepoch):
        theta = distr.rsample((bs,))
        logq = distr.log_prob(theta)
        logp = target_logpdf(theta)
        w = logq + 2.0 * log_expm1(logp - logq)
        loss1 = torch.logsumexp(w, dim=0) - math.log(bs)
        loss2 = torch.mean(logq) - torch.mean(logp)
        loss = loss1 + loss2

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if i % 50 == 0:
            print("epoch {}, loss = {}".format(i, loss.item()))

def fit_distr_kl(flow, target_logpdf, bs=2000, nepoch=2000, lr=0.01):
    distr = flow_to_distr(flow)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    dens = []
    for i in range(nepoch):
        theta = distr.rsample((bs,))
        loss = distr.log_prob(theta) - target_logpdf(theta)
        loss = torch.mean(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 10 == 0:
            print(f"epoch {i}, loss = {loss.item()}")
            den = torch.exp(distr.log_prob(x0.view(-1, 1)))
            dens.append(den.squeeze().detach().numpy())
    return dens

def fit_distr_l2(flow, target_logpdf, bs=2000, nepoch=2000, lr=0.01):
    distr = flow_to_distr(flow)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    dens = []
    for i in range(nepoch):
        theta = distr.rsample((bs,))
        logq = distr.log_prob(theta)
        unnormp = target_logpdf(theta)
        logz = torch.logsumexp(unnormp - logq, dim=0) - math.log(bs)
        logp = unnormp - logz
        w = logq + 2.0 * log_expm1(logp - logq)
        loss = torch.logsumexp(w, dim=0) - math.log(bs)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 10 == 0:
            print(f"epoch {i}, loss = {loss.item()}")
            den = torch.exp(distr.log_prob(x0.view(-1, 1)))
            dens.append(den.squeeze().detach().numpy())
    return dens

# Visualize the fitted distribution
def vis_fitted(flow, target):
    distr = flow_to_distr(flow)
    z0 = x0.reshape(-1, 1)
    fz0 = torch.exp(target.log_prob(z0))
    fz1 = torch.exp(distr.log_prob(z0))
    pdat = pd.DataFrame({
        "z": z0.detach().numpy().squeeze(),
        "fz0": fz0.detach().numpy().squeeze(),
        "fz1": fz1.detach().numpy().squeeze()
    })
    fig = plt.figure(figsize=(15, 5))
    sub = fig.add_subplot(111)
    sns.lineplot(data=pdat, x="z", y="fz0", label="True density")
    sns.lineplot(data=pdat, x="z", y="fz1", label="Estimated density")
    plt.show()

# Train the flow to fit the second distribution
np.random.seed(123)
torch.manual_seed(123)
spline = dist.transforms.Spline(1, count_bins=200, bound=15.0)
fit_distr_oracle(spline, mix2.log_prob, bs=5000, nepoch=2000, lr=0.01)

# Check the fitted distribution
distr = flow_to_distr(spline)
fig = plt.figure(figsize=(12, 5))
sub = fig.add_subplot(121)
z0 = torch.linspace(-8.0, 15.0, 500).reshape(-1, 1)
pdat = pd.DataFrame({
    "z": z0.detach().numpy().squeeze(),
    "den2": torch.exp(mix2.log_prob(z0)).detach().numpy().squeeze(),
    "den_est": torch.exp(distr.log_prob(z0)).detach().numpy().squeeze()
})
sns.lineplot(data=pdat, x="z", y="den2", label="True mix2 density")
sns.lineplot(data=pdat, x="z", y="den_est", label="Estimated mix2 density")
sub = fig.add_subplot(122)
z0 = torch.linspace(-8.0, 15.0, 500).reshape(-1, 1)
fz0 = spline(z0)
pdat = pd.DataFrame({
    "z": z0.detach().numpy().squeeze(),
    "fz0": fz0.detach().numpy().squeeze()
})
sns.lineplot(data=pdat, x="z", y="fz0", label="Transformation\nFunction")
plt.show()



# Fit the target distribution using KL
np.random.seed(123)
torch.manual_seed(123)
spline_kl = copy_spline(spline)
dens_kl = fit_distr_kl(spline_kl, mix.log_prob, bs=5000, nepoch=1001, lr=0.001)
dens_kl[0] = torch.exp(mix2.log_prob(x0.view(-1, 1))).squeeze().detach().numpy()
dens_kl = np.array(dens_kl)

vis_fitted(spline_kl, mix)



# Fit the target distribution using L2
np.random.seed(123)
torch.manual_seed(123)
spline_l2 = copy_spline(spline)
dens_l2 = fit_distr_l2(spline_l2, mix.log_prob, bs=5000, nepoch=1001, lr=0.001)
dens_l2[0] = torch.exp(mix2.log_prob(x0.view(-1, 1))).squeeze().detach().numpy()
dens_l2 = np.array(dens_l2)

vis_fitted(spline_l2, mix)



# Save data
np.savez_compressed("cache/fig2-l2-sampler.npz",
                    x=x0.detach().numpy(), den=den1.detach().numpy(),
                    dens_kl=dens_kl, dens_l2=dens_l2)
