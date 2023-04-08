import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import pyro.distributions as dist

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)
torch.manual_seed(123)

# Unimodal distribution
uni = D.TransformedDistribution(
    base_distribution=D.Gamma(torch.tensor([3.0]), torch.tensor([1.0])),
    transforms=[D.transforms.ExpTransform().inv,
                D.transforms.AffineTransform(loc=0.0, scale=3.0)]
)
# Mixture normal
# 0.7 * N(1, 1) + 0.3 * N(8, 0.25)
mix = D.MixtureSameFamily(
    mixture_distribution=D.Categorical(probs=torch.tensor([0.7, 0.3])),
    component_distribution=D.Normal(loc=torch.tensor([1.0, 8.0]),
                                    scale=torch.tensor([1.0, 0.5]))
)
# 0.7 * N(1, 1) + 0.3 * N(6, 0.25)
mix6 = D.MixtureSameFamily(
    mixture_distribution=D.Categorical(probs=torch.tensor([0.7, 0.3])),
    component_distribution=D.Normal(loc=torch.tensor([1.0, 6.0]),
                                    scale=torch.tensor([1.0, 0.5]))
)
# 0.7 * N(1, 1) + 0.3 * N(4, 0.25)
mix4 = D.MixtureSameFamily(
    mixture_distribution=D.Categorical(probs=torch.tensor([0.7, 0.3])),
    component_distribution=D.Normal(loc=torch.tensor([1.0, 4.0]),
                                    scale=torch.tensor([1.0, 0.5]))
)

# Plot density functions
np.random.seed(123)
torch.manual_seed(123)
x0 = torch.linspace(-5.0, 10.0, 300)
denu = torch.exp(uni.log_prob(x0))
denm = torch.exp(mix.log_prob(x0))
denm6 = torch.exp(mix6.log_prob(x0))
denm4 = torch.exp(mix4.log_prob(x0))
dat = pd.DataFrame({"x": x0.numpy(), "denu": denu.numpy(),
                    "denm": denm.numpy(), "denm6": denm6.numpy(),
                    "denm4": denm4.numpy()})
fig = plt.figure(figsize=(12, 10))
sub = fig.add_subplot(221)
sns.lineplot(data=dat, x="x", y="denu")
plt.title("Densify Function")
sub = fig.add_subplot(222)
sns.lineplot(data=dat, x="x", y="denm")
plt.title("Densify Function")
sub = fig.add_subplot(223)
sns.lineplot(data=dat, x="x", y="denm6")
plt.title("Densify Function")
sub = fig.add_subplot(224)
sns.lineplot(data=dat, x="x", y="denm4")
plt.title("Densify Function")
plt.show()

def flow_to_distr(flow):
    base = D.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
    distr = D.TransformedDistribution(
        base_distribution=base,
        transforms=flow)
    return distr

def fit_distr(flow, target, bs=2000, nepoch=2000, lr=0.01):
    distr = flow_to_distr(flow)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    dens = []
    for i in range(nepoch):
        theta = distr.rsample((bs,))
        loss = distr.log_prob(theta) - target.log_prob(theta)
        loss = torch.mean(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 10 == 0:
            print(f"epoch {i}, loss = {loss.item()}")
            den = torch.exp(distr.log_prob(x0.view(-1, 1)))
            dens.append(den.squeeze().detach().numpy())
    return dens

def vis_flow(flow):
    distr = flow_to_distr(flow)
    z = torch.linspace(-5.0, 5.0, 300).reshape(-1, 1)
    den = torch.exp(distr.log_prob(z))
    fz = flow(z)
    pdat = pd.DataFrame({
        "z": z.detach().numpy().squeeze(),
        "den": den.detach().numpy().squeeze(),
        "fz": fz.detach().numpy().squeeze()
    })

    fig = plt.figure(figsize=(12, 5))
    sub = fig.add_subplot(121)
    sns.lineplot(data=pdat, x="z", y="den", label="Estimated density")
    sub = fig.add_subplot(122)
    sns.lineplot(data=pdat, x="z", y="fz", label="Transformation\nFunction")
    plt.show()

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



# Initialization - train the flow to fit a standard normal
np.random.seed(123)
torch.manual_seed(123)
norm = D.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
# Spline flow
spline = dist.transforms.Spline(1, count_bins=100, bound=10.0)
_ = fit_distr(spline, norm, bs=2000, nepoch=2000, lr=0.01)
# Verify that it is close to the standard normal
vis_flow(spline)

# Fit the unimodal distribution
np.random.seed(123)
torch.manual_seed(123)
est = fit_distr(spline, uni, bs=2000, nepoch=501, lr=0.005)

# Visualize the fitted distribution
vis_fitted(spline, uni)

# Set the first density curve to standard normal
est[0] = torch.exp(norm.log_prob(x0.view(-1, 1))).squeeze().detach().numpy()
densu = np.array(est)



# Initialization - train the flow to fit a standard normal
np.random.seed(123)
torch.manual_seed(123)
# Spline flow
spline = dist.transforms.Spline(1, count_bins=100, bound=10.0)
_ = fit_distr(spline, norm, bs=2000, nepoch=2000, lr=0.01)

# Fit the mixture distribution
np.random.seed(123)
torch.manual_seed(123)
est = fit_distr(spline, mix, bs=2000, nepoch=501, lr=0.005)

# Visualize the fitted distribution
vis_fitted(spline, mix)

# Set the first density curve to standard normal
est[0] = torch.exp(norm.log_prob(x0.view(-1, 1))).squeeze().detach().numpy()
densm = np.array(est)



# Initialization - train the flow to fit a standard normal
np.random.seed(123)
torch.manual_seed(123)
# Spline flow
spline = dist.transforms.Spline(1, count_bins=100, bound=10.0)
_ = fit_distr(spline, norm, bs=2000, nepoch=2000, lr=0.01)

# Fit the mixture distribution (gap = 6)
np.random.seed(123)
torch.manual_seed(123)
est = fit_distr(spline, mix6, bs=2000, nepoch=501, lr=0.005)

# Visualize the fitted distribution
vis_fitted(spline, mix6)

# Set the first density curve to standard normal
est[0] = torch.exp(norm.log_prob(x0.view(-1, 1))).squeeze().detach().numpy()
densm6 = np.array(est)



# Initialization - train the flow to fit a standard normal
np.random.seed(123)
torch.manual_seed(123)
# Spline flow
spline = dist.transforms.Spline(1, count_bins=100, bound=10.0)
_ = fit_distr(spline, norm, bs=2000, nepoch=2000, lr=0.01)

# Fit the mixture distribution (gap = 4)
np.random.seed(123)
torch.manual_seed(123)
est = fit_distr(spline, mix4, bs=2000, nepoch=501, lr=0.005)

# Visualize the fitted distribution
vis_fitted(spline, mix4)

# Set the first density curve to standard normal
est[0] = torch.exp(norm.log_prob(x0.view(-1, 1))).squeeze().detach().numpy()
densm4 = np.array(est)



# Save data
np.savez_compressed("cache/fig1-figs1-kl-sampler.npz",
                    x=x0.numpy(),
                    denu=denu.numpy(), densu=densu,
                    denm=denm.numpy(), densm=densm,
                    denm6=denm6.numpy(), densm6=densm6,
                    denm4=denm4.numpy(), densm4=densm4)
