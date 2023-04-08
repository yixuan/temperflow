import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import pyro.distributions as dist

from temperflow.block import MACNet, MaskedAffineCoupling, Dense, SplineAutoReg,\
    SplineBlockAutoReg, SplineAutoRegWithinBlock, SplineCoupling, Spline2D

# Transformation mapping based on masked affine coupling layers
class BijectorMAC(nn.Module):
    def __init__(
        self, dim, depth=10, hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True),
        device=torch.device("cpu")
    ):
        super(BijectorMAC, self).__init__()
        self.args = (dim, depth, hlayers, nonlinearity, device)
        # The neural networks in the affine coupling layer
        # Put into a module list to register the parameters
        self.ac_nets = nn.ModuleList()
        # The list of transformations
        self.flow = []
        # Compute permutation masks
        masks = []
        for i in range(depth):
            keep_dim = dim // 2
            if dim == 2:
                mask = [i % 2, (i + 1) % 2]
            else:
                mask = np.zeros(dim)
                mask[:keep_dim] = 1.0
                np.random.shuffle(mask)
            masks.append(mask)
        self.masks = nn.Parameter(torch.tensor(masks, device=device), requires_grad=False)

        for i in range(depth):
            # Masked affine coupling transformation
            hypernet = MACNet(dim, hlayers, dim, nonlinearity, device=device)
            # Compile the module
            hypernet = torch.jit.script(hypernet)
            self.ac_nets.append(hypernet)
            ac = MaskedAffineCoupling(self.masks[i], hypernet, device=device)
            self.flow.append(ac)

        # Base distribution
        base = D.Independent(
            D.Normal(loc=torch.zeros(dim, device=device), scale=torch.ones(dim, device=device)),
            reinterpreted_batch_ndims=1)
        # Transformed distribution
        self.distr = dist.TransformedDistribution(
            base_distribution=base,
            transforms=self.flow)

    def clone(self):
        # Save current model parameters to a temporary file
        temp_file = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        torch.save(self.state_dict(), temp_file)
        # Create a new model object
        new_bij = BijectorMAC(*self.args)
        new_bij.load_state_dict(torch.load(temp_file))
        new_bij.eval()
        # Cleanup
        os.unlink(temp_file)
        return new_bij

    def turn_off_grad(self):
        for param in self.parameters():
            param.requires_grad = False

# Transformation mapping based on spline coupling layers
class BijectorSC(nn.Module):
    def __init__(
        self, dim, depth=10, count_bins=8, bound=None, hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True),
        device=torch.device("cpu")
    ):
        super(BijectorSC, self).__init__()
        self.args = (dim, depth, count_bins, bound, hlayers, nonlinearity, device)
        # The neural networks in the affine coupling layer
        # Put into a module list to register the parameters
        self.ac_nets = nn.ModuleList()
        # The list of transformations
        self.flow = []
        # Compute permutation indices
        perms = []
        for i in range(depth):
            perm = [1, 0] if dim == 2 else np.random.permutation(dim)
            perms.append(perm)
        perms = np.array(perms)
        self.perms = nn.Parameter(torch.tensor(perms, dtype=torch.long, device=device), requires_grad=False)

        for i in range(depth):
            # Permutation transformation
            self.flow.append(dist.transforms.Permute(self.perms[i]))
            # Spline coupling transformation
            keep_dim = dim // 2
            rem_dim = dim - keep_dim
            param_dim = rem_dim * (4 * count_bins - 1)
            hypernet = Dense(keep_dim, hlayers, param_dim, nonlinearity, device)
            hypernet = torch.jit.script(hypernet)
            self.ac_nets.append(hypernet)
            ac = SplineCoupling(dim, keep_dim, hypernet, count_bins, bound).to(device=device)
            self.flow.append(ac)

        # Base distribution
        base = D.Independent(
            D.Normal(loc=torch.zeros(dim, device=device), scale=torch.ones(dim, device=device)),
            reinterpreted_batch_ndims=1)
        # Transformed distribution
        self.distr = dist.TransformedDistribution(
            base_distribution=base,
            transforms=self.flow)

    def clone(self):
        # Save current model parameters to a temporary file
        temp_file = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        torch.save(self.state_dict(), temp_file)
        # Create a new model object
        new_bij = BijectorSC(*self.args)
        new_bij.load_state_dict(torch.load(temp_file))
        new_bij.eval()
        # Cleanup
        os.unlink(temp_file)
        return new_bij

# Transformation mapping based on 2D spline
class Bijector2D(nn.Module):
    def __init__(
        self, count_bins=8, bound=None, order="linear", hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True),
        device=torch.device("cpu")
    ):
        super(Bijector2D, self).__init__()
        self.args = (count_bins, bound, order, hlayers, nonlinearity, device)
        self.spline = Spline2D(count_bins=count_bins, bound=bound, order=order, hlayers=hlayers,
                               nonlinearity=nonlinearity, device=device)

        # Base distribution
        base = D.Independent(
            D.Normal(loc=torch.zeros(2, device=device), scale=torch.ones(2, device=device)),
            reinterpreted_batch_ndims=1)
        # Transformed distribution
        self.distr = dist.TransformedDistribution(
            base_distribution=base,
            transforms=self.spline)

    def clone(self):
        # Save current model parameters to a temporary file
        temp_file = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        torch.save(self.state_dict(), temp_file)
        # Create a new model object
        new_bij = Bijector2D(*self.args)
        new_bij.load_state_dict(torch.load(temp_file))
        new_bij.eval()
        # Cleanup
        os.unlink(temp_file)
        return new_bij

    def turn_off_grad(self):
        for param in self.parameters():
            param.requires_grad = False

# Transformation mapping based on autoregressive splines
class BijectorAutoRegSpline(nn.Module):
    def __init__(
        self, dim, perm_dim=None, count_bins=8, bound=None, order="linear", hlayers=[32, 32],
        nonlinearity=nn.ReLU(inplace=True), device=torch.device("cpu")
    ):
        super(BijectorAutoRegSpline, self).__init__()
        self.args = (dim, perm_dim, count_bins, bound, order, hlayers, nonlinearity, device)

        if perm_dim is None:
            self.spline1 = SplineAutoReg(
                dim, partial_dim=None, count_bins=count_bins, bound=bound, hlayers=hlayers,
                nonlinearity=nonlinearity, first_identity=False, device=device)
            transforms = [self.spline1.inv]
        else:
            self.spline1 = SplineAutoReg(
                dim, partial_dim=None, count_bins=count_bins, bound=bound, hlayers=hlayers,
                nonlinearity=nonlinearity, first_identity=True, device=device)
            partial_dim = min(perm_dim, dim)
            partial_dim = max(2, partial_dim)
            perm = torch.arange(dim, device=device)
            perm[:partial_dim] = torch.arange(partial_dim, device=device).flip(0)
            self.perm = dist.transforms.Permute(perm)
            self.spline2 = SplineAutoReg(
                dim, partial_dim=partial_dim, count_bins=count_bins, bound=bound, hlayers=hlayers,
                nonlinearity=nonlinearity, first_identity=True, device=device)
            transforms = [self.spline1.inv, self.perm, self.spline2.inv]

        # Base distribution
        base = D.Independent(
            D.Normal(loc=torch.zeros(dim, device=device), scale=torch.ones(dim, device=device)),
            reinterpreted_batch_ndims=1)
        # Transformed distribution
        self.distr = dist.TransformedDistribution(
            base_distribution=base,
            transforms=transforms)

    def clone(self):
        # Save current model parameters to a temporary file
        temp_file = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        torch.save(self.state_dict(), temp_file)
        # Create a new model object
        new_bij = BijectorAutoRegSpline(*self.args)
        new_bij.load_state_dict(torch.load(temp_file))
        new_bij.eval()
        # Cleanup
        os.unlink(temp_file)
        return new_bij

    def turn_off_grad(self):
        for param in self.parameters():
            param.requires_grad = False

# Benchmarking
if __name__ == "__main__":
    import time
    import torch
    from temperflow.bijector import BijectorAutoRegSpline

    # Distribution
    torch.manual_seed(123)
    dim = 32
    bs = 2000
    nrep = 1000
    device = torch.device("cuda")
    bij = BijectorAutoRegSpline(dim=dim, perm_dim=2, count_bins=64,
                                bound=8.0, hlayers=[64, 64], device=device)

    def test_fun(bij, bs):
        x = bij.distr.sample((bs,))
        logp = bij.distr.log_prob(x)
        x2 = x.clone()
        logp2 = bij.distr.log_prob(x2)
        return x, logp, logp2

    t1 = time.time()
    _ = test_fun(bij, bs)
    t2 = time.time()
    print(f"warmup finished in {t2 - t1} seconds")
    t1 = time.time()
    for i in range(nrep):
        _ = test_fun(bij, bs)
    t2 = time.time()
    print(f"finished in {t2 - t1} seconds")

# Transformation mapping based on autoregressive splines
class BijectorBlockAutoRegSpline(nn.Module):
    def __init__(
        self, dim, block_size, count_bins=8, bound=None, hlayers=[32, 32],
        nonlinearity=nn.ReLU(inplace=True), device=torch.device("cpu")
    ):
        super(BijectorBlockAutoRegSpline, self).__init__()
        self.args = (dim, block_size, count_bins, bound, hlayers, nonlinearity, device)

        # First layer: block autoregressive
        self.spline1 = SplineBlockAutoReg(*self.args)
        # Second layer: autoregressive within block
        self.spline2 = SplineAutoRegWithinBlock(*self.args)
        # transforms = [self.spline1.inv, self.spline2.inv]

        partial_dim = 2
        perm = torch.arange(dim, device=device)
        perm[:partial_dim] = torch.arange(partial_dim, device=device).flip(0)
        self.perm = dist.transforms.Permute(perm)
        self.spline3 = SplineAutoReg(
            dim, partial_dim=partial_dim, count_bins=count_bins, bound=bound, hlayers=hlayers,
            nonlinearity=nonlinearity, first_identity=True, device=device)
        transforms = [self.spline1.inv, self.spline2.inv, self.perm, self.spline3.inv]

        # Base distribution
        base = D.Independent(
            D.Normal(loc=torch.zeros(dim, device=device), scale=torch.ones(dim, device=device)),
            reinterpreted_batch_ndims=1)
        # Transformed distribution
        self.distr = dist.TransformedDistribution(
            base_distribution=base,
            transforms=transforms)

    def clone(self):
        # Save current model parameters to a temporary file
        temp_file = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        torch.save(self.state_dict(), temp_file)
        # Create a new model object
        new_bij = BijectorBlockAutoRegSpline(*self.args)
        new_bij.load_state_dict(torch.load(temp_file))
        new_bij.eval()
        # Cleanup
        os.unlink(temp_file)
        return new_bij

    def turn_off_grad(self):
        for param in self.parameters():
            param.requires_grad = False
