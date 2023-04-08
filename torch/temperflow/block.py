import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
import pyro.distributions as dist
from typing import List
from typing_extensions import Final

from temperflow.spline import LinearRationalSpline

# Fully-connected layers that can be compiled by TorchScript
class Dense(nn.Module):
    def __init__(
        self, in_dim, hlayers, out_dim,
        nonlinearity=nn.ReLU(inplace=True), device=torch.device("cpu")
    ):
        super(Dense, self).__init__()
        # Layers
        layers = []
        in_size = in_dim
        for out_size in hlayers:
            linear = nn.Linear(in_size, out_size, device=device)
            # nn.init.normal_(linear.bias, 0.0, 0.01)
            # nn.init.normal_(linear.weight, 0.0, 0.01)
            layers.append(linear)
            layers.append(nonlinearity)
            in_size = out_size
        # Output layer
        linear = nn.Linear(in_size, out_dim, device=device)
        # nn.init.normal_(linear.bias, 0.0, 0.01)
        # nn.init.normal_(linear.weight, 0.0, 0.01)
        layers.append(linear)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

# torch.manual_seed(123)
# dense = torch.jit.script(Dense(2, [3, 5], 3, device=torch.device("cuda")))
# print(dense.code)
# x = torch.randn(10, 2, device=torch.device("cuda"))
# print(dense(x))

# Autoregressive networks that can be compiled by TorchScript
class AutoRegDense(nn.Module):
    # Constants
    in_dim: Final[int]
    first_identity: Final[bool]

    def __init__(
        self, in_dim, hlayers, out_dim, nonlinearity=nn.ReLU(inplace=True),
        first_identity=False, device=torch.device("cpu")
    ):
        super(AutoRegDense, self).__init__()
        self.in_dim = in_dim
        self.first_identity = first_identity

        # Output for the first variable
        self.out0 = nn.Parameter(0.01 * torch.randn(1, out_dim, device=device))

        nets = []
        # The output of nets[i] is a function of x[0], ..., x[i-1]
        for i in range(1, in_dim):
            # Input: x[0], ..., x[i-1]
            net = Dense(in_dim=i, hlayers=hlayers, out_dim=out_dim,
                nonlinearity=nonlinearity, device=device)
            nets.append(net)
        self.nets = nn.ModuleList(nets)

    # x: [n x in_dim]
    # output: [n x (in_dim-1) x out_dim] if first_identity == True
    #         [n x in_dim x out_dim]     if first_identity == False
    def forward(self, x):
        n = x.shape[0]
        out = [] if self.first_identity else [torch.tile(self.out0, (n, 1))]
        for i, net in enumerate(self.nets):
            outi = net(x[:, :(i + 1)])
            out.append(outi)
        return torch.stack(out, dim=1)

# torch.manual_seed(123)
# ard = torch.jit.script(AutoRegDense(5, [3, 5], 3, device=torch.device("cuda")))
# print(ard.code)
# x = torch.randn(10, 5, device=torch.device("cuda"))
# print(ard(x))

# Neural networks used to construct masked affine coupling layers
# Can be compiled by TorchScript
class MACNet(nn.Module):
    def __init__(
        self, in_dim, hlayers, out_dim,
        nonlinearity=nn.ReLU6(inplace=True), device=torch.device("cpu")
    ):
        super(MACNet, self).__init__()
        self.shift = Dense(in_dim, hlayers, out_dim, nonlinearity, device)
        self.log_scale = Dense(in_dim, hlayers, out_dim, nonlinearity, device)

    def forward(self, x):
        shift = self.shift(x)
        log_scale = self.log_scale(x).clip_(-10.0, 10.0)
        scale = 2.0 * torch.sigmoid(log_scale) + 0.01
        log_scale = torch.log(scale)
        return shift, log_scale, scale

# Affine coupling layer, translated from https://github.com/xqding/RealNVP
# mask[i] == 1 means the i-th component is kept unchanged
class MaskedAffineCoupling(dist.TransformModule):
    bijective = True

    @dist.constraints.dependent_property(is_discrete=False)
    def domain(self):
        return dist.constraints.independent(dist.constraints.real, 1)

    @dist.constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return dist.constraints.independent(dist.constraints.real, 1)

    def __init__(self, mask, hypernet, device=torch.device("cpu")):
        super().__init__(cache_size=1)
        # self.mask = torch.tensor(mask, device=device)
        # self.maskc = 1.0 - self.mask
        self.mask = torch.tensor(mask, dtype=torch.bool, device=device)
        self.maskc = torch.logical_not(self.mask)
        self.hypernet = hypernet
        self.cached_logdet = None

    def _call(self, x):
        # masked_x = x * self.mask
        masked_x = x.masked_fill(self.maskc, 0.0)
        shift, log_scale, scale = self.hypernet(masked_x)
        # y = masked_x + self.maskc * (x * scale + shift)
        y = shift.addcmul_(x, scale).masked_fill_(self.mask, 0.0).add_(masked_x)
        # self.cached_logdet = torch.sum(self.maskc * log_scale, dim=1)
        self.cached_logdet = torch.sum(log_scale.masked_fill_(self.mask, 0.0), dim=1)
        return y

    def _inverse(self, y):
        # masked_y = y * self.mask
        masked_y = y.masked_fill(self.maskc, 0.0)
        shift, log_scale, scale = self.hypernet(masked_y)
        # x = masked_y + self.maskc * (y - shift) / scale
        x = masked_y.addcdiv((y - shift).masked_fill_(self.mask, 0.0), scale)
        # self.cached_logdet = torch.sum(self.maskc * log_scale, dim=1)
        self.cached_logdet = torch.sum(log_scale.masked_fill_(self.mask, 0.0), dim=1)
        return x

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if self.cached_logdet is not None and x is x_old and y is y_old:
            return self.cached_logdet
        else:
            # masked_x = x * self.mask
            masked_x = x.masked_fill(self.maskc, 0.0)
            _, log_scale, _ = self.hypernet(masked_x)
            # return torch.sum(self.maskc * log_scale, dim=1)
            return torch.sum(log_scale.masked_fill_(self.mask, 0.0), dim=1)

# 1D spline
class Spline1D(nn.Module):
    def __init__(
        self, spline, count_bins=8, device=torch.device("cpu")
    ):
        super(Spline1D, self).__init__()
        param_dim = 4 * count_bins - 1
        self.params_unnorm = nn.Parameter(0.01 * torch.randn(1, param_dim, device=device))
        self.spline = spline

    # x [n x 1]
    def forward(self, x):
        n = x.shape[0]
        x = x.view(n, 1)
        # Compute spline function values
        y, logdet = self.spline(x, self.params_unnorm)
        return y, logdet.squeeze()

    # y [n x 1]
    @torch.jit.export
    def inverse(self, y):
        n = y.shape[0]
        y = y.view(n, 1)
        # Compute inverse
        x, logdet = self.spline.inverse(y, self.params_unnorm)
        return x, -logdet.squeeze()

# 1D conditional spline
class CondSpline1D(nn.Module):
    def __init__(
        self, spline, count_bins=8, hlayers=[32, 32],
        nonlinearity=nn.ReLU(inplace=True), device=torch.device("cpu")
    ):
        super(CondSpline1D, self).__init__()
        param_dim = 4 * count_bins - 1
        self.net = Dense(in_dim=1, hlayers=hlayers, out_dim=param_dim, nonlinearity=nonlinearity, device=device)
        self.spline = spline

    # x [n x 1], condx [n x 1]
    def forward(self, x, condx):
        n = x.shape[0]
        x = x.view(n, 1)
        condx = condx.view(n, 1)
        # Unnormalized parameters
        params_unnorm = self.net(condx)
        # Compute spline function values
        y, logdet = self.spline(x, params_unnorm.unsqueeze(dim=-2))
        return y, logdet.squeeze()

    # y [n x 1], condx [n x 1]
    @torch.jit.export
    def inverse(self, y, condx):
        n = y.shape[0]
        y = y.view(n, 1)
        condx = condx.view(n, 1)
        # Unnormalized parameters
        params_unnorm = self.net(condx)
        # Compute spline function values
        x, logdet = self.spline.inverse(y, params_unnorm.unsqueeze(dim=-2))
        return x, -logdet.squeeze()

class Spline2DBase(nn.Module):
    # Constants
    unbounded: Final[bool]
    bound: Final[float]

    def __init__(
        self, count_bins=8, bound=None, hlayers=[32, 32],
        nonlinearity=nn.ReLU(inplace=True), device=torch.device("cpu")
    ):
        super().__init__()

        # If bound is not given, we set bound=5 and then add an affine transformation
        self.unbounded = False
        self.bound = bound
        if bound is None:
            self.unbounded = True
            self.bound = 5.0
            self.loc = nn.Parameter(torch.zeros(2, device=device))
            # scale=softplus(uscale), and softplus(0.54132)~=1.0
            self.uscale = nn.Parameter(torch.tensor([0.54132, 0.54132], device=device))

        # Linear rational spline
        self.spline = LinearRationalSpline(num_bins=count_bins, bound=self.bound)
        # Spline for the first variable, f1(x1)
        self.fx1 = Spline1D(self.spline, count_bins=count_bins, device=device)
        # (Conditional) spline for the second variable, f2(x2)
        self.fx2 = CondSpline1D(self.spline, count_bins=count_bins, hlayers=hlayers,
                                nonlinearity=nonlinearity, device=device)

    # x [n x 2]
    def forward(self, x):
        x1 = x[:, 0].view(-1, 1)
        x2 = x[:, 1].view(-1, 1)
        y1, logdet1 = self.fx1(x1)
        y2, logdet2 = self.fx2(x2, x1)
        y = torch.cat((y1, y2), dim=1)
        if self.unbounded:
            scale = F.softplus(self.uscale)
            # y = scale * y + self.loc
            y = torch.addcmul(self.loc, scale, y)
            logdet = logdet1 + logdet2 + torch.sum(torch.log(scale))
        else:
            logdet = logdet1 + logdet2
        return y, logdet

    # y [n x 2]
    @torch.jit.export
    def inverse(self, y):
        if self.unbounded:
            scale = F.softplus(self.uscale)
            y = (y - self.loc) / scale
        y1 = y[:, 0].view(-1, 1)
        y2 = y[:, 1].view(-1, 1)
        x1, logdet1 = self.fx1.inverse(y1)
        x2, logdet2 = self.fx2.inverse(y2, x1)
        x = torch.cat((x1, x2), dim=1)
        if self.unbounded:
            logdet = logdet1 + logdet2 + torch.sum(torch.log(scale))
        else:
            logdet = logdet1 + logdet2
        return x, logdet

# 2D spline with adaptive bounds
class Spline2D(dist.TransformModule):
    bijective = True

    @dist.constraints.dependent_property(is_discrete=False)
    def domain(self):
        return dist.constraints.independent(dist.constraints.real, 1)

    @dist.constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return dist.constraints.independent(dist.constraints.real, 1)

    def __init__(self, count_bins=8, bound=None, order="linear",
                 hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True), device=torch.device("cpu")):
        super().__init__(cache_size=1)

        # Spline module
        self.spline = torch.jit.script(
            Spline2DBase(count_bins=count_bins, bound=bound, hlayers=hlayers,
                         nonlinearity=nonlinearity, device=device)
        )
        # Cached log-determinant of Jacobian
        self.cached_logdet = None

    def _call(self, x):
        y, self.cached_logdet = self.spline(x)
        return y

    def _inverse(self, y):
        x, self.cached_logdet = self.spline.inverse(y)
        return x

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if self.cached_logdet is None or x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_det
            print("recomputing...")
            self(x)
        return self.cached_logdet

class SplineAutoRegBase(nn.Module):
    # Constants
    input_dim: Final[int]
    bound: Final[float]
    first_identity: Final[bool]

    def __init__(
        self, input_dim, count_bins=8, bound=None,
        hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True),
        first_identity=False, device=torch.device("cpu")
    ):
        super().__init__()

        self.input_dim = input_dim
        self.bound = bound
        self.first_identity = first_identity

        self.autoreg = AutoRegDense(
            input_dim, hlayers, 4 * count_bins - 1, nonlinearity,
            first_identity, device
        )
        self.spline = LinearRationalSpline(num_bins=count_bins, bound=bound)

    def forward(self, x):
        # If first_identity == True
        #     Remaining variables [n, input_dim-1, 4*count_bins-1]
        # else
        #     All variables [n, input_dim, 4*count_bins-1]
        params_unnorm = self.autoreg(x)
        # Call spline function
        if self.first_identity:
            yi, logdet = self.spline(x[:, 1:], params_unnorm)
            y = torch.cat((x[:, 0].view(-1, 1), yi), dim=-1)
        else:
            y, logdet = self.spline(x, params_unnorm)
        return y, torch.sum(logdet, dim=-1)

    @torch.jit.ignore
    def inverse(self, y):
        x = torch.empty_like(y)
        # The first variable
        if self.first_identity:
            y0 = y[:, 0]
            x[:, 0], logdet = y0, torch.zeros_like(y0)
        else:
            y0 = y[:, 0].view(-1, 1)
            x0, logdet = self.spline.inverse(y0, self.autoreg.out0)
            x[:, 0] = x0.squeeze()
            logdet = -logdet.squeeze()
        # Remaining variables
        for i, net in enumerate(self.autoreg.nets):
            params_unnorm = net(x[:, :(i + 1)].clone())
            yi = y[:, i + 1].view(-1, 1)
            xi, logdeti = self.spline.inverse(yi, params_unnorm.unsqueeze(dim=-2))
            x[:, i + 1] = xi.squeeze()
            logdeti = -logdeti.squeeze()
            logdet = logdet + logdeti
        return x, logdet

# Benchmarking
if __name__ == "__main__":
    import time
    import torch
    from temperflow.block import SplineAutoRegBase
    torch.manual_seed(123)
    n = 2000
    p = 64
    device = torch.device("cuda")
    x = torch.randn(n, p, device=device)
    net = SplineAutoRegBase(input_dim=p, count_bins=64, bound=6.0,
                            hlayers=[64, 64], first_identity=True,
                            device=device)
    net = torch.jit.script(net)
    out, logdet = net(x)
    print(out.shape)
    in_rec, logdet = net.inverse(out)
    print(in_rec.shape)

    torch.manual_seed(123)
    nrep = 100
    t_prepare, t_forward, t_inverse = 0.0, 0.0, 0.0
    for _ in range(nrep):
        t1 = time.time()
        x = torch.randn(n, p, device=device)
        t2 = time.time()

        out, logdet = net(x)
        t3 = time.time()

        in_rec, logdet = net.inverse(out)
        t4 = time.time()

        t_prepare += (t2 - t1)
        t_forward += (t3 - t2)
        t_inverse += (t4 - t3)

    print(f"prepare time = {t_prepare} seconds with {nrep} runs")
    print(f"forward time = {t_forward} seconds with {nrep} runs")
    print(f"inverse time = {t_inverse} seconds with {nrep} runs")

# Autoregressive spline
class SplineAutoReg(dist.TransformModule):
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector
    bijective = True
    autoregressive = True

    def __init__(
        self, input_dim, partial_dim=None, count_bins=8, bound=None,
        hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True),
        first_identity=False, device=torch.device("cpu")
    ):
        super().__init__(cache_size=1)
        self.partial = (partial_dim is not None)
        self.real_dim = partial_dim if self.partial else input_dim
        self.autoreg = torch.jit.script(
            SplineAutoRegBase(self.real_dim, count_bins, bound, hlayers, nonlinearity, first_identity, device)
        )

        # Cached log-determinant of Jacobian
        self.cached_logdet = None

    def _call(self, x):
        if self.partial:
            yp, logdet = self.autoreg(x[:, :self.real_dim])
            y = torch.cat((yp, x[:, self.real_dim:]), dim=-1)
        else:
            y, logdet = self.autoreg(x)
        self.cached_logdet = logdet
        return y

    def _inverse(self, y):
        if self.partial:
            xp, logdet = self.autoreg.inverse(y[:, :self.real_dim])
            x = torch.cat((xp, y[:, self.real_dim:]), dim=-1)
        else:
            x, logdet = self.autoreg.inverse(y)
        self.cached_logdet = logdet
        return x

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if self.cached_logdet is None or x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_det
            print("recomputing...")
            self(x)
        return self.cached_logdet

# Block autoregressive spline
class SplineBlockAutoRegBase(nn.Module):
    # Constants
    block_size: Final[int]
    blk_start: Final[List[int]]
    blk_end: Final[List[int]]
    blk_dim: Final[List[int]]
    cond_ind: Final[List[int]]
    param_dim: Final[int]

    def __init__(
        self, input_dim, block_size, count_bins=8, bound=None,
        hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True),
        device=torch.device("cpu")
    ):
        super().__init__()
        assert input_dim > block_size

        self.block_size = block_size
        # Variable indices for each block
        blk_ind = torch.split(torch.arange(input_dim, device=device), block_size)
        self.blk_start = [blk[0].item() for blk in blk_ind]
        self.blk_end = [blk[-1].item() + 1 for blk in blk_ind]
        # Number of variables for each block
        self.blk_dim = [blk.shape[0] for blk in blk_ind]
        # Total number of blocks
        blks = len(blk_ind)
        # Ending indices of the conditional variables for each block
        self.cond_ind = [i * block_size for i in range(blks)]
        # Neural networks
        nets = []
        self.param_dim = 4 * count_bins - 1
        for i in range(1, blks):
            # Current block size
            dimi = self.blk_dim[i]
            # Conditional variables: x[:, :cond_ind[i]]
            # Output: [n x dimi x param_dim]
            net = Dense(in_dim=self.cond_ind[i], hlayers=hlayers, out_dim=dimi * self.param_dim,
                        nonlinearity=nonlinearity, device=device)
            nets.append(net)
        self.nets = nn.ModuleList(nets)
        self.spline = LinearRationalSpline(num_bins=count_bins, bound=bound)

    def forward(self, x):
        # The first block remains unchanged
        x0 = x[:, :self.block_size]
        y0 = x0
        # Spline parameters for the remaining variables
        params_unnorm = []
        for i, net in enumerate(self.nets, 1):
            # Current block size
            dimi = self.blk_dim[i]
            # Conditional variables
            condx = x[:, :self.cond_ind[i]]
            # parami: [n x dimi x param_dim]
            parami = net(condx).view(-1, dimi, self.param_dim)
            params_unnorm.append(parami)
        # params_unnorm: [n x (input_dim - block_size) x param_dim]
        params_unnorm = torch.concat(params_unnorm, dim=1)
        # Transformation of the variables
        xrem = x[:, self.block_size:]
        yrem, logdet = self.spline(xrem, params_unnorm)
        y = torch.concat((y0, yrem), dim=1)
        return y, torch.sum(logdet, dim=-1)

    @torch.jit.ignore
    def inverse(self, y):
        # The first block
        y0 = y[:, :self.block_size]
        x0, logdet = y0, torch.zeros(y.shape[0], device=y.device)
        x = x0
        # Remaining blocks
        for i, net in enumerate(self.nets, 1):
            # Current block size
            dimi = self.blk_dim[i]
            # Spline parameters for the current block
            params_unnorm = net(x).view(-1, dimi, self.param_dim)
            # Inverse for the current block
            yi = y[:, self.blk_start[i]:self.blk_end[i]]
            xi, logdeti = self.spline.inverse(yi, params_unnorm)
            logdeti = -torch.sum(logdeti, dim=-1)
            logdet = logdet + logdeti
            x = torch.concat((x, xi), dim=-1)
        return x, logdet

class SplineBlockAutoReg(dist.TransformModule):
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector
    bijective = True
    autoregressive = True

    def __init__(
        self, input_dim, block_size, count_bins=8, bound=None,
        hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True),
        device=torch.device("cpu")
    ):
        super().__init__(cache_size=1)
        assert input_dim > block_size

        self.blockautoreg = torch.jit.script(
            SplineBlockAutoRegBase(
                input_dim, block_size, count_bins, bound,
                hlayers, nonlinearity, device)
        )
        # Cached log-determinant of Jacobian
        self.cached_logdet = None

    def _call(self, x):
        y, logdet = self.blockautoreg(x)
        self.cached_logdet = logdet
        return y

    def _inverse(self, y):
        x, logdet = self.blockautoreg.inverse(y)
        self.cached_logdet = logdet
        return x

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if self.cached_logdet is None or x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_det
            print("recomputing...")
            self(x)
        return self.cached_logdet

# Autoregressive spline within blocks
class SplineAutoRegWithinBlockBase(nn.Module):
    # Constants
    blk_start: Final[List[int]]
    blk_end: Final[List[int]]

    def __init__(
        self, input_dim, block_size, count_bins=8, bound=None,
        hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True),
        device=torch.device("cpu")
    ):
        super().__init__()
        assert input_dim > block_size

        # Variable indices for each block
        blk_ind = torch.split(torch.arange(input_dim, device=device), block_size)
        self.blk_start = [blk[0].item() for blk in blk_ind]
        self.blk_end = [blk[-1].item() + 1 for blk in blk_ind]
        # Number of variables for each block
        blk_dim = [blk.shape[0] for blk in blk_ind]
        # Total number of blocks
        blks = len(blk_ind)
        # Autoregressive bijectors
        autoregs = []
        for i in range(blks):
            # Current block size
            dimi = blk_dim[i]
            autoreg = SplineAutoRegBase(dimi, count_bins, bound, hlayers, nonlinearity,
                first_identity=(i != 0), device=device)
            autoregs.append(autoreg)
        self.autoregs = nn.ModuleList(autoregs)

    def forward(self, x):
        logdet = 0.0
        ys = []
        for i, autoreg in enumerate(self.autoregs):
            xi = x[:, self.blk_start[i]:self.blk_end[i]]
            yi, logdeti = autoreg(xi)
            logdet = logdet + logdeti
            ys.append(yi)
        y = torch.concat(ys, dim=-1)
        return y, logdet

    @torch.jit.ignore
    def inverse(self, y):
        logdet = 0.0
        xs = []
        for i, autoreg in enumerate(self.autoregs):
            yi = y[:, self.blk_start[i]:self.blk_end[i]]
            xi, logdeti = autoreg.inverse(yi)
            logdet = logdet + logdeti
            xs.append(xi)
        x = torch.concat(xs, dim=-1)
        return x, logdet

class SplineAutoRegWithinBlock(dist.TransformModule):
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector
    bijective = True
    autoregressive = True

    def __init__(
        self, input_dim, block_size, count_bins=8, bound=None,
        hlayers=[32, 32], nonlinearity=nn.ReLU(inplace=True),
        device=torch.device("cpu")
    ):
        super().__init__(cache_size=1)
        assert input_dim > block_size

        self.autoregwithinblock = torch.jit.script(
            SplineAutoRegWithinBlockBase(
                input_dim, block_size, count_bins, bound,
                hlayers, nonlinearity, device)
        )
        # Cached log-determinant of Jacobian
        self.cached_logdet = None

    def _call(self, x):
        y, logdet = self.autoregwithinblock(x)
        self.cached_logdet = logdet
        return y

    def _inverse(self, y):
        x, logdet = self.autoregwithinblock.inverse(y)
        self.cached_logdet = logdet
        return x

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if self.cached_logdet is None or x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_det
            print("recomputing...")
            self(x)
        return self.cached_logdet

# Spline coupling layer, modified from pyro.distributions.transforms.SplineCoupling
class SplineCoupling(dist.TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    # hypernet: split_dim => (input_dim - split_dim) * (4 * count_bins - 1)
    def __init__(self, input_dim, split_dim, hypernet, count_bins=8, bound=3.0):
        super(SplineCoupling, self).__init__(cache_size=1)

        self.hypernet = hypernet
        self.split_dim = split_dim
        self.remain_dim = input_dim - split_dim
        self.param_dim = 4 * count_bins - 1
        self.spline = LinearRationalSpline(num_bins=count_bins, bound=bound)

    def _call(self, x):
        x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]
        # Keep the first s variables identical
        y1 = x1
        # Compute spline parameters conditional on the first s variables
        params_unnorm = self.hypernet(x1).view(-1, self.remain_dim, self.param_dim)
        # Apply splines to the remaining d-s variables
        y2, logdet = self.spline(x2, params_unnorm)
        self._cache_log_detJ = torch.sum(logdet, dim=-1)
        return torch.cat([y1, y2], dim=-1)

    def _inverse(self, y):
        y1, y2 = y[..., :self.split_dim], y[..., self.split_dim:]
        # Keep the first s variables identical
        x1 = y1
        # Compute spline parameters conditional on the first s variables
        params_unnorm = self.hypernet(y1).view(-1, self.remain_dim, self.param_dim)
        # Apply inverse splines to the remaining d-s variables
        x2, logdet = self.spline.inverse(y2, params_unnorm)
        self._cache_log_detJ = -torch.sum(logdet, dim=-1)
        return torch.cat([x1, x2], dim=-1)

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)
        return self._cache_log_detJ
