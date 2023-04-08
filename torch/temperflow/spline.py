from typing import Tuple, List
from typing_extensions import Final
import torch
import torch.nn as nn
import torch.nn.functional as F

# Modified version of _searchsorted() in pyro.distributions.transforms.spline
# https://docs.pyro.ai/en/stable/_modules/pyro/distributions/transforms/spline.html
# Uses the PyTorch function torch.searchsorted()
# Note that this version adds a dimension at the end
# sorted_sequence: [batch_size, input_dim, param_dim] or [input_dim, param_dim]
# values: [batch_size, input_dim]
# Returns [batch_size, input_dim, 1]
@torch.jit.script
def _searchsorted(sorted_sequence, values):
    # type: (Tensor, Tensor) -> Tensor
    if len(sorted_sequence.shape) == 2:
        batch_size = values.shape[:-1]
        sorted_sequence = sorted_sequence.reshape((1,) * len(batch_size) + sorted_sequence.shape)
        sorted_sequence = sorted_sequence.expand(batch_size + (-1,) * 2)
    return torch.searchsorted(sorted_sequence, values[..., None], right=True) - 1

# Modified version of _select_bins() in pyro.distributions.transforms.spline
# https://docs.pyro.ai/en/stable/_modules/pyro/distributions/transforms/spline.html
# x:   [batch_size, input_dim, param_dim]
# idx: [batch_size, input_dim, 1]
@torch.jit.script
def _select_bins(x, idx):
    # type: (Tensor, Tensor) -> Tensor
    idx = idx.clamp(min=0, max=x.size(-1) - 1)
    # Pyro version: len(idx.shape) >= len(x.shape)
    # We can omit the "==" case
    if len(idx.shape) > len(x.shape):
        x = x.reshape((1,) * (len(idx.shape) - len(x.shape)) + x.shape)
        x = x.expand(idx.shape[:-2] + (-1,) * 2)
    return x.gather(-1, idx).squeeze(-1)

# Copy of _calculate_knots() in pyro.distributions.transforms.spline
# https://docs.pyro.ai/en/stable/_modules/pyro/distributions/transforms/spline.html
@torch.jit.script
def _calculate_knots(lengths, lower, upper):
    # type: (Tensor, float, float) -> Tuple[Tensor, Tensor]
    # Cumulative widths gives x (y for inverse) position of knots
    knots = torch.cumsum(lengths, dim=-1)
    # Pad left of last dimension with 1 zero to compensate for dim lost to cumsum
    knots = F.pad(knots, pad=(1, 0), mode="constant", value=0.0)
    # Translate [0,1] knot points to [-B, B]
    knots = (upper - lower) * knots + lower
    # Convert the knot points back to lengths
    # NOTE: Are following two lines a necessary fix for accumulation (round-off) error?
    knots[..., 0] = lower
    knots[..., -1] = upper
    lengths = knots[..., 1:] - knots[..., :-1]
    return lengths, knots

# Modified version of _monotonic_rational_spline() in pyro.distributions.transforms.spline
# https://docs.pyro.ai/en/stable/_modules/pyro/distributions/transforms/spline.html
# Only supports linear rational splines
@torch.jit.script
def _monotonic_rational_spline(
    inputs: torch.Tensor,       # [n x input_dim]
    widths: torch.Tensor,       # [n x input_dim x count_bins] or [1 x input_dim x count_bins]
    heights: torch.Tensor,      # [n x input_dim x count_bins] or [1 x input_dim x count_bins]
    derivatives: torch.Tensor,  # [n x input_dim x (count_bins-1)] or [1 x input_dim x (count_bins-1)]
    lambdas: torch.Tensor,      # [n x input_dim x count_bins] or [1 x input_dim x count_bins]
    inverse: bool=False,
    bound: float=3.0,
    min_bin_width: float=1e-3,
    min_bin_height: float=1e-3,
    min_derivative: float=1e-3,
    min_lambda: float=0.025,
    eps: float=1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculating a monotonic rational spline (linear or quadratic) or its inverse,
    plus the log(abs(detJ)) required for normalizing flows.
    NOTE: I omit the docstring with parameter descriptions for this method since it
    is not considered "public" yet!
    """

    # Ensure bound is positive
    # NOTE: For simplicity, we apply the identity function outside [-B, B] X [-B, B] rather than allowing arbitrary
    # corners to the bounding box. If you want a different bounding box you can apply an affine transform before and
    # after the input
    assert bound > 0.0

    num_bins = widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    # inputs, inside_interval_mask, outside_interval_mask ~ (batch_dim, input_dim)
    left, right = -bound, bound
    bottom, top = -bound, bound
    # inside_interval_mask = (inputs >= left) & (inputs <= right)
    # outside_interval_mask = ~inside_interval_mask
    outside_interval_mask = torch.bitwise_or(inputs < left, inputs > right)

    # outputs, logabsdet ~ (batch_dim, input_dim)
    # outputs = torch.zeros_like(inputs)
    # logabsdet = torch.zeros_like(inputs)

    # For numerical stability, put lower/upper limits on parameters. E.g .give every bin min_bin_width,
    # then add width fraction of remaining length
    # NOTE: Do this here rather than higher up because we want everything to ensure numerical
    # stability within this function
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    derivatives = min_derivative + derivatives

    # Cumulative widths are x (y for inverse) position of knots
    # Similarly, cumulative heights are y (x for inverse) position of knots
    widths, cumwidths = _calculate_knots(widths, left, right)
    heights, cumheights = _calculate_knots(heights, bottom, top)

    # Pad left and right derivatives with fixed values at first and last knots
    # These are 1 since the function is the identity outside the bounding box and the derivative is continuous
    # NOTE: Not sure why this is 1.0 - min_derivative rather than 1.0. I've copied this from original implementation
    derivatives = F.pad(
        derivatives, pad=(1, 1), mode="constant", value=1.0 - min_derivative
    )

    widths = widths.squeeze(dim=0)
    heights = heights.squeeze(dim=0)
    derivatives = derivatives.squeeze(dim=0)
    lambdas = lambdas.squeeze(dim=0)
    cumwidths = cumwidths.squeeze(dim=0)
    cumheights = cumheights.squeeze(dim=0)

    # Get the index of the bin that each input is in
    # bin_idx ~ (batch_dim, input_dim, 1)
    bin_idx = _searchsorted(cumheights + eps if inverse else cumwidths + eps, inputs)

    # Select the value for the relevant bin for the variables used in the main calculation
    input_widths = _select_bins(widths, bin_idx)
    input_cumwidths = _select_bins(cumwidths, bin_idx)
    input_cumheights = _select_bins(cumheights, bin_idx)
    input_derivatives = _select_bins(derivatives, bin_idx)
    input_derivatives_plus_one = _select_bins(derivatives[..., 1:], bin_idx)
    input_heights = _select_bins(heights, bin_idx)
    # input_delta = _select_bins(heights / widths, bin_idx)
    input_delta = input_heights / input_widths

    # Calculate monotonic *linear* rational spline
    lambdas = (1 - 2 * min_lambda) * lambdas + min_lambda
    input_lambdas = _select_bins(lambdas, bin_idx)

    # The weight, w_a, at the left-hand-side of each bin
    # We are free to choose w_a, so set it to 1
    wa = 1.0

    # The weight, w_b, at the right-hand-side of each bin
    # This turns out to be a multiple of the w_a
    # TODO: Should this be done in log space for numerical stability?
    wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa

    # The weight, w_c, at the division point of each bin
    # Recall that each bin is divided into two parts so we have enough d.o.f. to fit spline
    # wc = (
    #     input_lambdas * wa * input_derivatives
    #     + (1 - input_lambdas) * wb * input_derivatives_plus_one
    # ) / input_delta
    lwa = input_lambdas * wa
    lwb = input_lambdas * wb
    # wc = (lwa * input_derivatives + (wb - lwb) * input_derivatives_plus_one) / input_delta
    wc = torch.addcmul(lwa * input_derivatives, wb - lwb, input_derivatives_plus_one) / input_delta

    # Calculate y coords of bins
    ya = input_cumheights
    yb = input_heights + input_cumheights
    # yc = ((1.0 - input_lambdas) * wa * ya + input_lambdas * wb * yb) / (
    #     (1.0 - input_lambdas) * wa + input_lambdas * wb
    # )
    # yc = ((wa - lwa) * ya + lwb * yb) / (wa - lwa + lwb)
    yc = torch.addcmul(lwb * yb, wa - lwa, ya) / (wa - lwa + lwb)

    waya = wa * ya
    wbyb = wb * yb
    wcyc = wc * yc

    if inverse:
        # numerator = (input_lambdas * wa * (ya - inputs)) * (
        #     inputs <= yc
        # ).float() + (
        #     (wc - input_lambdas * wb) * inputs + input_lambdas * wb * yb - wc * yc
        # ) * (
        #     inputs > yc
        # ).float()
        indicator = (inputs <= yc)
        numerator = torch.where(
            indicator,
            lwa * (ya - inputs),
            (wc - lwb) * inputs + lwb * yb - wcyc
        )

        # denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (
        #     inputs <= yc
        # ).float() + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()
        denominator = torch.where(
            indicator,
            (wc - wa) * inputs + waya,
            (wc - wb) * inputs + wbyb
        ) - wcyc

        theta = numerator / denominator

        # outputs = theta * input_widths + input_cumwidths
        outputs = torch.addcmul(input_cumwidths, theta, input_widths)

        # derivative_numerator = (
        #     wa * wc * input_lambdas * (yc - ya) * (inputs <= yc).float()
        #     + wb * wc * (1 - input_lambdas) * (yb - yc) * (inputs > yc).float()
        # ) * input_widths
        derivative_numerator = wc * torch.where(
            indicator,
            lwa * (yc - ya),
            (wb - lwb) * (yb - yc)
        ) * input_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))

    else:
        theta = (inputs - input_cumwidths) / input_widths

        # numerator = (wa * ya * (input_lambdas - theta) + wc * yc * theta) * (
        #     theta <= input_lambdas
        # ).float() + (wc * yc * (1 - theta) + wb * yb * (theta - input_lambdas)) * (
        #     theta > input_lambdas
        # ).float()
        indicator = (theta <= input_lambdas)
        ltheta = input_lambdas - theta
        numerator = torch.where(
            indicator,
            wcyc * theta + waya * ltheta,
            wcyc * (1 - theta) - wbyb * ltheta
        )

        # denominator = (wa * (input_lambdas - theta) + wc * theta) * (
        #     theta <= input_lambdas
        # ).float() + (wc * (1 - theta) + wb * (theta - input_lambdas)) * (
        #     theta > input_lambdas
        # ).float()
        denominator = torch.where(
            indicator,
            wc * theta + wa * ltheta,
            wc * (1 - theta) - wb * ltheta
        )

        outputs = numerator / denominator

        # derivative_numerator = (
        #     wa * wc * input_lambdas * (yc - ya) * (theta <= input_lambdas).float()
        #     + wb
        #     * wc
        #     * (1 - input_lambdas)
        #     * (yb - yc)
        #     * (theta > input_lambdas).float()
        # ) / input_widths
        derivative_numerator = wc * torch.where(
            indicator,
            wa * input_lambdas * (yc - ya),
            wb * (1 - input_lambdas) * (yb - yc)
        ) / input_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))

    # Apply the identity function outside the bounding box
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0.0
    return outputs, logabsdet



# Split the output of autoregressive networks to get parameters
# params_unnorm: [n x input_dim x param_dim] or [input_dim x param_dim]
# w, h, d, l: [n x input_dim x splits[*]] or [input_dim x splits[*]]
@torch.jit.script
def split_params(params_unnorm, splits):
    # type: (Tensor, List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    w, h, d, l = torch.split(params_unnorm, splits, dim=-1)
    w = F.softmax(w, dim=-1)
    h = F.softmax(h, dim=-1)
    d = F.softplus(d)
    l = torch.sigmoid(l)
    return w, h, d, l

# Unit test
if __name__ == "__main__":
    print("Testing split_params()")
    n, input_dim, param_dim = 3, 2, 10
    params_unnorm = torch.sin(torch.arange(n * input_dim * param_dim)).reshape(n, input_dim, param_dim)
    splits = [2, 3, 1, 4]
    w, h, d, l = split_params(params_unnorm, splits)
    print("w =\n", w)
    print("h =\n", h)
    print("d =\n", d)
    print("l =\n", l)

# lengths: [n x input_dim x count_bins] or [input_dim x count_bins]
@torch.jit.script
def calculate_knots(lengths, lower: float, upper: float):
    # Cumulative widths gives x (y for inverse) position of knots
    knots = torch.cumsum(lengths, dim=-1)
    # Translate [0,1] knot points to [-B, B]
    knots.mul_(upper - lower).add_(lower)
    # Pad left of last dimension with lower bound to compensate for dim lost to cumsum
    knots = F.pad(knots, pad=(1, 0), mode="constant", value=lower)
    knots[..., -1] = upper
    # Convert the knot points back to lengths
    lengths = torch.diff(knots, dim=-1)
    return lengths, knots

# Unit test
if __name__ == "__main__":
    print("Testing calculate_knots()")
    n, input_dim, param_dim = 3, 2, 10
    params_unnorm = torch.sin(torch.arange(n * input_dim * param_dim)).reshape(n, input_dim, param_dim)
    splits = [2, 3, 1, 4]
    w, h, d, l = split_params(params_unnorm, splits)
    lengths, knots = calculate_knots(w, -6.0, 6.0)
    print("w lengths =\n", lengths)
    print("w knots =\n", knots)
    lengths, knots = calculate_knots(h, -6.0, 6.0)
    print("h lengths =\n", lengths)
    print("h knots =\n", knots)

# sorted_sequence: [n x input_dim x param_dim] or [input_dim x param_dim]
# values: [n x input_dim]
# Returns [n x input_dim x 1]
@torch.jit.script
def search_sorted(sorted_sequence, values):
    if len(sorted_sequence.shape) == 2:
        n = values.shape[0]
        sorted_sequence = sorted_sequence.unsqueeze(dim=0).expand(n, -1, -1)
    return torch.searchsorted(sorted_sequence, values[..., None], right=True) - 1

# Unit test
if __name__ == "__main__":
    print("Testing search_sorted()")
    n, input_dim, param_dim = 3, 2, 10
    params_unnorm = torch.sin(torch.arange(n * input_dim * param_dim)).reshape(n, input_dim, param_dim)
    splits = [2, 3, 1, 4]
    w, h, d, l = split_params(params_unnorm, splits)
    widths, cumwidths = calculate_knots(w, -6.0, 6.0)
    values = torch.linspace(-7.0, 7.0, steps=n * input_dim).reshape(n, input_dim)
    print("w_index =\n", search_sorted(cumwidths, values))
    heights, cumheights = calculate_knots(h, -6.0, 6.0)
    print("h_index =\n", search_sorted(cumheights, values))

# x:   [n x input_dim x param_dim] or [input_dim x param_dim]
# idx: [n x input_dim x 1]
# out: [n x input_dim]
@torch.jit.script
def select_bins(x, idx):
    idx = idx.clamp(min=0, max=x.size(-1) - 1)
    if len(x.shape) == 2:
        n = idx.shape[0]
        x = x.unsqueeze(dim=0).expand(n, -1, -1)
    return x.gather(-1, idx).squeeze(dim=-1)

# Unit test
if __name__ == "__main__":
    print("Testing select_bins()")
    n, input_dim, param_dim = 3, 2, 10
    params_unnorm = torch.sin(torch.arange(n * input_dim * param_dim)).reshape(n, input_dim, param_dim)
    splits = [2, 3, 1, 4]
    w, h, d, l = split_params(params_unnorm, splits)
    widths, cumwidths = calculate_knots(w, -6.0, 6.0)
    values = torch.linspace(-7.0, 7.0, steps=n * input_dim).reshape(n, input_dim)
    ind = search_sorted(cumwidths, values)
    print("input_widths =\n", select_bins(widths, ind))
    print("input_cumwidths =\n", select_bins(cumwidths, ind))

# Module version of the linear rational spline
class LinearRationalSpline(nn.Module):
    # Constants
    num_bins: Final[float]
    bound: Final[float]
    min_bin_width: Final[float]
    min_bin_height: Final[float]
    min_derivative: Final[float]
    min_lambda: Final[float]
    eps: Final[float]
    splits: Final[List[int]]

    def __init__(
        self, num_bins, bound=3.0,
        min_bin_width=1e-3, min_bin_height=1e-3,
        min_derivative=1e-3, min_lambda=0.025, eps=1e-6
    ):
        super(LinearRationalSpline, self).__init__()
        self.num_bins = num_bins
        self.bound = bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.min_lambda = min_lambda
        self.eps = eps
        self.splits = [num_bins, num_bins, num_bins - 1, num_bins]

        # Ensure bound is positive
        # NOTE: For simplicity, we apply the identity function outside [-B, B] X [-B, B] rather than allowing arbitrary
        # corners to the bounding box. If you want a different bounding box you can apply an affine transform before and
        # after the input
        assert bound > 0.0

        if min_bin_width * num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if min_bin_height * num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")

    # params_unnorm: [n x input_dim x param_dim] or [input_dim x param_dim]
    # widths:        [n x input_dim x count_bins] or [input_dim x count_bins]
    # heights:       [n x input_dim x count_bins] or [input_dim x count_bins]
    # derivatives:   [n x input_dim x (count_bins-1)] or [input_dim x (count_bins-1)]
    # lambdas:       [n x input_dim x count_bins] or [input_dim x count_bins]
    @torch.jit.export
    def normalize_params(self, params_unnorm):
        # Split and normalize parameters
        widths, heights, derivatives, lambdas = split_params(params_unnorm, self.splits)

        # Bounds
        left, right = -self.bound, self.bound
        bottom, top = -self.bound, self.bound

        # For numerical stability, put lower/upper limits on parameters. E.g .give every bin min_bin_width,
        # then add width fraction of remaining length
        # NOTE: Do this here rather than higher up because we want everything to ensure numerical
        # stability within this function
        widths = self.min_bin_width + (1.0 - self.min_bin_width * self.num_bins) * widths
        heights = self.min_bin_height + (1.0 - self.min_bin_height * self.num_bins) * heights
        derivatives = self.min_derivative + derivatives
        lambdas = (1 - 2 * self.min_lambda) * lambdas + self.min_lambda

        # Cumulative widths are x (y for inverse) position of knots
        # Similarly, cumulative heights are y (x for inverse) position of knots
        widths, cumwidths = calculate_knots(widths, left, right)
        heights, cumheights = calculate_knots(heights, bottom, top)

        # Pad left and right derivatives with fixed values at first and last knots
        # These are 1 since the function is the identity outside the bounding box and the derivative is continuous
        # NOTE: Not sure why this is 1.0 - min_derivative rather than 1.0. I've copied this from original implementation
        derivatives = F.pad(
            derivatives, pad=(1, 1), mode="constant", value=1.0 - self.min_derivative
        )
        return widths, cumwidths, heights, cumheights, derivatives, lambdas

    @torch.jit.export
    def input_params(self, inputs, params_norm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                      torch.Tensor, torch.Tensor, torch.Tensor],
                     inverse: bool):
        widths, cumwidths, heights, cumheights, derivatives, lambdas = params_norm

        # Get the index of the bin that each input is in
        # bin_idx ~ (batch_dim, input_dim, 1)
        bin_idx = search_sorted(cumheights + self.eps if inverse else cumwidths + self.eps, inputs)

        # Select the value for the relevant bin for the variables used in the main calculation
        input_widths = select_bins(widths, bin_idx)
        input_heights = select_bins(heights, bin_idx)
        input_delta = input_heights / input_widths
        input_cumwidths = select_bins(cumwidths, bin_idx)
        input_cumheights = select_bins(cumheights, bin_idx)
        input_derivatives = select_bins(derivatives, bin_idx)
        input_derivatives_plus_one = select_bins(derivatives[..., 1:], bin_idx)
        input_lambdas = select_bins(lambdas, bin_idx)

        # The weight, w_a, at the left-hand-side of each bin
        # We are free to choose w_a, so set it to 1
        wa = 1.0

        # The weight, w_b, at the right-hand-side of each bin
        # This turns out to be a multiple of the w_a
        # TODO: Should this be done in log space for numerical stability?
        # wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa
        wb = torch.sqrt(input_derivatives / input_derivatives_plus_one)

        # The weight, w_c, at the division point of each bin
        # Recall that each bin is divided into two parts so we have enough d.o.f. to fit spline
        # lwa = input_lambdas * wa
        lwa = input_lambdas
        lwb = input_lambdas * wb
        # wc = (lwa * input_derivatives + (wb - lwb) * input_derivatives_plus_one) / input_delta
        wc = torch.addcmul(lwa * input_derivatives, wb - lwb, input_derivatives_plus_one) / input_delta

        # Calculate y coords of bins
        ya = input_cumheights
        yb = input_heights + input_cumheights
        # yc = ((wa - lwa) * ya + lwb * yb) / (wa - lwa + lwb)
        # yc = torch.addcmul(lwb * yb, wa - lwa, ya) / (wa - lwa + lwb)
        l1 = 1.0 - input_lambdas  # wa - lwa
        yc = torch.addcmul(lwb * yb, l1, ya) / (l1 + lwb)
        return wb, wc, lwb, ya, yb, yc, input_widths, input_cumwidths, input_lambdas

    def forward(self, inputs, params_unnorm):
        params_norm = self.normalize_params(params_unnorm)
        in_params = self.input_params(inputs, params_norm, inverse=False)
        wb, wc, lwb, ya, yb, yc, input_widths, input_cumwidths, input_lambdas = in_params

        waya = ya  # waya = wa * ya
        wbyb = wb * yb
        wcyc = wc * yc

        theta = (inputs - input_cumwidths) / input_widths
        indicator = (theta <= input_lambdas)
        ltheta = input_lambdas - theta

        wcyctheta = wcyc * theta
        numerator = torch.where(
            indicator,
            torch.addcmul(wcyctheta, waya, ltheta),  # wcyc * theta + waya * ltheta
            torch.addcmul(wcyc - wcyctheta, wbyb, ltheta, value=-1)  # wcyc * (1 - theta) - wbyb * ltheta
        )

        wctheta = wc * theta
        denominator = torch.where(
            indicator,
            wctheta + ltheta,  # wc * theta + wa * ltheta,
            torch.addcmul(wc - wctheta, wb, ltheta, value=-1)  # wc * (1 - theta) - wb * ltheta
        )

        outputs = numerator / denominator

        derivative_numerator = wc * torch.where(
            indicator,
            input_lambdas * (yc - ya),  # wa * input_lambdas * (yc - ya),
            (wb - lwb) * (yb - yc),  #  wb * (1 - input_lambdas) * (yb - yc)
        ) / input_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))

        # Apply the identity function outside the bounding box
        left, right = -self.bound, self.bound
        outside_interval_mask = torch.bitwise_or(inputs < left, inputs > right)
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet.masked_fill_(outside_interval_mask, 0.0)
        return outputs, logabsdet

    @torch.jit.export
    def inverse(self, inputs, params_unnorm):
        params_norm = self.normalize_params(params_unnorm)
        in_params = self.input_params(inputs, params_norm, inverse=True)
        wb, wc, lwb, ya, yb, yc, input_widths, input_cumwidths, input_lambdas = in_params

        waya = ya  # waya = wa * ya
        wbyb = wb * yb
        wcyc = wc * yc

        indicator = (inputs <= yc)
        numerator = torch.where(
            indicator,
            input_lambdas * (ya - inputs),  # lwa * (ya - inputs),
            (wc - lwb) * inputs + lwb * yb - wcyc
        )

        denominator = torch.where(
            indicator,
            (wc - 1.0) * inputs + waya,  # (wc - wa) * inputs + waya,
            (wc - wb) * inputs + wbyb
        ) - wcyc

        theta = numerator / denominator

        # outputs = theta * input_widths + input_cumwidths
        outputs = torch.addcmul(input_cumwidths, theta, input_widths)

        derivative_numerator = wc * torch.where(
            indicator,
            input_lambdas * (yc - ya),  # lwa * (yc - ya),
            (wb - lwb) * (yb - yc)
        ) * input_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))

        # Apply the identity function outside the bounding box
        left, right = -self.bound, self.bound
        outside_interval_mask = torch.bitwise_or(inputs < left, inputs > right)
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet.masked_fill_(outside_interval_mask, 0.0)
        return outputs, logabsdet

# Unit test
if __name__ == "__main__":
    import torch
    from temperflow.spline import LinearRationalSpline
    print("Testing class LinearRationalSpline")
    num_bins = 4
    spline = LinearRationalSpline(num_bins=num_bins)

    print("|__ Testing normalize_params()")
    n, input_dim, param_dim = 5, 2, 4 * num_bins - 1
    params_unnorm = torch.sin(torch.arange(n * input_dim * param_dim)).reshape(n, input_dim, param_dim)
    splits = [num_bins, num_bins, num_bins - 1, num_bins]
    params_norm = spline.normalize_params(params_unnorm)
    [print(p, "\n") for p in params_norm]
    print()

    print("|__ Testing input_params()")
    inputs = torch.linspace(-7.0, 7.0, steps=n * input_dim).reshape(n, input_dim)
    in_params = spline.input_params(inputs, params_norm, inverse=False)
    [print(p, "\n") for p in in_params]
    print()

    print("|__ Testing forward()")
    outputs, logabsdet = spline(inputs, params_unnorm)
    print("outputs =\n", outputs)
    print()
    print("logabsdet =\n", logabsdet)
    print()

    print("|__ Testing inverse()")
    inputs_recover, logabsdet = spline.inverse(outputs, params_unnorm)
    print("inputs_recover =\n", inputs_recover)
    print()
    print("inputs =\n", inputs)
    print()
    print("logabsdet =\n", logabsdet)

# Benchmarking
if __name__ == "__main__":
    import time
    import torch
    from temperflow.spline import LinearRationalSpline

    torch.manual_seed(123)
    device = torch.device("cuda")
    num_bins = 64
    n, input_dim, param_dim = 2000, 16, 4 * num_bins - 1
    nrep = 1000
    t_prepare, t_forward, t_inverse = 0.0, 0.0, 0.0

    spline = LinearRationalSpline(num_bins=num_bins)
    spline = torch.jit.script(spline)

    for _ in range(nrep):
        t1 = time.time()
        params_unnorm = torch.randn(n, input_dim, param_dim, device=device)
        splits = [num_bins, num_bins, num_bins - 1, num_bins]
        inputs = torch.randn(n, input_dim, device=device)
        t2 = time.time()

        outputs, logabsdet = spline(inputs, params_unnorm)
        # print(outputs.shape)
        t3 = time.time()

        inputs_recover, logabsdet = spline.inverse(outputs, params_unnorm)
        # print(inputs_recover.shape)
        t4 = time.time()

        t_prepare += (t2 - t1)
        t_forward += (t3 - t2)
        t_inverse += (t4 - t3)

    print(f"prepare time = {t_prepare} seconds with {nrep} runs")
    print(f"forward time = {t_forward} seconds with {nrep} runs")
    print(f"inverse time = {t_inverse} seconds with {nrep} runs")
