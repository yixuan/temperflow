from typing import List
import tensorflow as tf

import temperflow.option as opts

# Whether to JIT compile functions
JIT = opts.opts["jit"]
JIT_DEBUG = opts.opts["debug"]

# Split the output of autoregressive networks to get parameters
# params_unnorm: [n x input_dim x param_dim] or [input_dim x param_dim]
# w, h, d, l: [n x input_dim x splits[*]] or [input_dim x splits[*]]
@tf.function(jit_compile=JIT)
def split_params(params_unnorm: tf.Tensor, splits: List[int]):
    print("tracing split_params()") if JIT_DEBUG else None
    w, h, d, l = tf.split(params_unnorm, splits, axis=-1)
    w = tf.nn.softmax(w, axis=-1)
    h = tf.nn.softmax(h, axis=-1)
    d = tf.math.softplus(d)
    l = tf.math.sigmoid(l)
    return w, h, d, l

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.spline import split_params

    print("Testing split_params()")
    n, input_dim, param_dim = 3, 2, 10
    params_unnorm = tf.math.sin(
        tf.range(n * input_dim * param_dim, dtype=float))
    params_unnorm = tf.reshape(params_unnorm, (n, input_dim, param_dim))
    splits = [2, 3, 1, 4]
    w, h, d, l = split_params(params_unnorm, splits)
    print("w =\n", w)
    print("h =\n", h)
    print("d =\n", d)
    print("l =\n", l)

# lengths: [n x input_dim x count_bins] or [input_dim x count_bins]
@tf.function(jit_compile=JIT)
def calculate_knots(lengths: tf.Tensor, lower: float, upper: float):
    print("tracing calculate_knots()") if JIT_DEBUG else None
    # Cumulative widths gives x (y for inverse) position of knots
    knots = tf.cumsum(lengths, axis=-1)
    # Translate [0,1] knot points to [-B, B]
    knots = tf.math.scalar_mul(upper - lower, knots) + lower
    # Pad left of last dimension with lower bound to compensate for dim lost to cumsum
    # Also manually set the rightmost value to upper
    pad_shape = knots.shape[:-1] + [1]
    l = tf.fill(pad_shape, lower)
    l = tf.cast(l, knots.dtype)
    u = tf.fill(pad_shape, upper)
    u = tf.cast(u, knots.dtype)
    knots = tf.concat([l, knots[..., :-1], u], axis=-1)
    # Convert the knot points back to lengths
    # lengths = tf.experimental.numpy.diff(knots, axis=-1)
    lengths = knots[..., 1:] - knots[..., :-1]
    return lengths, knots

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.spline import split_params, calculate_knots

    print("Testing calculate_knots()")
    n, input_dim, param_dim = 3, 2, 10
    params_unnorm = tf.math.sin(
        tf.range(n * input_dim * param_dim, dtype=float))
    params_unnorm = tf.reshape(params_unnorm, (n, input_dim, param_dim))
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
@tf.function(jit_compile=JIT)
def search_sorted(sorted_sequence: tf.Tensor, values: tf.Tensor):
    print("tracing search_sorted()") if JIT_DEBUG else None
    if len(sorted_sequence.shape) == 2:
        n = values.shape[0]
        sorted_sequence = tf.expand_dims(sorted_sequence, axis=0)
        sorted_sequence = tf.tile(sorted_sequence, (n, 1, 1))
    return tf.searchsorted(sorted_sequence, values[..., None], side="right") - 1

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.spline import split_params, calculate_knots, search_sorted

    print("Testing search_sorted()")
    n, input_dim, param_dim = 3, 2, 10
    params_unnorm = tf.math.sin(
        tf.range(n * input_dim * param_dim, dtype=float))
    params_unnorm = tf.reshape(params_unnorm, (n, input_dim, param_dim))
    splits = [2, 3, 1, 4]
    w, h, d, l = split_params(params_unnorm, splits)
    widths, cumwidths = calculate_knots(w, -6.0, 6.0)
    values = tf.linspace(-7.0, 7.0, num=n * input_dim)
    values = tf.reshape(values, (n, input_dim))
    print("w_index =\n", search_sorted(cumwidths, values))
    heights, cumheights = calculate_knots(h, -6.0, 6.0)
    print("h_index =\n", search_sorted(cumheights, values))

# x:   [n x input_dim x param_dim] or [input_dim x param_dim]
# idx: [n x input_dim x 1]
# out: [n x input_dim]
@tf.function(jit_compile=JIT)
def select_bins(x: tf.Tensor, idx: tf.Tensor):
    print("tracing select_bins()") if JIT_DEBUG else None
    idx = tf.clip_by_value(idx, 0, x.shape[-1] - 1)
    idx = tf.stop_gradient(idx)
    if len(x.shape) == 2:
        n = idx.shape[0]
        x = tf.expand_dims(x, axis=0)
        x = tf.tile(x, (n, 1, 1))
    return tf.squeeze(tf.gather(x, idx, batch_dims=2), axis=-1)

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.spline import split_params, calculate_knots,\
        search_sorted, select_bins

    print("Testing select_bins()")
    n, input_dim, param_dim = 3, 2, 10
    params_unnorm = tf.math.sin(
        tf.range(n * input_dim * param_dim, dtype=float))
    params_unnorm = tf.reshape(params_unnorm, (n, input_dim, param_dim))
    splits = [2, 3, 1, 4]
    w, h, d, l = split_params(params_unnorm, splits)
    widths, cumwidths = calculate_knots(w, -6.0, 6.0)
    values = tf.linspace(-7.0, 7.0, num=n * input_dim)
    values = tf.reshape(values, (n, input_dim))
    ind = search_sorted(cumwidths, values)
    print("input_widths =\n", select_bins(widths, ind))
    print("input_cumwidths =\n", select_bins(cumwidths, ind))

# Module version of the linear rational spline
class LinearRationalSpline(tf.Module):
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
    @tf.function(jit_compile=JIT)
    def normalize_params(self, params_unnorm: tf.Tensor):
        print("tracing normalize_params()") if JIT_DEBUG else None
        # Split and normalize parameters
        widths, heights, derivatives, lambdas = split_params(params_unnorm, self.splits)

        # Bounds
        left, right = -self.bound, self.bound
        bottom, top = -self.bound, self.bound

        # For numerical stability, put lower/upper limits on parameters. E.g .give every bin min_bin_width,
        # then add width fraction of remaining length
        # NOTE: Do this here rather than higher up because we want everything to ensure numerical
        # stability within this function
        widths = self.min_bin_width + tf.scalar_mul(1.0 - self.min_bin_width * self.num_bins, widths)
        heights = self.min_bin_height + tf.scalar_mul(1.0 - self.min_bin_height * self.num_bins, heights)
        derivatives = self.min_derivative + derivatives
        lambdas = self.min_lambda + tf.scalar_mul(1 - 2 * self.min_lambda, lambdas)

        # Cumulative widths are x (y for inverse) position of knots
        # Similarly, cumulative heights are y (x for inverse) position of knots
        widths, cumwidths = calculate_knots(widths, left, right)
        heights, cumheights = calculate_knots(heights, bottom, top)

        # Pad left and right derivatives with fixed values at first and last knots
        # These are 1 since the function is the identity outside the bounding box and the derivative is continuous
        # NOTE: Not sure why this is 1.0 - min_derivative rather than 1.0. I've copied this from original implementation
        pad_shape = derivatives.shape[:-1] + [1]
        d = tf.fill(pad_shape, 1.0 - self.min_derivative)
        d = tf.cast(d, derivatives.dtype)
        derivatives = tf.concat([d, derivatives, d], axis=-1)
        return widths, cumwidths, heights, cumheights, derivatives, lambdas

    @tf.function(jit_compile=JIT)
    def input_params(self, inputs: tf.Tensor, params_norm, inverse: bool):
        print("tracing input_params()") if JIT_DEBUG else None
        widths, cumwidths, heights, cumheights, derivatives, lambdas = params_norm

        # Get the index of the bin that each input is in
        # bin_idx ~ (batch_dim, input_dim, 1)
        bin_idx = search_sorted(cumheights + self.eps if inverse else cumwidths + self.eps, inputs)
        bin_idx = tf.stop_gradient(bin_idx)

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
        # wa = 1.0

        # The weight, w_b, at the right-hand-side of each bin
        # This turns out to be a multiple of the w_a
        # TODO: Should this be done in log space for numerical stability?
        # wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa
        wb = tf.math.sqrt(input_derivatives / input_derivatives_plus_one)

        # The weight, w_c, at the division point of each bin
        # Recall that each bin is divided into two parts so we have enough d.o.f. to fit spline
        # lwa = input_lambdas * wa
        lwa = input_lambdas
        lwb = input_lambdas * wb
        wc = (lwa * input_derivatives + (wb - lwb) * input_derivatives_plus_one) / input_delta

        # Calculate y coords of bins
        ya = input_cumheights
        yb = input_heights + input_cumheights
        # yc = ((wa - lwa) * ya + lwb * yb) / (wa - lwa + lwb)
        # yc = torch.addcmul(lwb * yb, wa - lwa, ya) / (wa - lwa + lwb)
        l1 = 1.0 - input_lambdas  # wa - lwa
        yc = (l1 * ya + lwb * yb) / (l1 + lwb)
        return wb, wc, lwb, ya, yb, yc, input_widths, input_cumwidths, input_lambdas

    @tf.function(jit_compile=JIT)
    def __call__(self, inputs: tf.Tensor, params_unnorm: tf.Tensor):
        print("tracing __call__()") if JIT_DEBUG else None
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
        numerator = tf.where(
            indicator,
            wcyctheta + waya * ltheta,  # wcyc * theta + waya * ltheta
            wcyc - wcyctheta - wbyb * ltheta  # wcyc * (1 - theta) - wbyb * ltheta
        )

        wctheta = wc * theta
        denominator = tf.where(
            indicator,
            wctheta + ltheta,  # wc * theta + wa * ltheta,
            wc - wctheta - wb * ltheta  # wc * (1 - theta) - wb * ltheta
        )

        outputs = numerator / denominator

        derivative_numerator = wc * tf.where(
            indicator,
            input_lambdas * (yc - ya),  # wa * input_lambdas * (yc - ya),
            (wb - lwb) * (yb - yc),  #  wb * (1 - input_lambdas) * (yb - yc)
        ) / input_widths

        logabsdet = tf.math.log(derivative_numerator) - 2.0 * tf.math.log(tf.math.abs(denominator))

        # Apply the identity function outside the bounding box
        left, right = -self.bound, self.bound
        outside_interval_mask = tf.math.logical_or(inputs < left, inputs > right)
        # outputs[outside_interval_mask] = inputs[outside_interval_mask]
        outputs = tf.where(outside_interval_mask, inputs, outputs)
        # logabsdet[outside_interval_mask] = 0.0
        logabsdet = tf.where(outside_interval_mask, 0.0, logabsdet)
        return outputs, logabsdet

    @tf.function(jit_compile=JIT)
    def inverse(self, inputs: tf.Tensor, params_unnorm: tf.Tensor):
        print("tracing inverse()") if JIT_DEBUG else None
        params_norm = self.normalize_params(params_unnorm)
        in_params = self.input_params(inputs, params_norm, inverse=True)
        wb, wc, lwb, ya, yb, yc, input_widths, input_cumwidths, input_lambdas = in_params

        waya = ya  # waya = wa * ya
        wbyb = wb * yb
        wcyc = wc * yc

        indicator = (inputs <= yc)
        numerator = tf.where(
            indicator,
            input_lambdas * (ya - inputs),  # lwa * (ya - inputs),
            (wc - lwb) * inputs + lwb * yb - wcyc
        )

        denominator = tf.where(
            indicator,
            (wc - 1.0) * inputs + waya,  # (wc - wa) * inputs + waya,
            (wc - wb) * inputs + wbyb
        ) - wcyc

        theta = numerator / denominator

        outputs = theta * input_widths + input_cumwidths

        derivative_numerator = wc * tf.where(
            indicator,
            input_lambdas * (yc - ya),  # lwa * (yc - ya),
            (wb - lwb) * (yb - yc)
        ) * input_widths

        logabsdet = tf.math.log(derivative_numerator) - 2.0 * tf.math.log(tf.math.abs(denominator))

        # Apply the identity function outside the bounding box
        left, right = -self.bound, self.bound
        outside_interval_mask = tf.math.logical_or(inputs < left, inputs > right)
        # outputs[outside_interval_mask] = inputs[outside_interval_mask]
        outputs = tf.where(outside_interval_mask, inputs, outputs)
        # logabsdet[outside_interval_mask] = 0.0
        logabsdet = tf.where(outside_interval_mask, 0.0, logabsdet)
        return outputs, logabsdet

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.spline import LinearRationalSpline
    print("Testing class LinearRationalSpline")
    num_bins = 4
    spline = LinearRationalSpline(num_bins=num_bins)

    print("|__ Testing normalize_params()")
    n, input_dim, param_dim = 5, 2, 4 * num_bins - 1
    params_unnorm = tf.math.sin(
        tf.range(n * input_dim * param_dim, dtype=float))
    params_unnorm = tf.reshape(params_unnorm, (n, input_dim, param_dim))
    splits = [num_bins, num_bins, num_bins - 1, num_bins]
    params_norm = spline.normalize_params(params_unnorm)
    [print(p, "\n") for p in params_norm]
    print()

    print("|__ Testing input_params()")
    inputs = tf.linspace(-7.0, 7.0, num=n * input_dim)
    inputs = tf.reshape(inputs, (n, input_dim))
    in_params = spline.input_params(inputs, params_norm, inverse=False)
    [print(p, "\n") for p in in_params]
    print()

    print("|__ Testing __call__()")
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
    import tensorflow as tf
    from temperflow.spline import LinearRationalSpline

    tf.random.set_seed(123)
    num_bins = 64
    n, input_dim, param_dim = 2000, 16, 4 * num_bins - 1
    nrep = 1001
    spline = LinearRationalSpline(num_bins=num_bins)

    t_prepare, t_forward, t_inverse = 0.0, 0.0, 0.0
    for _ in range(nrep):
        t1 = time.time()
        params_unnorm = tf.random.normal((n, input_dim, param_dim))
        splits = [num_bins, num_bins, num_bins - 1, num_bins]
        inputs = tf.random.normal((n, input_dim))
        t2 = time.time()

        outputs, logabsdet = spline(inputs, params_unnorm)
        # print(spline.__call__.experimental_get_compiler_ir(inputs, params_unnorm)(stage="optimized_hlo"))
        # print(outputs.shape)
        t3 = time.time()

        inputs_recover, logabsdet = spline.inverse(outputs, params_unnorm)
        # print(inputs_recover.shape)
        t4 = time.time()

        # First iteration for warm-up
        if _ > 0:
            t_prepare += (t2 - t1)
            t_forward += (t3 - t2)
            t_inverse += (t4 - t3)

    print(f"prepare time = {t_prepare} seconds with {nrep} runs")
    print(f"forward time = {t_forward} seconds with {nrep} runs")
    print(f"inverse time = {t_inverse} seconds with {nrep} runs")

# Profiler
if __name__ == "__main__":
    tf.random.set_seed(123)
    num_bins = 64
    n, input_dim, param_dim = 2000, 16, 4 * num_bins - 1
    nrep = 100
    spline = LinearRationalSpline(num_bins=num_bins)

    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)
    tf.profiler.experimental.start("logdir", options=options)
    #========================================================
    for _ in range(nrep):
        params_unnorm = tf.random.normal((n, input_dim, param_dim))
        splits = [num_bins, num_bins, num_bins - 1, num_bins]
        inputs = tf.random.normal((n, input_dim))
        outputs, logabsdet = spline(inputs, params_unnorm)
    #========================================================
    tf.profiler.experimental.stop()

# Gradient
if __name__ == "__main__":
    import time
    import tensorflow as tf
    from temperflow.spline import LinearRationalSpline

    num_bins = 8
    n, input_dim, param_dim = 5, 2, 4 * num_bins - 1

    @tf.function(jit_compile=True)
    def get_grad(z, params_unnorm):
        spline = LinearRationalSpline(num_bins=num_bins)
        x, logdet = spline(z, params_unnorm)
        # Just a demonstration, not a meaningful loss
        loss = tf.math.reduce_mean(x) + tf.math.reduce_mean(logdet)
        grad = tf.gradients(loss, params_unnorm)
        return grad[0]

    tf.random.set_seed(123)
    params_unnorm = tf.random.normal(shape=(n, input_dim, param_dim))
    inputs = tf.random.normal(shape=(n, input_dim))
    # Run once to compile
    t1 = time.time()
    grad = get_grad(inputs, params_unnorm)
    print(tf.linalg.norm(grad))
    t2 = time.time()
    print(f"Compilation time ~ {t2 - t1} seconds")
    # Second run
    t1 = time.time()
    grad = get_grad(inputs, params_unnorm)
    print(tf.linalg.norm(grad))
    t2 = time.time()
    print(f"Running time ~ {t2 - t1} seconds")
    # Now we modify the value
    params_unnorm = tf.random.normal(shape=(n, input_dim, param_dim))
    inputs = tf.random.normal(shape=(n, input_dim))
    # The running time approximately equals the compilation time
    t1 = time.time()
    grad = get_grad(inputs, params_unnorm)
    print(tf.linalg.norm(grad))
    t2 = time.time()
    print(f"Running time ~ {t2 - t1} seconds")

    print(get_grad.experimental_get_compiler_ir(inputs, params_unnorm)(stage="optimized_hlo"))

@tf.function(jit_compile=JIT)
def split_params3(params_unnorm: tf.Tensor, splits: List[int]):
    print("tracing split_params3()") if JIT_DEBUG else None
    w, h, d = tf.split(params_unnorm, splits, axis=-1)
    w = tf.nn.softmax(w, axis=-1)
    h = tf.nn.softmax(h, axis=-1)
    d = tf.math.softplus(d)
    return w, h, d

# Module version of the quadratic rational spline
class QuadraticRationalSpline(tf.Module):
    def __init__(
        self, num_bins, bound=3.0,
        min_bin_width=1e-3, min_bin_height=1e-3,
        min_derivative=1e-3, eps=1e-6
    ):
        super(QuadraticRationalSpline, self).__init__()
        self.num_bins = num_bins
        self.bound = bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.eps = eps
        self.splits = [num_bins, num_bins, num_bins - 1]

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
    @tf.function(jit_compile=JIT)
    def normalize_params(self, params_unnorm: tf.Tensor):
        print("tracing normalize_params()") if JIT_DEBUG else None
        # Split and normalize parameters
        widths, heights, derivatives = split_params3(params_unnorm, self.splits)

        # Bounds
        left, right = -self.bound, self.bound
        bottom, top = -self.bound, self.bound

        # For numerical stability, put lower/upper limits on parameters. E.g .give every bin min_bin_width,
        # then add width fraction of remaining length
        # NOTE: Do this here rather than higher up because we want everything to ensure numerical
        # stability within this function
        widths = self.min_bin_width + tf.scalar_mul(1.0 - self.min_bin_width * self.num_bins, widths)
        heights = self.min_bin_height + tf.scalar_mul(1.0 - self.min_bin_height * self.num_bins, heights)
        derivatives = self.min_derivative + derivatives

        # Cumulative widths are x (y for inverse) position of knots
        # Similarly, cumulative heights are y (x for inverse) position of knots
        widths, cumwidths = calculate_knots(widths, left, right)
        heights, cumheights = calculate_knots(heights, bottom, top)

        # Pad left and right derivatives with fixed values at first and last knots
        # These are 1 since the function is the identity outside the bounding box and the derivative is continuous
        # NOTE: Not sure why this is 1.0 - min_derivative rather than 1.0. I've copied this from original implementation
        pad_shape = derivatives.shape[:-1] + [1]
        d = tf.fill(pad_shape, 1.0 - self.min_derivative)
        d = tf.cast(d, derivatives.dtype)
        derivatives = tf.concat([d, derivatives, d], axis=-1)
        return widths, cumwidths, heights, cumheights, derivatives

    @tf.function(jit_compile=JIT)
    def input_params(self, inputs: tf.Tensor, params_norm, inverse: bool):
        print("tracing input_params()") if JIT_DEBUG else None
        widths, cumwidths, heights, cumheights, derivatives = params_norm

        # Get the index of the bin that each input is in
        # bin_idx ~ (batch_dim, input_dim, 1)
        bin_idx = search_sorted(cumheights + self.eps if inverse else cumwidths + self.eps, inputs)
        bin_idx = tf.stop_gradient(bin_idx)

        # Select the value for the relevant bin for the variables used in the main calculation
        input_widths = select_bins(widths, bin_idx)
        input_heights = select_bins(heights, bin_idx)
        input_delta = input_heights / input_widths
        input_cumwidths = select_bins(cumwidths, bin_idx)
        input_cumheights = select_bins(cumheights, bin_idx)
        input_derivatives = select_bins(derivatives, bin_idx)
        input_derivatives_plus_one = select_bins(derivatives[..., 1:], bin_idx)
        return input_widths, input_heights, input_delta, \
            input_cumwidths, input_cumheights, \
            input_derivatives, input_derivatives_plus_one

    @tf.function(jit_compile=JIT)
    def __call__(self, inputs: tf.Tensor, params_unnorm: tf.Tensor):
        print("tracing __call__()") if JIT_DEBUG else None
        params_norm = self.normalize_params(params_unnorm)
        in_params = self.input_params(inputs, params_norm, inverse=False)
        input_widths, input_heights, input_delta, \
            input_cumwidths, input_cumheights, \
            input_derivatives, input_derivatives_plus_one = in_params

        theta = (inputs - input_cumwidths) / input_widths
        theta_one_minus_theta = theta * (1.0 - theta)
        theta_sq = tf.math.square(theta)
        one_minus_theta_sq = tf.math.square(1.0 - theta)

        numerator = input_heights * (
            input_delta * theta_sq + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + theta_one_minus_theta * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = tf.math.square(input_delta) * (
            input_derivatives_plus_one * theta_sq +
                2.0 * input_delta * theta_one_minus_theta +
                input_derivatives * one_minus_theta_sq
        )
        logabsdet = tf.math.log(derivative_numerator) - 2.0 * tf.math.log(denominator)

        # Apply the identity function outside the bounding box
        left, right = -self.bound, self.bound
        outside_interval_mask = tf.math.logical_or(inputs < left, inputs > right)
        # outputs[outside_interval_mask] = inputs[outside_interval_mask]
        outputs = tf.where(outside_interval_mask, inputs, outputs)
        # logabsdet[outside_interval_mask] = 0.0
        logabsdet = tf.where(outside_interval_mask, 0.0, logabsdet)
        return outputs, logabsdet

    @tf.function(jit_compile=JIT)
    def inverse(self, inputs: tf.Tensor, params_unnorm: tf.Tensor):
        print("tracing inverse()") if JIT_DEBUG else None
        params_norm = self.normalize_params(params_unnorm)
        in_params = self.input_params(inputs, params_norm, inverse=True)
        input_widths, input_heights, input_delta, \
            input_cumwidths, input_cumheights, \
            input_derivatives, input_derivatives_plus_one = in_params

        # Find inputs that are outside the bounding box
        left, right = -self.bound, self.bound
        outside_interval_mask = tf.math.logical_or(inputs < left, inputs > right)

        common1 = inputs - input_cumheights
        common2 = input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        common3 = common1 * common2
        a = input_heights * (input_delta - input_derivatives) + common3
        b = input_heights * input_derivatives - common3
        c = -input_delta * common1
        discriminant = tf.square(b) - 4.0 * a * c
        # Make sure outside_interval input can be reversed as identity.
        discriminant = tf.where(outside_interval_mask, 0.0, discriminant)

        root = (2.0 * c) / (-b - tf.math.sqrt(discriminant))
        outputs = root * input_widths + input_cumwidths

        theta_one_minus_theta = root * (1.0 - root)
        denominator = input_delta + theta_one_minus_theta * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        )
        derivative_numerator = tf.math.square(input_delta) * (
            input_derivatives_plus_one * tf.math.square(root) +
                2.0 * input_delta * theta_one_minus_theta +
                input_derivatives * tf.math.square(1.0 - root)
        )
        logabsdet = -(tf.math.log(derivative_numerator) - 2.0 * tf.math.log(denominator))

        # Apply the identity function outside the bounding box
        # outputs[outside_interval_mask] = inputs[outside_interval_mask]
        outputs = tf.where(outside_interval_mask, inputs, outputs)
        # logabsdet[outside_interval_mask] = 0.0
        logabsdet = tf.where(outside_interval_mask, 0.0, logabsdet)
        return outputs, logabsdet

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.spline import QuadraticRationalSpline
    print("Testing class QuadraticRationalSpline")
    num_bins = 4
    spline = QuadraticRationalSpline(num_bins=num_bins)

    print("|__ Testing normalize_params()")
    n, input_dim, param_dim = 5, 2, 3 * num_bins - 1
    params_unnorm = tf.math.sin(
        tf.range(n * input_dim * param_dim, dtype=float))
    params_unnorm = tf.reshape(params_unnorm, (n, input_dim, param_dim))
    splits = [num_bins, num_bins, num_bins - 1]
    params_norm = spline.normalize_params(params_unnorm)
    [print(p, "\n") for p in params_norm]
    print()

    print("|__ Testing input_params()")
    inputs = tf.linspace(-7.0, 7.0, num=n * input_dim)
    inputs = tf.reshape(inputs, (n, input_dim))
    in_params = spline.input_params(inputs, params_norm, inverse=False)
    [print(p, "\n") for p in in_params]
    print()

    print("|__ Testing __call__()")
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
