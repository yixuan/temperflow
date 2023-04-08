import math
import tensorflow as tf
import tensorflow_probability as tfp
from keras import Sequential
import keras.layers as nn
from keras.engine.base_layer import Layer

import temperflow.option as opts
from temperflow.spline import LinearRationalSpline, QuadraticRationalSpline

# Whether to JIT compile functions
JIT = opts.opts["jit"]

# Smoothed ReLU activation
class SmoothedReLU(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return tfp.math.log_cosh(tf.nn.relu(inputs))

# log(a + b * sigmoid(x)), a>0, b>0
# = log((a + b) * exp(x) + a) - log(1 + exp(x))
# = log(a) + log(1 + exp(x + log(1 + b/a))) - softplus(x)
# = log(a) + softplus(x + log(1 + b/a)) - softplus(x)
@tf.function(jit_compile=JIT)
def log_sigmoid_affine(x, a, b):
    loga = math.log(a)
    shift = math.log(1.0 + b / a)
    return loga + tf.math.softplus(x + shift) - tf.math.softplus(x)

# Fully-connected layers
class Dense(tf.Module):
    def __init__(self, in_dim, hlayers, out_dim, nonlinearity=nn.ReLU):
        super(Dense, self).__init__()
        # Layers
        self.nets = Sequential()
        self.nets.add(nn.InputLayer(input_shape=(in_dim,)))
        # Without this explicit seed, the weights of linear layers are not reproducible
        # when JIT compiling is enabled
        seed = tf.random.uniform([1], 0, 2 ** 30, dtype=tf.int32)
        init = tf.keras.initializers.GlorotUniform(seed=seed[0])
        # init = tf.keras.initializers.RandomNormal(seed=seed[0])
        for out_size in hlayers:
            linear = nn.Dense(units=out_size, kernel_initializer=init)
            self.nets.add(linear)
            self.nets.add(nonlinearity())
        # Output layer
        linear = nn.Dense(units=out_dim, kernel_initializer=init)
        self.nets.add(linear)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        return self.nets(x)

# Test compilation
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.block import Dense
    tf.random.set_seed(123)
    x = tf.random.normal((10, 3))
    net = Dense(in_dim=3, hlayers=[8, 10, 6], out_dim=2)
    out = net(x)
    print(net.__call__.experimental_get_compiler_ir(x)(stage="optimized_hlo"))

# Affine coupling layer
class AffineCouplingBase(tf.Module):
    def __init__(
        self, input_dim, keep_dim, hlayers=[32, 32], nonlinearity=nn.ReLU
    ):
        super().__init__()
        self.args = (input_dim, keep_dim, hlayers, nonlinearity)

        rem_dim = input_dim - keep_dim
        self.keep_dim = keep_dim
        self.shift = Dense(keep_dim, hlayers, rem_dim, nonlinearity)
        self.log_scale = Dense(keep_dim, hlayers, rem_dim, nonlinearity)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        xkeep = x[:, :self.keep_dim]
        shift = self.shift(xkeep)
        log_scale = self.log_scale(xkeep)
        log_scale = log_sigmoid_affine(log_scale, 0.01, 2.0)
        scale = tf.math.exp(log_scale)
        yrem = shift + scale * x[:, self.keep_dim:]
        y = tf.concat((xkeep, yrem), axis=-1)
        logdet = tf.math.reduce_sum(log_scale, axis=-1)
        return y, logdet

    # In PyTorch, the inverse function always caches the forward log_det_jac
    # But here inverse() should return -log_det_jac
    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        xkeep = y[:, :self.keep_dim]
        shift = self.shift(xkeep)
        log_scale = self.log_scale(xkeep)
        log_scale = log_sigmoid_affine(log_scale, 0.01, 2.0)
        scale = tf.math.exp(log_scale)
        xrem = (y[:, self.keep_dim:] - shift) / scale
        x = tf.concat((xkeep, xrem), axis=-1)
        logdet = -tf.math.reduce_sum(log_scale, axis=-1)
        return x, logdet

    def copy(self):
        # Create the new object
        new_model = AffineCouplingBase(*self.args)
        # Copy the parameters
        for new_var, self_var in zip(new_model.variables, self.variables):
            new_var.assign(self_var.value())
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.block import AffineCouplingBase
    from temperflow.bijector import test_copy, test_simple
    tf.random.set_seed(123)
    bij = AffineCouplingBase(input_dim=9, keep_dim=5)
    x = tf.linspace(-5.0, 5.0, num=27)
    x = tf.reshape(x, (3, 9))
    test_copy(bij, x)
    test_simple(bij, x)

# Autoregressive networks
class AutoRegDense(tf.Module):
    def __init__(
        self, in_dim, hlayers, out_dim, nonlinearity=nn.ReLU,
        first_identity=False
    ):
        super(AutoRegDense, self).__init__()
        self.first_identity = first_identity

        # Output for the first variable
        if not first_identity:
            self.out0 = tf.Variable(tf.random.normal(shape=(1, out_dim), stddev=0.01))

        self.nets = []
        # The output of nets[i] is a function of x[0], ..., x[i-1]
        for i in range(1, in_dim):
            # Input: x[0], ..., x[i-1]
            net = Dense(in_dim=i, hlayers=hlayers, out_dim=out_dim, nonlinearity=nonlinearity)
            self.nets.append(net)

    # x: [n x in_dim]
    # output: [n x (in_dim-1) x out_dim] if first_identity == True
    #         [n x in_dim x out_dim]     if first_identity == False
    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        n = x.shape[0]
        out = [] if self.first_identity else [tf.tile(self.out0, (n, 1))]
        for i, net in enumerate(self.nets):
            outi = net(x[:, :(i + 1)])
            out.append(outi)
        return tf.stack(out, axis=1)

# Test compilation
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.block import AutoRegDense
    tf.random.set_seed(123)
    x = tf.random.normal((10, 3))
    net = AutoRegDense(in_dim=3, hlayers=[8, 8], out_dim=2, first_identity=True)
    out = net(x)
    print(out.shape)
    print(net.__call__.experimental_get_compiler_ir(x)(stage="optimized_hlo"))

    net = AutoRegDense(in_dim=3, hlayers=[8, 8], out_dim=2, first_identity=False)
    out = net(x)
    print(out.shape)
    print(net.__call__.experimental_get_compiler_ir(x)(stage="optimized_hlo"))

# Autoregressive flow
class AutoRegAffineBase(tf.Module):
    def __init__(
        self, input_dim, hlayers=[32, 32], nonlinearity=nn.ReLU
    ):
        super().__init__()
        self.args = (input_dim, hlayers, nonlinearity)

        self.input_dim = input_dim
        self.autoreg = AutoRegDense(
            input_dim, hlayers, 2, nonlinearity, first_identity=False)

    def stable_shift_scale(self, params):
        shift = params[..., 0]
        s = params[..., 1]
        # s = tf.clip_by_value(s, -10.0, 10.0)
        log_scale = log_sigmoid_affine(s, 0.01, 2.0)
        scale = tf.math.exp(log_scale)  # = 2.0 * tf.math.sigmoid(s) + 0.01
        return shift, scale, log_scale

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        # print("tracing call()")
        # [n x input_dim x 2]
        params = self.autoreg(x)
        shift, scale, log_scale = self.stable_shift_scale(params)
        y = shift + scale * x
        logdet = tf.math.reduce_sum(log_scale, axis=-1)
        return y, logdet

    # In PyTorch, the inverse function always caches the forward log_det_jac
    # But here inverse() should return -log_det_jac
    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        # print("tracing inverse()")
        y0 = tf.reshape(y[:, 0], (-1, 1))
        shift0, scale0, log_scale0 = self.stable_shift_scale(self.autoreg.out0)
        x = (y0 - shift0) / scale0
        logdet = tf.zeros_like(x) + log_scale0
        logdet = tf.squeeze(logdet)

        # Remaining variables
        for i, net in enumerate(self.autoreg.nets):
            # [n x 2]
            params = net(x)
            shifti, scalei, log_scalei = self.stable_shift_scale(params)
            shifti = tf.reshape(shifti, (-1, 1))
            scalei = tf.reshape(scalei, (-1, 1))
            log_scalei = tf.reshape(log_scalei, (-1, 1))
            yi = tf.reshape(y[:, i + 1], (-1, 1))
            xi = (yi - shifti) / scalei
            x = tf.concat((x, xi), axis=-1)
            logdeti = tf.squeeze(log_scalei)
            logdet = logdet + logdeti
        return x, -logdet

    def copy(self):
        # Create the new object
        new_model = AutoRegAffineBase(*self.args)
        # Copy the parameters
        for new_var, self_var in zip(new_model.variables, self.variables):
            new_var.assign(self_var.value())
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.block import AutoRegAffineBase
    from temperflow.bijector import test_copy, test_simple
    tf.random.set_seed(123)
    bij = AutoRegAffineBase(input_dim=9)
    x = tf.linspace(-5.0, 5.0, num=27)
    x = tf.reshape(x, (3, 9))
    test_copy(bij, x)
    test_simple(bij, x)

# Autoregressive spline
class SplineAutoRegBase(tf.Module):
    def __init__(
        self, input_dim, count_bins=8, bound=6.0, hlayers=[32, 32],
        nonlinearity=nn.ReLU, first_identity=False, spline="linear"
    ):
        super().__init__()
        self.args = (input_dim, count_bins, bound, hlayers,
                     nonlinearity, first_identity, spline)

        self.input_dim = input_dim
        self.bound = bound
        self.first_identity = first_identity

        if spline == "linear":
            params_dim = 4 * count_bins - 1
            SplineClass = LinearRationalSpline
        else:
            params_dim = 3 * count_bins - 1
            SplineClass = QuadraticRationalSpline
        self.autoreg = AutoRegDense(
            input_dim, hlayers, params_dim, nonlinearity, first_identity)
        self.spline = SplineClass(num_bins=count_bins, bound=bound)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        # print("tracing call()")
        # If first_identity == True
        #     Remaining variables [n, input_dim-1, 4*count_bins-1]
        # else
        #     All variables [n, input_dim, 4*count_bins-1]
        params_unnorm = self.autoreg(x)
        # Call spline function
        if self.first_identity:
            yi, logdet = self.spline(x[:, 1:], params_unnorm)
            y = tf.concat((x[:, :1], yi), axis=-1)
        else:
            y, logdet = self.spline(x, params_unnorm)
        return y, tf.math.reduce_sum(logdet, axis=-1)

    # In PyTorch, the inverse function always caches the forward log_det_jac
    # But here inverse() should return -log_det_jac
    # Spline has already added the negative sign
    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        # print("tracing inverse()")
        y0 = tf.reshape(y[:, 0], (-1, 1))
        # The first variable
        if self.first_identity:
            x, logdet = y0, tf.zeros_like(y0)
        else:
            x, logdet = self.spline.inverse(y0, self.autoreg.out0)
        logdet = tf.squeeze(logdet)

        # Remaining variables
        for i, net in enumerate(self.autoreg.nets):
            params_unnorm = tf.expand_dims(net(x), axis=-2)
            yi = tf.reshape(y[:, i + 1], (-1, 1))
            xi, logdeti = self.spline.inverse(yi, params_unnorm)
            x = tf.concat((x, xi), axis=-1)
            logdeti = tf.squeeze(logdeti)
            logdet = logdet + logdeti
        return x, logdet

    def copy(self):
        # Create the new object
        new_model = SplineAutoRegBase(*self.args)
        # Copy the parameters
        for new_var, self_var in zip(new_model.variables, self.variables):
            new_var.assign(self_var.value())
        return new_model

# Test compilation
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.block import SplineAutoRegBase
    tf.random.set_seed(123)
    x = tf.random.normal((10, 5))
    net = SplineAutoRegBase(input_dim=5, count_bins=16, bound=6.0,
                            hlayers=[8, 8], first_identity=True)
    out, logdet = net(x)
    print("out =\n", out)
    print()
    print("logdet =\n", logdet)
    print()
    in_rec, logdet = net.inverse(out)
    print("in_rec =\n", in_rec)
    print()
    print("in =\n", x)
    print()
    print("logdet =\n", logdet)
    print()
    print(net.__call__.experimental_get_compiler_ir(x)(stage="optimized_hlo"))

    net = SplineAutoRegBase(input_dim=5, count_bins=16, bound=6.0,
                            hlayers=[8, 8], first_identity=False)
    out, logdet = net(x)
    print("out =\n", out)
    print()
    print("logdet =\n", logdet)
    print()
    in_rec, logdet = net.inverse(out)
    print("in_rec =\n", in_rec)
    print()
    print("in =\n", x)
    print()
    print("logdet =\n", logdet)
    print()
    print(net.__call__.experimental_get_compiler_ir(x)(stage="optimized_hlo"))

# Benchmarking
if __name__ == "__main__":
    import time
    import tensorflow as tf

    # import temperflow.option as opts
    # opts.set_opts(jit=True)
    from temperflow.block import SplineAutoRegBase
    tf.random.set_seed(123)
    n = 2000
    p = 64
    x = tf.random.normal((n, p))
    net = SplineAutoRegBase(input_dim=p, count_bins=64, bound=6.0,
                            hlayers=[64, 64], first_identity=True)
    out, logdet = net(x)
    print(out.shape)
    in_rec, logdet = net.inverse(out)
    print(in_rec.shape)

    # tf.saved_model.save(net, "model")
    # net = tf.saved_model.load("model")

    tf.random.set_seed(123)
    nrep = 100
    t_prepare, t_forward, t_inverse = 0.0, 0.0, 0.0
    for _ in range(nrep):
        t1 = time.time()
        x = tf.random.normal((n, p))
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

# Block autoregressive spline
class SplineBlockAutoRegBase(tf.Module):
    def __init__(
        self, input_dim, block_size, count_bins=8, bound=6.0,
        hlayers=[32, 32], nonlinearity=nn.ReLU, spline="linear"
    ):
        super().__init__()
        assert input_dim > block_size
        self.args = (input_dim, block_size, count_bins, bound,
                     hlayers, nonlinearity, spline)

        self.input_dim = input_dim
        self.block_size = block_size
        # Total number of blocks
        blks = (input_dim - 1) // block_size + 1
        # Variable indices for each block
        blk_ind = list(range(input_dim))
        blk_ind = [blk_ind[(i * block_size):((i + 1) * block_size)] for i in range(blks)]
        self.blk_start = [blk[0] for blk in blk_ind]
        self.blk_end = [blk[-1] + 1 for blk in blk_ind]
        # Number of variables for each block
        self.blk_dim = [len(blk) for blk in blk_ind]
        # Ending indices of the conditional variables for each block
        self.cond_ind = [i * block_size for i in range(blks)]
        # Neural networks
        if spline == "linear":
            params_dim = 4 * count_bins - 1
            SplineClass = LinearRationalSpline
        else:
            params_dim = 3 * count_bins - 1
            SplineClass = QuadraticRationalSpline
        nets = []
        self.param_dim = params_dim
        for i in range(1, blks):
            # Current block size
            dimi = self.blk_dim[i]
            # Conditional variables: x[:, :cond_ind[i]]
            # Output: [n x dimi x param_dim]
            net = Dense(in_dim=self.cond_ind[i], hlayers=hlayers, out_dim=dimi * self.param_dim,
                        nonlinearity=nonlinearity)
            nets.append(net)
        self.nets = nets
        self.spline = SplineClass(num_bins=count_bins, bound=bound)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
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
            parami = tf.reshape(net(condx), (-1, dimi, self.param_dim))
            params_unnorm.append(parami)
        # params_unnorm: [n x (input_dim - block_size) x param_dim]
        params_unnorm = tf.concat(params_unnorm, axis=1)
        # Transformation of the variables
        xrem = x[:, self.block_size:]
        yrem, logdet = self.spline(xrem, params_unnorm)
        y = tf.concat((y0, yrem), axis=-1)
        return y, tf.math.reduce_sum(logdet, axis=-1)

    # In PyTorch, the inverse function always caches the forward log_det_jac
    # But here inverse() should return -log_det_jac
    # Spline has already added the negative sign
    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        # The first block
        y0 = y[:, :self.block_size]
        x0, logdet = y0, tf.zeros(y.shape[0])
        x = x0
        # Remaining blocks
        for i, net in enumerate(self.nets, 1):
            # Current block size
            dimi = self.blk_dim[i]
            # Spline parameters for the current block
            params_unnorm = tf.reshape(net(x), (-1, dimi, self.param_dim))
            # Inverse for the current block
            yi = y[:, self.blk_start[i]:self.blk_end[i]]
            xi, logdeti = self.spline.inverse(yi, params_unnorm)
            logdeti = tf.math.reduce_sum(logdeti, axis=-1)
            logdet = logdet + logdeti
            x = tf.concat((x, xi), axis=-1)
        return x, logdet

    def copy(self):
        # Create the new object
        new_model = SplineBlockAutoRegBase(*self.args)
        # Copy the parameters
        for new_var, self_var in zip(new_model.variables, self.variables):
            new_var.assign(self_var.value())
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.block import SplineBlockAutoRegBase
    from temperflow.bijector import test_copy, test_simple
    tf.random.set_seed(123)
    bij = SplineBlockAutoRegBase(input_dim=10, block_size=3)
    x = tf.linspace(-5.0, 5.0, num=30)
    x = tf.reshape(x, (3, 10))
    test_copy(bij, x)
    test_simple(bij, x)

# Block autoregressive spline
class SplineAutoRegWithinBlockBase(tf.Module):
    def __init__(
        self, input_dim, block_size, count_bins=8, bound=6.0,
        hlayers=[32, 32], nonlinearity=nn.ReLU, spline="linear"
    ):
        super().__init__()
        assert input_dim > block_size
        self.args = (input_dim, block_size, count_bins, bound,
                     hlayers, nonlinearity, spline)

        self.input_dim = input_dim
        # Total number of blocks
        blks = (input_dim - 1) // block_size + 1
        # Variable indices for each block
        blk_ind = list(range(input_dim))
        blk_ind = [blk_ind[(i * block_size):((i + 1) * block_size)] for i in range(blks)]
        self.blk_start = [blk[0] for blk in blk_ind]
        self.blk_end = [blk[-1] + 1 for blk in blk_ind]
        # Number of variables for each block
        blk_dim = [len(blk) for blk in blk_ind]
        # Spline object
        if spline == "linear":
            SplineClass = LinearRationalSpline
        else:
            SplineClass = QuadraticRationalSpline
        self.spline = SplineClass(num_bins=count_bins, bound=bound)
        # Autoregressive bijectors
        autoregs = []
        for i in range(blks):
            # Current block size
            dimi = blk_dim[i]
            autoreg = SplineAutoRegBase(dimi, count_bins, bound, hlayers, nonlinearity,
                                        first_identity=False, spline=spline)
            # Small hack to share the spline object, which saves some compilation time
            autoreg.spline = self.spline
            autoregs.append(autoreg)
        self.autoregs = autoregs

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        params_unnorm = []
        for i, autoreg in enumerate(self.autoregs):
            # [n x dimi]
            xi = x[:, self.blk_start[i]:self.blk_end[i]]
            # [n x dimi x param_dim], param_dim = 4 * count_bins - 1
            param_unnorm = autoreg.autoreg(xi)
            params_unnorm.append(param_unnorm)
        # [n x input_dim x param_dim]
        params_unnorm = tf.concat(params_unnorm, axis=1)
        # Transformation of the variables
        y, logdet = self.spline(x, params_unnorm)
        return y, tf.math.reduce_sum(logdet, axis=-1)

    # In PyTorch, the inverse function always caches the forward log_det_jac
    # But here inverse() should return -log_det_jac
    # Spline has already added the negative sign
    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        logdet = 0.0
        xs = []
        for i, autoreg in enumerate(self.autoregs):
            yi = y[:, self.blk_start[i]:self.blk_end[i]]
            xi, logdeti = autoreg.inverse(yi)
            logdet = logdet + logdeti
            xs.append(xi)
        x = tf.concat(xs, axis=-1)
        return x, logdet

    def copy(self):
        # Create the new object
        new_model = SplineAutoRegWithinBlockBase(*self.args)
        # Copy the parameters
        for new_var, self_var in zip(new_model.variables, self.variables):
            new_var.assign(self_var.value())
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.block import SplineAutoRegWithinBlockBase
    from temperflow.bijector import test_copy, test_simple
    tf.random.set_seed(123)
    # FIXME: Problematic if the last block has only one variable
    bij = SplineAutoRegWithinBlockBase(input_dim=9, block_size=3)
    x = tf.linspace(-5.0, 5.0, num=27)
    x = tf.reshape(x, (3, 9))
    test_copy(bij, x)
    test_simple(bij, x)
