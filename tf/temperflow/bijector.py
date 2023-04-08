import numpy as np
import tensorflow as tf
import keras.layers as nn
import tensorflow_probability as tfp
tfb = tfp.bijectors
from copy import deepcopy

import temperflow.option as opts
from temperflow.spline import LinearRationalSpline, QuadraticRationalSpline
from temperflow.block import Dense, AffineCouplingBase, AutoRegAffineBase, \
    SplineAutoRegBase, SplineBlockAutoRegBase, SplineAutoRegWithinBlockBase

# Whether to JIT compile functions
JIT = opts.opts["jit"]
JIT_DEBUG = opts.opts["debug"]

# Demonstration
if __name__ == "__main__":
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfb = tfp.bijectors

    tf.random.set_seed(123)
    bij = tfb.Softplus()
    x = tf.linspace(-5.0, 5.0, num=15)
    x = tf.reshape(x, (5, 3))
    y = bij(x)
    print("x =\n", x)
    print()
    print("y =\n", y)
    print()
    logdet = bij.forward_log_det_jacobian(x)
    print("logdet =\n", logdet)
    print()
    logdet = bij.inverse_log_det_jacobian(y)
    print("logdet =\n", logdet)

    import torch
    import pyro.distributions as dist

    bij = dist.transforms.SoftplusTransform()
    x = torch.linspace(-5.0, 5.0, 15).reshape(5, 3)
    y = bij(x)
    print("x =\n", x)
    print()
    print("y =\n", y)
    print()
    logdet = bij.log_abs_det_jacobian(x, y)
    print("logdet =\n", logdet)
    print()
    logdet = bij.inv.log_abs_det_jacobian(y, x)
    print("logdet =\n", logdet)

class ExpDemo(tfb.Bijector):
    def __init__(self, validate_args=False, name='exp'):
        super(ExpDemo, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            name=name)

    def _forward(self, x):
        print("_forward() called")
        return tf.math.exp(x)

    def _inverse(self, y):
        print("_inverse() called")
        return tf.math.log(y)

    def _inverse_log_det_jacobian(self, y):
        print("_inverse_log_det_jacobian() called")
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
        print("_forward_log_det_jacobian() called")
        return x

# Test caching
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import ExpDemo

    bij = ExpDemo()
    x = tf.linspace(-5.0, 5.0, num=15)
    x = tf.reshape(x, (5, 3))
    y = bij(x)  # _forward() will be called
    print("x =\n", x)
    print()
    print("y =\n", y)
    print()
    xrec = bij.inverse(y)  # _inverse() will NOT be called
    print("xrec =\n", xrec)
    print()
    xrec = bij.inverse(tf.identity(y))  # _inverse() will be called
    print("xrec =\n", xrec)
    print()
    logdet = bij.forward_log_det_jacobian(x)
    print("logdet =\n", logdet)
    print()
    logdet = bij.inverse_log_det_jacobian(y)
    print("logdet =\n", logdet)

# Wrapper of a tfb bijector
class TFBWrapper(tf.Module):
    def __init__(
        self, tfb_bijector
    ):
        super().__init__()
        self.bijector = tfb_bijector

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing TFBWrapper.__call__() *****") if JIT_DEBUG else None
        logdet = self.bijector.forward_log_det_jacobian(x)
        y = self.bijector(x)
        return y, tf.reduce_sum(logdet, axis=-1)

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing TFBWrapper.inverse() *****") if JIT_DEBUG else None
        logdet = self.bijector.inverse_log_det_jacobian(y)
        x = self.bijector.inverse(y)
        return x, tf.reduce_sum(logdet, axis=-1)

    def copy(self):
        return TFBWrapper(deepcopy(self.bijector))

def test_simple(bij, x):
    y, logdet = bij(x)
    xrec, logdet_inv = bij.inverse(y)
    print("y =\n", y)
    print()
    print("xrec =\n", xrec)
    print()
    print("x =\n", x)
    print()
    print("logdet =\n", logdet)
    print()
    print("logdet_inv =\n", logdet_inv)
    print()
    print("||-logdet - logdet_inv|| =", tf.linalg.norm(logdet + logdet_inv).numpy())
    print("||x - xrec|| =", tf.linalg.norm(x - xrec).numpy())

def test_copy(bij, x):
    bij2 = bij.copy()

    y, logdet = bij(x)
    xrec, logdet_inv = bij.inverse(y)

    y2, logdet2 = bij2(x)
    xrec2, logdet_inv2 = bij2.inverse(y2)

    ydiff = tf.linalg.norm(y2 - y).numpy()
    lddiff = tf.linalg.norm(logdet2 - logdet).numpy()
    xrdiff = tf.linalg.norm(xrec2 - xrec).numpy()
    lgidiff = tf.linalg.norm(logdet_inv2 - logdet_inv).numpy()
    print(f"y_diff = {ydiff}")
    print(f"logdet_diff = {lddiff}")
    print(f"xrec_diff = {xrdiff}")
    print(f"logdet_inv_diff = {lgidiff}")

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import ExpDemo, TFBWrapper
    from temperflow.bijector import test_copy, test_simple

    bij = TFBWrapper(ExpDemo())
    x = tf.linspace(-5.0, 5.0, num=15)
    x = tf.reshape(x, (5, 3))
    test_copy(bij, x)
    test_simple(bij, x)

# The inverse bijector
class Inverse(tf.Module):
    def __init__(
        self, bijector
    ):
        super().__init__()
        self.bijector = bijector

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing Inverse.__call__() *****") if JIT_DEBUG else None
        return self.bijector.inverse(x)

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing Inverse.inverse() *****") if JIT_DEBUG else None
        return self.bijector(y)

    def copy(self):
        return Inverse(self.bijector.copy())

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfb = tfp.bijectors
    from temperflow.bijector import ExpDemo, TFBWrapper, Inverse
    from temperflow.bijector import test_copy, test_simple

    bij = Inverse(TFBWrapper(ExpDemo()))
    x = tf.linspace(1.0, 10.0, num=15)
    x = tf.reshape(x, (5, 3))
    test_copy(bij, x)
    test_simple(bij, x)

    bij2 = TFBWrapper(tfb.Log())
    test_simple(bij2, x)

# Composition of bijectors
class Composition(tf.Module):
    def __init__(
        self, bijectors
    ):
        super().__init__()
        self.bijectors = bijectors
        self.len = len(bijectors)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing Composition.__call__() *****") if JIT_DEBUG else None
        y, logdet = self.bijectors[0](x)
        for i in range(1, self.len):
            y, logdeti = self.bijectors[i](y)
            logdet = logdet + logdeti
        return y, logdet

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing Composition.inverse() *****") if JIT_DEBUG else None
        x, logdet = self.bijectors[-1].inverse(y)
        for i in range(self.len - 2, -1, -1):
            x, logdeti = self.bijectors[i].inverse(x)
            logdet = logdet + logdeti
        return x, logdet

    def copy(self):
        return Composition([bij.copy() for bij in self.bijectors])

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfb = tfp.bijectors
    from temperflow.bijector import TFBWrapper, Composition
    from temperflow.bijector import test_copy, test_simple

    bij = Composition([TFBWrapper(tfb.Softplus()),
                       TFBWrapper(tfb.Log()),
                       TFBWrapper(tfb.Sigmoid())])
    x = tf.linspace(-5.0, 5.0, num=15)
    x = tf.reshape(x, (5, 3))
    test_copy(bij, x)
    test_simple(bij, x)

    # Notice the order
    bij2 = TFBWrapper(tfb.Chain([tfb.Sigmoid(), tfb.Log(), tfb.Softplus()]))
    test_simple(bij2, x)

# Update the first partial_dim dimensions using the given bijector
class Partial(tf.Module):
    def __init__(
        self, bijector, partial_dim=None
    ):
        super().__init__()
        self.bijector = bijector
        self.partial_dim = partial_dim

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing Partial.__call__() *****") if JIT_DEBUG else None
        p = x.shape[-1]
        if self.partial_dim is None or self.partial_dim >= p:
            y, logdet = self.bijector(x)
        else:
            yp, logdet = self.bijector(x[:, :self.partial_dim])
            y = tf.concat((yp, x[:, self.partial_dim:]), axis=-1)
        return y, logdet

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing Partial.inverse() *****") if JIT_DEBUG else None
        p = y.shape[-1]
        if self.partial_dim is None or self.partial_dim >= p:
            x, logdet = self.bijector.inverse(y)
        else:
            xp, logdet = self.bijector.inverse(y[:, :self.partial_dim])
            x = tf.concat((xp, y[:, self.partial_dim:]), axis=-1)
        return x, logdet

    def copy(self):
        return Partial(self.bijector.copy(), self.partial_dim)

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfb = tfp.bijectors
    from temperflow.bijector import TFBWrapper, Partial
    from temperflow.bijector import test_copy, test_simple

    bij = Partial(TFBWrapper(tfb.Sigmoid()), 3)
    x = tf.linspace(-5.0, 5.0, num=15)
    x = tf.reshape(x, (3, 5))
    test_copy(bij, x)
    test_simple(bij, x)

    bij2 = TFBWrapper(tfb.Sigmoid())
    test_simple(bij2, x[:, :3])

# Permutation bijector
class Permute(tf.Module):
    def __init__(
        self, perm
    ):
        super().__init__()
        # tf.Variable of type int32 will always be on CPU
        # Need to use int64 if jit_compile=True is used
        # See https://www.tensorflow.org/xla/known_issues
        perm = tf.constant(perm, dtype=tf.int64)
        self.perm = tf.Variable(perm, trainable=False)
        self.inv_perm = tf.Variable(tf.math.invert_permutation(perm), trainable=False)
        self.partial_dim = self.perm.shape[0]

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing Permute.__call__() *****") if JIT_DEBUG else None
        n = x.shape[0]
        p = x.shape[-1]
        if self.partial_dim >= p:
            y = tf.gather(x, self.perm, axis=-1)
        else:
            yp = tf.gather(x[:, :self.partial_dim], self.perm, axis=-1)
            y = tf.concat((yp, x[:, self.partial_dim:]), axis=-1)
        logdet = tf.zeros(n)
        return y, logdet

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing Permute.inverse() *****") if JIT_DEBUG else None
        n = y.shape[0]
        p = y.shape[-1]
        if self.partial_dim >= p:
            x = tf.gather(y, self.inv_perm, axis=-1)
        else:
            xp = tf.gather(y[:, :self.partial_dim], self.inv_perm, axis=-1)
            x = tf.concat((xp, y[:, self.partial_dim:]), axis=-1)
        logdet = tf.zeros(n)
        return x, logdet

    def copy(self):
        return Permute(tf.identity(self.perm.value()))

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import Permute
    from temperflow.bijector import test_copy, test_simple

    bij = Permute([2, 0, 1])
    x = tf.linspace(-5.0, 5.0, num=15)
    x = tf.reshape(x, (3, 5))
    test_copy(bij, x)
    test_simple(bij, x)

    bij2 = Permute([2, 0, 4, 3, 1])
    test_simple(bij2, x)



# Transformation mapping based on affine coupling layers
class AffineCoupling(tf.Module):
    def __init__(
        self, dim, depth=10, hlayers=[32, 32], nonlinearity=nn.ReLU
    ):
        super().__init__()
        self.args = (dim, depth, hlayers, nonlinearity)

        # The list of transformations
        transforms = []
        for i in range(depth):
            # Permutation transformation
            perm = [1, 0] if dim == 2 else np.random.permutation(dim)
            perm = Permute(perm)
            transforms.append(perm)
            # Spline coupling transformation
            keep_dim = (dim - 1) // 2 + 1
            ac = AffineCouplingBase(
                input_dim=dim, keep_dim=keep_dim,
                hlayers=hlayers, nonlinearity=nonlinearity)
            transforms.append(ac)

        self.comp = Composition(transforms)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing AffineCoupling.__call__() *****") if JIT_DEBUG else None
        return self.comp(x)

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing AffineCoupling.inverse() *****") if JIT_DEBUG else None
        return self.comp.inverse(y)

    def copy(self):
        new_model = AffineCoupling(*self.args)
        new_model.comp = self.comp.copy()
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import AffineCoupling
    from temperflow.bijector import test_copy, test_simple
    tf.random.set_seed(123)
    bij = AffineCoupling(dim=8, depth=10)
    x = tf.linspace(-5.0, 5.0, num=24)
    x = tf.reshape(x, (3, 8))
    test_copy(bij, x)
    test_simple(bij, x)

# Transformation mapping based on affine coupling layers
class AutoRegAffine(tf.Module):
    def __init__(
        self, dim, depth=10, hlayers=[32, 32], nonlinearity=nn.ReLU
    ):
        super().__init__()
        self.args = (dim, depth, hlayers, nonlinearity)

        # The list of transformations
        transforms = []
        for i in range(depth):
            # Permutation transformation
            perm = [1, 0] if dim == 2 else np.random.permutation(dim)
            perm = Permute(perm)
            transforms.append(perm)
            # Autoregressive affine transformation
            ac = AutoRegAffineBase(
                input_dim=dim, hlayers=hlayers, nonlinearity=nonlinearity)
            transforms.append(Inverse(ac))

        self.comp = Composition(transforms)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing AutoRegAffine.__call__() *****") if JIT_DEBUG else None
        return self.comp(x)

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing AutoRegAffine.inverse() *****") if JIT_DEBUG else None
        return self.comp.inverse(y)

    def copy(self):
        new_model = AutoRegAffine(*self.args)
        new_model.comp = self.comp.copy()
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import AutoRegAffine
    from temperflow.bijector import test_copy, test_simple
    tf.random.set_seed(123)
    bij = AutoRegAffine(dim=8, depth=10)
    x = tf.linspace(-5.0, 5.0, num=24)
    x = tf.reshape(x, (3, 8))
    test_copy(bij, x)
    test_simple(bij, x)

# Transformation mapping based on 2D spline
class Spline2D(tf.Module):
    def __init__(
        self, count_bins=8, bound=None, hlayers=[32, 32],
        nonlinearity=nn.ReLU, spline="linear"
    ):
        super().__init__()
        self.args = (count_bins, bound, hlayers, nonlinearity, spline)

        # If bound is not given, we set bound=5 and then add an affine transformation
        self.unbounded = False
        self.bound = bound
        if bound is None:
            self.unbounded = True
            self.bound = 5.0
            self.loc = tf.Variable(tf.zeros([2]))
            # scale=softplus(uscale), and softplus(0.54132)~=1.0
            self.uscale = tf.Variable(tf.constant([0.54132, 0.54132]))

        # Linear rational spline
        if spline == "linear":
            self.spline = LinearRationalSpline(num_bins=count_bins, bound=self.bound)
            param_dim = 4 * count_bins - 1
        else:
            self.spline = QuadraticRationalSpline(num_bins=count_bins, bound=self.bound)
            param_dim = 3 * count_bins - 1
        # Spline for the first variable, f1(x1)
        self.params_unnorm = tf.Variable(tf.random.normal((1, param_dim), stddev=0.01))
        # (Conditional) spline for the second variable, f2(x2)
        self.net = Dense(in_dim=1, hlayers=hlayers, out_dim=param_dim, nonlinearity=nonlinearity)

    # x [n x 2]
    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        x1 = x[:, :1]
        x2 = x[:, 1:]
        # First variable
        y1, logdet1 = self.spline(x1, self.params_unnorm)
        logdet1 = tf.squeeze(logdet1)
        # Second variable
        params_unnorm = self.net(x1)
        y2, logdet2 = self.spline(x2, tf.expand_dims(params_unnorm, axis=-2))
        logdet2 = tf.squeeze(logdet2)
        # Combine result
        y = tf.concat((y1, y2), axis=1)
        if self.unbounded:
            scale = tf.math.softplus(self.uscale)
            y = scale * y + self.loc
            logdet = logdet1 + logdet2 + tf.math.reduce_sum(tf.math.log(scale))
        else:
            logdet = logdet1 + logdet2
        return y, logdet

    # y [n x 2]
    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        if self.unbounded:
            scale = tf.math.softplus(self.uscale)
            y = (y - self.loc) / scale
        y1 = y[:, :1]
        y2 = y[:, 1:]
        # First variable
        x1, logdet1 = self.spline.inverse(y1, self.params_unnorm)
        logdet1 = tf.squeeze(logdet1)
        # Second variable
        params_unnorm = self.net(x1)
        x2, logdet2 = self.spline.inverse(y2, tf.expand_dims(params_unnorm, axis=-2))
        logdet2 = tf.squeeze(logdet2)
        # Combine result
        x = tf.concat((x1, x2), axis=1)
        if self.unbounded:
            logdet = logdet1 + logdet2 - tf.math.reduce_sum(tf.math.log(scale))
        else:
            logdet = logdet1 + logdet2
        return x, logdet

    def copy(self):
        # Create the new object
        new_model = Spline2D(*self.args)
        # Copy the parameters
        for new_var, self_var in zip(new_model.variables, self.variables):
            new_var.assign(self_var.value())
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import Spline2D
    from temperflow.bijector import test_copy, test_simple
    tf.random.set_seed(123)
    bij = Spline2D(count_bins=8, bound=None, hlayers=[32, 32])
    x = tf.linspace(-6.0, 6.0, num=10)
    x = tf.reshape(x, (5, 2))
    test_copy(bij, x)
    test_simple(bij, x)

    bij = Spline2D(count_bins=8, bound=None, hlayers=[32, 32], spline="quadratic")
    test_copy(bij, x)
    test_simple(bij, x)

# Transformation mapping based on autoregressive splines
class AutoRegSpline(tf.Module):
    def __init__(
        self, dim, perm_dim=None, count_bins=8, bound=6.0,
        hlayers=[32, 32], nonlinearity=nn.ReLU, spline="linear"
    ):
        super().__init__()
        self.args = (dim, perm_dim, count_bins, bound,
                     hlayers, nonlinearity, spline)

        if perm_dim is None:
            spline1 = SplineAutoRegBase(
                dim, count_bins=count_bins, bound=bound, hlayers=hlayers,
                nonlinearity=nonlinearity, first_identity=False, spline=spline)
            transforms = [Inverse(spline1)]
        else:
            spline1 = SplineAutoRegBase(
                dim, count_bins=count_bins, bound=bound, hlayers=hlayers,
                nonlinearity=nonlinearity, first_identity=True, spline=spline)
            partial_dim = min(perm_dim, dim)
            partial_dim = max(2, partial_dim)
            perm = list(reversed(range(partial_dim)))
            perm = Permute(perm)
            spline2 = SplineAutoRegBase(
                partial_dim, count_bins=count_bins, bound=bound, hlayers=hlayers,
                nonlinearity=nonlinearity, first_identity=True, spline=spline)
            partial = Partial(spline2, partial_dim)
            transforms = [Inverse(spline1), perm, Inverse(partial)]

        self.comp = Composition(transforms)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing AutoRegSpline.__call__() *****") if JIT_DEBUG else None
        return self.comp(x)

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing AutoRegSpline.inverse() *****") if JIT_DEBUG else None
        return self.comp.inverse(y)

    def copy(self):
        new_model = AutoRegSpline(*self.args)
        new_model.comp = self.comp.copy()
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import AutoRegSpline
    from temperflow.bijector import test_copy, test_simple

    bij = AutoRegSpline(dim=5)
    x = tf.linspace(-5.0, 5.0, num=15)
    x = tf.reshape(x, (3, 5))
    test_copy(bij, x)
    test_simple(bij, x)

    bij2 = AutoRegSpline(dim=5, perm_dim=2)
    test_copy(bij2, x)
    test_simple(bij2, x)



    import tensorflow as tf
    c = tf.Variable(3.0)
    sig1 = dict(a=tf.constant(1.0), b=tf.constant(2.0))
    sig2 = {"Variable:0": tf.constant(3.0)}
    @tf.function(jit_compile=True)
    def f(sig1, sig2):
        return sig1["a"] + sig1["b"] + sig2[c.name]
    f(sig1, sig2)
    print(f.pretty_printed_concrete_signatures())
    print(f.experimental_get_compiler_ir(sig1, sig2)(stage="optimized_hlo"))


# Transformation mapping based on autoregressive splines
class BlockAutoRegSpline(tf.Module):
    def __init__(
        self, dim, block_size, count_bins=8, bound=6.0,
        hlayers=[32, 32], nonlinearity=nn.ReLU, spline="linear"
    ):
        super().__init__()
        self.args = (dim, block_size, count_bins, bound,
                     hlayers, nonlinearity, spline)

        half_hlayers = [h // 2 for h in hlayers]
        args1 = self.args
        args2 = (dim, block_size, count_bins, bound, half_hlayers, nonlinearity, spline)

        # First layer: block autoregressive
        spline1 = SplineBlockAutoRegBase(*args1)
        # Second layer: autoregressive within block
        spline2 = SplineAutoRegWithinBlockBase(*args2)

        partial_dim = 2
        perm = Permute([1, 0])
        spline3 = SplineAutoRegBase(
            partial_dim, count_bins=count_bins, bound=bound, hlayers=half_hlayers,
            nonlinearity=nonlinearity, first_identity=True, spline=spline)
        partial = Partial(spline3, partial_dim)
        transforms = [Inverse(spline1), Inverse(spline2), perm, Inverse(partial)]

        self.comp = Composition(transforms)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing BlockAutoRegSpline.__call__() *****") if JIT_DEBUG else None
        return self.comp(x)

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing BlockAutoRegSpline.inverse() *****") if JIT_DEBUG else None
        return self.comp.inverse(y)

    def copy(self):
        new_model = BlockAutoRegSpline(*self.args)
        new_model.comp = self.comp.copy()
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import BlockAutoRegSpline
    from temperflow.bijector import test_copy, test_simple

    bij = BlockAutoRegSpline(dim=9, block_size=3)
    x = tf.linspace(-5.0, 5.0, num=27)
    x = tf.reshape(x, (3, 9))
    test_copy(bij, x)
    test_simple(bij, x)

# Transformation mapping based on spline coupling layers
class SplineCoupling(tf.Module):
    def __init__(
        self, dim, depth=10, count_bins=8, bound=6.0,
        hlayers=[32, 32], nonlinearity=nn.ReLU, spline="linear"
    ):
        super().__init__()
        self.args = (dim, depth, count_bins, bound,
                     hlayers, nonlinearity, spline)

        # The list of transformations
        transforms = []
        for i in range(depth):
            # Permutation transformation
            perm = [1, 0] if dim == 2 else np.random.permutation(dim)
            perm = Permute(perm)
            transforms.append(perm)
            # Spline coupling transformation
            keep_dim = (dim - 1) // 2 + 1
            sc = SplineBlockAutoRegBase(
                input_dim=dim, block_size=keep_dim, count_bins=count_bins,
                bound=bound, hlayers=hlayers, nonlinearity=nonlinearity,
                spline=spline)
            transforms.append(sc)

        self.comp = Composition(transforms)

    @tf.function(jit_compile=JIT)
    def __call__(self, x):
        print("***** tracing SplineCoupling.__call__() *****") if JIT_DEBUG else None
        return self.comp(x)

    @tf.function(jit_compile=JIT)
    def inverse(self, y):
        print("***** tracing SplineCoupling.inverse() *****") if JIT_DEBUG else None
        return self.comp.inverse(y)

    def copy(self):
        new_model = SplineCoupling(*self.args)
        new_model.comp = self.comp.copy()
        return new_model

# Unit test
if __name__ == "__main__":
    import tensorflow as tf
    from temperflow.bijector import SplineCoupling
    from temperflow.bijector import test_copy, test_simple

    bij = SplineCoupling(dim=8, depth=10)
    x = tf.linspace(-5.0, 5.0, num=24)
    x = tf.reshape(x, (3, 8))
    test_copy(bij, x)
    test_simple(bij, x)
