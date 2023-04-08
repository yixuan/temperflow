import math
import time
import numpy as np
import tensorflow as tf

import temperflow.option as opts
from temperflow.scheduler import one_cycle_lr

# Whether to JIT compile functions
JIT = opts.opts["jit"]
JIT_DEBUG = opts.opts["debug"]

# An abstract class that returns the tempered energy function value
# Users need to implement the energy() member function
class Energy:
    def __init__(self):
        super().__init__()

    def energy(self, x):
        raise NotImplementedError

    @tf.function(jit_compile=JIT)
    def tempered_energy(self, x, beta):
        return beta * self.energy(x)

    @tf.function(jit_compile=JIT)
    def __call__(self, x, beta):
        return self.tempered_energy(x, beta)

    @tf.function(jit_compile=JIT)
    def score_fn(self, x):
        logp = -self.energy(x)
        grad = tf.gradients(logp, x)
        return grad[0]

# f(x) = log|exp(x) - 1|
# For x > 0, f(x) = x + log(1 - exp(-x))
# For x < 0, f(x) = log(1 - exp(x))
# Therefore, f(x) = max(x, 0) + log(1 - exp(-|x|))
@tf.function(jit_compile=JIT)
def log_expm1(x):
    return tf.nn.relu(x) + tf.math.log(1.000001 - tf.math.exp(-tf.math.abs(x)))

# Exponential moving average
def exp_move_avg(x, fac=0.9):
    x = np.array(x)
    n = x.size
    if n < 1:
        return None
    i = np.linspace(n - 1, 0, num=n)
    weights = fac ** i
    return np.sum(x * weights) / np.sum(weights)

# Training a tempered distribution flow
class TemperFlow(tf.Module):
    def __init__(self, energy, bijdistr):
        super().__init__()
        self.energy = energy
        self.bijdistr = bijdistr
        self.ref = None
        # Initialize the optimizer with an arbitrary learning rate
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Use the current bijector-distribution as the reference distribution
    def update_ref(self):
        if self.ref is None:
            self.ref = self.bijdistr.copy()
        else:
            self.ref.copy_params_from(self.bijdistr)

    # Reset the Adam optimizer
    def reset_optimizer(self):
        for var in self.opt.variables():
            var.assign(tf.zeros_like(var))

    # JIT Compiling kl_grad() seems to be very costly, so we just turn it off
    @tf.function(jit_compile=False)
    def kl_grad(self, beta: tf.Tensor, bs: int, nbatch: int, seed: tf.Tensor, ref: bool):
        print(f"***** tracing TemperFlow.kl_grad(ref={ref}) *****") if JIT_DEBUG else None
        distr = self.ref if ref else self.bijdistr
        Es = []
        logrs = []
        for i in range(nbatch):
            theta, logr = distr.sample(bs, log_pdf=True, seed=seed + i)
            energy = self.energy.energy(theta)
            Es.append(energy)
            logrs.append(logr)

        E = tf.concat(Es, axis=0)
        logr = tf.concat(logrs, axis=0)
        logn = math.log(bs) + math.log(nbatch)
        log_ratio = -beta * E - logr
        zeta_1 = tf.math.reduce_logsumexp(-E - logr, axis=0) - logn
        zeta_beta = tf.math.reduce_logsumexp(log_ratio, axis=0) - logn
        ratio = tf.math.softmax(log_ratio - zeta_beta, axis=0) * bs * nbatch

        Emean = tf.math.reduce_mean(ratio * E)
        kl = (1.0 - beta) * Emean + zeta_1 - zeta_beta
        grad = (beta - 1.0) * tf.math.reduce_mean(
            ratio * tf.math.squared_difference(E, Emean))
        return kl, grad

    @tf.function(jit_compile=JIT)
    def sampling(self, beta, bs, seed, ref):
        print(f"***** tracing TemperFlow.sampling(ref={ref}) *****") if JIT_DEBUG else None
        if not ref:
            # Simple Monte Carlo estimator
            theta, logq = self.bijdistr.sample(bs, log_pdf=True, seed=seed)
            logr = logq
        else:
            # Importance sampling estimator
            theta, logr = self.ref.sample(bs, log_pdf=True, seed=seed)
            theta = tf.stop_gradient(theta)
            logr = tf.stop_gradient(logr)
            logq = self.bijdistr.log_prob(theta)

        energy = self.energy(theta, beta)
        return energy, logr, logq

    @staticmethod
    @tf.function(jit_compile=JIT)
    def log_partition(logr, energy, oldval, decay=0.8):
        print("***** tracing TemperFlow.log_partition() *****") if JIT_DEBUG else None
        bs = logr.shape[0]
        logr = tf.stop_gradient(logr)
        energy = tf.stop_gradient(energy)
        logz = tf.math.reduce_logsumexp(-energy - logr, axis=0) - math.log(bs)
        logz = tf.where(tf.math.is_inf(oldval),
                        logz,
                        decay * logz + (1.0 - decay) * oldval)
        return logz

    @tf.function(jit_compile=JIT)
    def loss_kl(self, beta, bs, seed, ref):
        print("***** tracing TemperFlow.loss_kl() *****") if JIT_DEBUG else None
        energy, logr, logq = self.sampling(beta, bs, seed, ref)

        if not ref:
            loss = tf.math.reduce_mean(logq + energy)
        else:
            ratio = tf.math.exp(logq - logr)
            loss = tf.math.reduce_mean(ratio * (logq + energy))
        return loss

    @tf.function(jit_compile=JIT)
    def loss_h2(self, beta, bs, seed, ref):
        print("***** tracing TemperFlow.loss_h2() *****") if JIT_DEBUG else None
        energy, logr, logq = self.sampling(beta, bs, seed, ref)

        log_pqratio = -energy - logq
        if not ref:
            loss = tf.math.reduce_logsumexp(0.5 * log_pqratio) - math.log(bs)
        else:
            log_ratio = logq - logr
            loss = tf.math.reduce_logsumexp(log_ratio + 0.5 * log_pqratio) - math.log(bs)
        return -loss

    @tf.function(jit_compile=JIT)
    def loss_l2(self, beta, bs, seed, ref, logz):
        print("***** tracing TemperFlow.loss_l2() *****") if JIT_DEBUG else None
        energy, logr, logq = self.sampling(beta, bs, seed, ref)
        new_logz = self.log_partition(logr, energy, oldval=logz, decay=0.8)
        new_logz = tf.stop_gradient(new_logz)

        # ||f-g||^2 = E_r[(f-g)^2/r], g is the target distribution
        # log[(f-g)^2/r] = 2 * log|f-g| - log(r) = 2 * log(f) + 2 * log|1 - g/f| - log(r)
        # z = E_r[exp(-energy) / r] = E_r[exp(-energy - logr)]
        logg = -energy - new_logz
        w = 2.0 * tf.math.maximum(logg, logq) + 2.0 * log_expm1(logg - logq) - logr
        loss = tf.math.reduce_logsumexp(w, axis=0) - math.log(bs)  # log(l2)
        return loss, new_logz

    # There are some issues when JIT compiling is applied to tf.GradientTape(),
    # so here just turn it off
    @tf.function(jit_compile=False)
    def train_one_batch(self, beta, fn, ref, bs, nepoch, lr, logz):
        print("***** tracing TemperFlow.train_one_batch() *****") if JIT_DEBUG else None

        # Update learning rate according to the scheduler
        # self.opt.learning_rate = cosine_decay(
        #     self.opt.iterations, lr_init=lr, decay_steps=nepoch, alpha=tf.constant(0.0))
        self.opt.learning_rate = one_cycle_lr(
            self.opt.iterations, max_lr=lr, total_steps=nepoch)

        seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
        with tf.GradientTape() as tape:
            if fn == "l2":
                loss, new_logz = self.loss_l2(beta, bs, seed, ref, logz)
            elif fn == "kl":
                loss, new_logz = self.loss_kl(beta, bs, seed, ref), tf.constant(np.inf)
            elif fn == "h2":
                loss, new_logz = self.loss_h2(beta, bs, seed, ref), tf.constant(np.inf)
            elif fn == "l2+kl":
                loss_l2, new_logz = self.loss_l2(beta, bs, seed, ref, logz)
                loss_kl = self.loss_kl(beta, bs, seed, ref=False)
                loss = loss_l2 + loss_kl
            else:
                raise NotImplementedError

        grad = tape.gradient(loss, self.bijdistr.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.bijdistr.trainable_variables))
        return loss, new_logz

    def train_one_beta(self, beta, fn="kl", ref=False,
                       bs=1000, nepoch=1001, lr=0.001, verbose=False):
        beta = tf.constant(beta)
        nepoch = tf.constant(nepoch)
        lr = tf.constant(lr)

        self.reset_optimizer()
        losses = []
        logz = tf.constant(np.inf)

        for i in range(nepoch):
            loss, logz = self.train_one_batch(beta, fn, ref, bs, nepoch, lr, logz)

            lossi = loss.numpy()
            losses.append(float(lossi))
            if verbose and i % 50 == 0:
                if fn == "l2":
                    print(f"epoch {i}, logz = {logz}, loss = {lossi}")
                else:
                    print(f"epoch {i}, loss = {lossi}")
        return losses

    def train_flow(self, beta0=0.1, kl_decay=0.8, nepoch=(2001, 2001, 2001),
                   bs=1000, lr=0.0001, max_betas=100, verbose=1,
                   ref_cutoff=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001],
                   checkpoint=None, recover=False):
        t0 = time.time()
        # Epochs at different stages
        if type(nepoch) is tuple:
            nepoch_init, nepoch_mid, nepoch_final = nepoch
        else:
            nepoch_init, nepoch_mid, nepoch_final = nepoch, nepoch, nepoch

        # History of log(beta)
        logbeta = math.log(beta0)
        logbetas = [logbeta]
        deltas = []
        # History of KL divergence
        kls = []
        # History of losses
        losses = []
        # Elapsed times
        times = []

        # Reference distributions for importance sampling
        if ref_cutoff is not None:
            ref_cutoff = ref_cutoff.copy()

        for i in range(max_betas):
            t1 = time.time()
            beta = math.exp(logbeta)
            print(f"i = {i}, logbeta = {logbeta}, beta = {beta}\n") if verbose > 0 else None

            # Train one tempered distribution
            # Heuristic choice of adaptive learning rate
            lri = lr / math.sqrt(beta) * math.pow(0.9, i)
            fni, refi, nepochi = ("kl", False, nepoch_init) if i == 0 else ("l2", True, nepoch_mid)
            if beta < 0.5:
                nepochi *= 2
            lossesi = self.train_one_beta(beta, fn=fni, ref=refi, bs=bs, nepoch=nepochi, lr=lri,
                                          verbose=(verbose > 1))
            losses.append(lossesi)

            if checkpoint is not None:
                self.bijdistr.save_params(f"{checkpoint}_{i}.npz")

            # Update reference distribution after the first iteration
            if i == 0:
                self.update_ref()

            # Estimate KL divergence
            seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
            kl, kl_grad = self.kl_grad(tf.constant(beta), bs=bs, nbatch=10, seed=seed, ref=True)
            kl, kl_grad = kl.numpy(), kl_grad.numpy()
            kls.append(float(kl))

            # Compute delta
            delta_abs_avg = exp_move_avg(np.abs(deltas), fac=0.9)
            delta = -(1 - kl_decay) * kl / beta / kl_grad
            delta = np.minimum(delta, 1.0) if i == 0 else delta
            if delta_abs_avg is not None:
                delta = np.minimum(math.fabs(delta), 2.0 * delta_abs_avg)
                delta = np.maximum(delta, 0.5 * delta_abs_avg)
            deltas.append(delta)
            t2 = time.time()
            times.append(t2 - t1)
            if verbose > 0:
                print(f"\nkl = {kl}, kl_grad = {kl_grad}")
                print(f"delta = {delta}, delta_avg = {delta_abs_avg}")
                print(f"{t2 - t1} seconds\n")

            # Update beta
            logbeta += delta
            if logbeta >= 0.0:
                break
            logbetas.append(logbeta)

            # If ref_cutoff is None, update the reference distribution at every beta
            if ref_cutoff is None:
                print("\n****** Reference distribution updated ******\n") if verbose > 0 else None
                self.update_ref()
            elif len(ref_cutoff) > 0 and kl < ref_cutoff[0] * kls[0]:
                print("\n****** Reference distribution updated ******\n") if verbose > 0 else None
                self.update_ref()
                ref_cutoff = ref_cutoff[1:]

        # beta=1
        print("\n****** Final training with beta=1 ******\n") if verbose > 0 else None
        t1 = time.time()
        lri = lr * math.pow(0.9, i + 1)
        lossesi1 = self.train_one_beta(beta=1.0, fn="l2", ref=True, bs=bs, nepoch=nepoch_mid, lr=lri, verbose=(verbose > 1))
        t2 = time.time()
        times.append(t2 - t1)

        # Final refining
        t1 = time.time()
        lri = lr * math.pow(0.9, i + 1)
        lossesi2 = self.train_one_beta(beta=1.0, fn="kl", ref=False, bs=bs, nepoch=nepoch_final, lr=lri, verbose=(verbose > 1))
        t2 = time.time()
        times.append(t2 - t1)

        losses.append(lossesi1)
        losses.append(lossesi2)

        trace = dict(logbeta=logbetas, kl=kls,
                     loss_init=losses[0], loss=losses[1:],
                     time=times, total_time=t2 - t0)
        return trace
