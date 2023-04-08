import math
import tensorflow as tf

import temperflow.option as opts

# Whether to JIT compile functions
JIT = opts.opts["jit"]
JIT_DEBUG = opts.opts["debug"]

# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
# https://github.com/keras-team/keras/blob/v2.9.0/keras/optimizers/schedules/learning_rate_schedule.py#L550-L641
@tf.function(jit_compile=JIT)
def cosine_decay(step, init_lr, decay_steps, alpha=0.0):
    dtype = init_lr.dtype
    decay_steps = tf.cast(decay_steps, dtype)
    PI = tf.constant(math.pi, dtype=dtype)

    global_step_recomp = tf.cast(step, dtype)
    global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
    completed_fraction = global_step_recomp / decay_steps
    cosine_decayed = 0.5 * (1.0 + tf.cos(PI * completed_fraction))

    decayed = (1.0 - alpha) * cosine_decayed + alpha
    return tf.multiply(init_lr, decayed)

# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
# https://github.com/benihime91/one_cycle_lr-tensorflow
@tf.function(jit_compile=JIT)
def one_cycle_lr(step, max_lr, total_steps, pct_start=0.3,
                 div_factor=25.0, final_div_factor=1e4):
    dtype = max_lr.dtype
    total_steps = tf.cast(total_steps, dtype)
    step = tf.cast(step, dtype)
    PI = tf.constant(math.pi, dtype=dtype)

    # Number of steps that increase and decrease learning rate
    step_size_up = pct_start * total_steps - 1.0
    step_size_down = total_steps - step_size_up - 1.0

    # Initialize learning rate variables
    init_lr = max_lr / div_factor
    min_lr = init_lr / final_div_factor

    # Whether the learning rate is now increasing or decreasing
    is_up = (step <= step_size_up)
    completed_fraction = tf.where(is_up,
        step / step_size_up, (step - step_size_up) / step_size_down)
    cosine_decayed = 0.5 * (1.0 + tf.cos(PI * completed_fraction))
    lr_start = tf.where(is_up, init_lr, max_lr)
    lr_end = tf.where(is_up, max_lr, min_lr)
    lr = (lr_start - lr_end) * cosine_decayed + lr_end
    return lr
