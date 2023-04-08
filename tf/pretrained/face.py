import numpy as np
import tensorflow as tf
from keras import Sequential
import keras.layers as nn

import temperflow.option as opts
from temperflow.flow import Energy

# Whether to JIT compile functions
JIT = opts.opts["jit"]

class Face_Encoder(tf.Module):
    def __init__(self, latent_dim, img_size, ncomp, base_channels=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.ncomp = ncomp
        ch1 = base_channels
        ch2 = ch1 * 2
        ch4 = ch1 * 4
        ch8 = ch1 * 8

        seed = tf.random.uniform([1], 0, 2 ** 30, dtype=tf.int32)
        init = tf.keras.initializers.GlorotUniform(seed=seed[0])
        conv_args = dict(kernel_size=4, strides=2, padding="same",
                         data_format="channels_first", kernel_initializer=init)
        conv1 = nn.Conv2D(filters=ch1, **conv_args)  # out 32 x 32
        conv2 = nn.Conv2D(filters=ch2, **conv_args)  # out 16 x 16
        conv3 = nn.Conv2D(filters=ch4, **conv_args)  # out 8 x 8
        conv4 = nn.Conv2D(filters=ch8, **conv_args)  # out 4 x 4
        num_features = latent_dim * 2
        conv5 = nn.Conv2D(filters=num_features, kernel_size=4, strides=2, padding="valid",
                          data_format="channels_first", kernel_initializer=init)  # out 1 x 1

        self.features = Sequential([
            nn.InputLayer(input_shape=(3, img_size, img_size)),
            conv1,
            nn.BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5),
            nn.LeakyReLU(alpha=0.2),
            conv2,
            nn.BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5),
            nn.LeakyReLU(alpha=0.2),
            conv3,
            nn.BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5),
            nn.LeakyReLU(alpha=0.2),
            conv4,
            nn.BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5),
            nn.LeakyReLU(alpha=0.2),
            conv5,
            nn.Flatten()
        ])
        self.mu = Sequential([
            nn.InputLayer(input_shape=(num_features,)),
            nn.Dense(units=latent_dim * ncomp, kernel_initializer=init)
        ])
        # self.logvar = Sequential([
        #     nn.InputLayer(input_shape=(num_features,)),
        #     nn.Dense(units=latent_dim * ncomp, kernel_initializer=init)
        # ])

    def __call__(self, x, y, training: bool = False):
        features = self.features(x, training=training)
        mus = self.mu(features)
        mus = tf.reshape(mus, shape=(-1, self.ncomp, self.latent_dim))
        mu = tf.gather(mus, y, axis=1, batch_dims=1)
        # logvars = self.logvar(features)
        # logvars = tf.reshape(logvars, shape=(-1, self.ncomp, self.latent_dim))
        # logvar = tf.gather(logvars, y, axis=-1, batch_dims=1)
        return mu, None

    def save_params(self, path):
        # https://stackoverflow.com/a/62806106
        params = [var.numpy() for var in self.variables]
        np.savez_compressed(path, *params)
        print(f"parameters saved to {path}")

    def load_params(self, path):
        # Load parameters
        data = np.load(path)
        params = [data[k] for k in data]
        if len(self.variables) != len(params):
            raise ValueError("parameter lengths do not match")
        for var, param in zip(self.variables, params):
            var.assign(tf.constant(param))
        print(f"loaded parameters from {path}")

class Face_Generator(tf.Module):
    def __init__(self, latent_dim, base_channels=16):
        super().__init__()
        ch1 = base_channels
        ch2 = ch1 * 2
        ch4 = ch1 * 4
        ch8 = ch1 * 8

        seed = tf.random.uniform([1], 0, 2 ** 30, dtype=tf.int32)
        init = tf.keras.initializers.GlorotUniform(seed=seed[0])
        ConvT = nn.Conv2DTranspose
        conv_args = dict(kernel_size=4, strides=2, padding="same",
                         data_format="channels_first", kernel_initializer=init)
        conv0 = ConvT(filters=ch8, kernel_size=4, strides=1, padding="valid",
                      data_format="channels_first", kernel_initializer=init)  # out 4 x 4
        conv1 = ConvT(filters=ch4, **conv_args)  # out 8 x 8
        conv2 = ConvT(filters=ch2, **conv_args)  # out 16 x 16
        conv3 = ConvT(filters=ch1, **conv_args)  # out 32 x 32
        conv4 = ConvT(filters=3, **conv_args)    # out 64 x 64
        self.net = Sequential([
            nn.InputLayer(input_shape=(latent_dim,)),
            nn.Reshape(target_shape=(latent_dim, 1, 1)),
            conv0,
            nn.BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5),
            nn.ReLU(),
            conv1,
            nn.BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5),
            nn.ReLU(),
            conv2,
            nn.BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5),
            nn.ReLU(),
            conv3,
            nn.BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5),
            nn.ReLU(),
            conv4,
            nn.Reshape(target_shape=(3, 64, 64))
        ])

    def __call__(self, z, apply_tanh: bool = True, training: bool = False):
        x = self.net(z, training=training)
        x = tf.clip_by_value(x, -10.0, 10.0)
        if apply_tanh is True:
            x = tf.math.tanh(x)
        return x

    def save_params(self, path):
        # https://stackoverflow.com/a/62806106
        params = [var.numpy() for var in self.variables]
        np.savez_compressed(path, *params)
        print(f"parameters saved to {path}")

    def load_params(self, path):
        # Load parameters
        data = np.load(path)
        params = [data[k] for k in data]
        if len(self.variables) != len(params):
            raise ValueError("parameter lengths do not match")
        for var, param in zip(self.variables, params):
            var.assign(tf.constant(param))
        print(f"loaded parameters from {path}")

class FaceLatentEnergy(Energy):
    def __init__(self, bijdistr):
        super(FaceLatentEnergy, self).__init__()
        self.bijdistr = bijdistr

    @tf.function(jit_compile=JIT)
    def energy(self, x):
        return -self.bijdistr.log_prob(x)

    def log_pdf(self, x):
        return -self.energy(x)
