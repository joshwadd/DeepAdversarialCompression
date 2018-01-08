import tensorflow as tf
import numpy as np
import constants as c
from tensorflow_functions import log2


class GSM_model:
    def __init__(self, no_mixtures):
        with tf.variable_scope('GSM'):
            init_mean = 0.5
            init_std = 0.2

            self.no_mixtures = no_mixtures
            self.pi = tf.nn.softmax(tf.Variable(tf.random_normal([1,no_mixtures], mean=init_mean, stddev=init_std, dtype=tf.float32)))
            self.sigma = self.elu(tf.Variable(tf.random_normal([1,no_mixtures], mean=init_mean, stddev=init_std, dtype=tf.float32)))
            self.mu = tf.Variable(tf.random_normal([1,no_mixtures], mean = 0.0, stddev = 1.0, dtype=tf.float32))

            for i in range(no_mixtures):
                tf.summary.scalar('mu_'+ str(i), self.mu[0,i])
                tf.summary.scalar('sigma_' + str(i), self.sigma[0,i])
                tf.summary.scalar('pi_' + str(i), self.pi[0,i])


    def elu(self, x, a=1.):
        e = 1e-15
        return tf.nn.elu(x)+1.+e

    def negative_log_likelihood(self, x):
        with tf.variable_scope('GSM'):

            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, x_shape[1]*x_shape[2]*x_shape[3]])

            alpha = tf.nn.softmax(tf.clip_by_value(self.pi, 1e-8, 1.0))

            exponent = log2(alpha) - 0.5*log2(2*np.pi) - log2(self.sigma) \
                -((tf.expand_dims(x,2) - self.mu)**2)/(2*(self.sigma)**2)

            log_GSM = self.log_sum_exp(exponent, axis=2)
            result = - tf.reduce_mean(tf.reduce_sum(log_GSM, axis=1))
        return result

    def log_sum_exp(self, x, axis=None):
        'Trick to increase numerical stability'
        x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
        return log2(tf.reduce_sum(tf.exp(x-x_max), axis=axis, keep_dims=True))+x_max

    def sample_data(self, no_samples):
        ds = tf.contrib.distributions
        components_list = []

        for i in range(self.no_mixtures):
            components_list.append(ds.Normal(loc=self.mu[0][i], scale=self.sigma[0][i]))

        mix_gauss = ds.Mixture(cat=ds.Categorical(probs=self.pi[0,:]), components=components_list)
        data = mix_gauss.sample(no_samples)
        return data
