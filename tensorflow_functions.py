import tensorflow as tf
import numpy as np
import constants as c


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum =0.9, name ='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name
    def __call__(self, x, train=True):
        x = tf.clip_by_value( x, -100., 100.)
        with tf.variable_scope(self.name):
            return tf.contrib.layers.batch_norm(x,
                                                decay=self.momentum,
                                                updates_collections=None,
                                                epsilon=self.epsilon,
                                                scale=True,
                                                is_training=train,
                                                scope=self.name)


def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def Quantize_model(input_layer):
    noise = tf.random_uniform(shape=tf.shape(input_layer), minval=0, maxval=1, dtype=tf.float32, name='quantisation_noise')
    return input_layer + noise





def new_conv_layer( bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
    with tf.variable_scope( name ):
        w = tf.get_variable(
                "W",
                shape=filter_shape,
                initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable(
                "b",
                shape=filter_shape[-1],
                initializer=tf.constant_initializer(0.))

        conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
        bias = activation(tf.nn.bias_add(conv, b))
        tf.summary.histogram(name + '_W', w)
        tf.summary.histogram(name + '_b', b)

    return bias


def new_deconv_layer( bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
    with tf.variable_scope(name):
        W = tf.get_variable(
                "W",
                shape=filter_shape,
                initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable(
                "b",
                shape=filter_shape[-2],
                initializer=tf.constant_initializer(0.))
        deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
        bias = activation(tf.nn.bias_add(deconv, b))
        tf.summary.histogram(name + '_W', W)
        tf.summary.histogram(name + '_b', b)

    return bias


def new_fc_layer( bottom, output_size, name ):
    shape = bottom.get_shape().as_list()
    dim = np.prod( shape[1:] )
    x = tf.reshape( bottom, [-1, dim])
    input_size = dim

    with tf.variable_scope(name):
        w = tf.get_variable(
                "W",
                shape=[input_size, output_size],
                initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable(
                "b",
                shape=[output_size],
                initializer=tf.constant_initializer(0.))
        fc = tf.nn.bias_add( tf.matmul(x, w), b)
        tf.summary.histogram(name + '_W', w)
        tf.summary.histogram(name + '_b', b)

    return fc

def channel_wise_fc_layer( input, name): # bottom: (7x7x512)
    _, width, height, n_feat_map = input.get_shape().as_list()
    input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
    input_transpose = tf.transpose( input_reshape, [2,0,1] )

    with tf.variable_scope(name):
        W = tf.get_variable(
                "W",
                shape=[n_feat_map,width*height, width*height], # (512,49,49)
                initializer=tf.random_normal_initializer(0., 0.005))
        output = tf.batch_matmul(input_transpose, W)
        tf.summary.histogram(name + '_W', W)

    output_transpose = tf.transpose(output, [1,2,0])
    output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )


    return output_reshape

def leaky_relu( bottom, leak=0.1):
    return tf.maximum(leak*bottom, bottom)
