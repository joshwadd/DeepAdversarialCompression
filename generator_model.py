import tensorflow as tf
import numpy as np
import tensorflow_functions as tfutils


import constants as c


class GeneratorModel:
    def __init__(self):

        self.bn1 = tfutils.batch_norm(name='g_bn1')
        self.bn2 = tfutils.batch_norm(name='g_bn2')
        self.bn3 = tfutils.batch_norm(name='g_bn3')
        self.bn4 = tfutils.batch_norm(name='g_bn4')
        self.bn5 = tfutils.batch_norm(name='g_bn5')


    def __call__(self, image, is_train, reuse=None):
        batch_size = image.get_shape().as_list()[0]
        input_height =  image.get_shape().as_list()[1]
        input_width = image.get_shape().as_list()[2]

        with tf.variable_scope('GEN', reuse=reuse):


            conv1 = tfutils.new_conv_layer(image, [4,4,3,64], stride=2, name='conv1')
            bn1 = tfutils.leaky_relu(self.bn1(conv1, is_train))

            conv2 = tfutils.new_conv_layer(bn1, [4,4,64,128], stride=2, name='conv2')
            bn2 = tfutils.leaky_relu(self.bn2(conv2, is_train))

            conv3 = tfutils.new_conv_layer(bn2, [4,4,128,c.DIM_REDUCE_SIZE], stride=2, name='conv3')

            self.pre_quantisation = conv3

            if c.QUANT_MODEL:
                self.Code = self.round_quantize(conv3)
            else:
                self.Code = tf.cond(is_train, lambda: tfutils.Quantize_model(conv3), lambda: tf.round(conv3))

            deconv2 = tfutils.new_deconv_layer( self.Code, [4,4,128,c.DIM_REDUCE_SIZE], conv2.get_shape().as_list(), stride=2 ,name="deconv2")
            debn2 = tfutils.leaky_relu(self.bn3(deconv2, is_train))

            deconv1 = tfutils.new_deconv_layer( debn2, [4,4,64,128], conv1.get_shape().as_list(), stride=2 ,name="deconv1")
            debn1 = tfutils.leaky_relu(self.bn4(deconv1, is_train))

            prediction = tfutils.new_deconv_layer( debn1, [4,4,3,64], [batch_size, input_height, input_width, 3], stride=2, name ="prediction")

        self.prediction = tf.clip_by_value(prediction, -127.5, 127.5)

        return self.prediction, self.Code, self.pre_quantisation

    def round_quantize(self, x, k=1):
        g = tf.get_default_graph()
        n = float(2**k -1)
        with g.gradient_override_map({"Round":"Identity"}):
            return tf.round(x)


