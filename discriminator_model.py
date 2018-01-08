import tensorflow as tf
import numpy as np

import constants as c
import tensorflow_functions as tfutils

class DiscriminatorModel:
    def __init__(self):

        self.bn1 = tfutils.batch_norm(name='d_bn1')
        self.bn2 = tfutils.batch_norm(name='d_bn2')
        self.bn3 = tfutils.batch_norm(name='d_bn3')
        self.bn4 = tfutils.batch_norm(name='d_bn4')
        self.bn5 = tfutils.batch_norm(name='d_bn5')

    def __call__(self, image, is_train, reuse=None):
        batch_size = tf.shape(image)[0]
        with tf.variable_scope('DIS', reuse=reuse):
            conv1 = tfutils.new_conv_layer(image, [4,4,3,64], stride=2 , name="conv1")
            bn1 = tfutils.leaky_relu(self.bn1(conv1, is_train))

            conv2 = tfutils.new_conv_layer(bn1, [4,4,64,128], stride=2 , name="conv2")
            bn2 = tfutils.leaky_relu(self.bn2(conv2, is_train))

            conv3 = tfutils.new_conv_layer(bn2, [4,4,128,256], stride=2 , name="conv3")
            bn3 = tfutils.leaky_relu(self.bn3(conv3, is_train))

            conv4 = tfutils.new_conv_layer(bn3, [4,4,256,512], stride=2 , name="conv4")
            bn4 = tfutils.leaky_relu(self.bn4(conv4, is_train))

            conv5 = tfutils.new_conv_layer(bn4, [4,4,512,1024], stride=2 , name="conv5")
            bn5 = tfutils.leaky_relu(self.bn5(conv5, is_train))

            output = tfutils.new_fc_layer(bn5, output_size=1, name='output')

        self.prediction = output[:,0]
        return self.prediction
