import tensorflow as tf
import numpy as np
from tensorflow_functions import log10
import constants as c



def PSNR_error_np(compressed_image, orig_image):
    shape = compressed_image.shape
    num_pixels = shape[1]*shape[2]*shape[3]
    square_difference = np.square(compressed_image - orig_image)
    mse = (1/num_pixels)*np.sum(square_difference, axis=(1,2,3))
    PIXEL_MAX = 255.0
    psnr_batch = 20*np.log10(PIXEL_MAX/np.sqrt(mse))

    return np.mean(psnr_batch)




def PSNR_error_tf(compressed_image, orig_image):
    shape = tf.shape(compressed_image)
    num_pixels = tf.to_float(shape[1]*shape[2]*shape[3])
    square_difference = tf.square(compressed_image - orig_image)

    errors_batch = 10*log10(1/ ((1 / num_pixels)*tf.reduce_sum(square_difference, [1,2,3])))
    return tf.reduce_mean(errors_batch)
