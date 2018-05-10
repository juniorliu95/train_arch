'''
Define layer functions
'''

from __future__ import print_function, division, absolute_import
import tensorflow as tf


def weight_variable(shape, stddev=0.1):
    initial_w = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial_w, dtype=tf.float32)

def bias_variable(shape):
    initial_b = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_b, dtype=tf.float32)

def conv2d(x, w, stride=1, padding = "VALID"):
    conv_2d = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
    return conv_2d

def deconv2d(x, w, stride=2):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='VALID')

def pool_avg(x, n=2):
    return tf.nn.avg_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def pool_max(x, n=2):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    #crop
    begin = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, begin=begin, size = size)
    return tf.concat([x1_crop, x2], 3)

#bi-classification
def pixel_wise_softmax2(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map, tf.reverse(exponential_map, [3]))
    return tf.clip_by_value(exponential_map/evidence, 1e-5, 1)

#multi-classification
def pixel_wise_softmax_mlt(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, axis=3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, multiples=tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.clip_by_value(exponential_map/tensor_sum_exp, 1e-8, 1)

def cross_entropy(y_, output_map):
    return 0 - tf.reduce_mean(tf.cast(y_, tf.float32) * tf.log(tf.cast(output_map, tf.float32)))


if __name__ =='__main__':
    seed = 3
    tf.set_random_seed(seed)
    with tf.Session() as sess:
        w1 = weight_variable(shape=[1,1,1,2])
        init = tf.global_variables_initializer()
        sess.run(init)
        dd = pixel_wise_softmax2(w1)
        print(sess.run(w1))
        print(sess.run(dd))
        print(sess.run(dd).shape)
        # print(sess.run(w2))
        # cw = crop_and_concat(w1, w2)
        # print(sess.run(cw))
        