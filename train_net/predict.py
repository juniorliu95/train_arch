import tensorflow as tf
import numpy as np

a = tf.Variable([[[[1.], [2.]],[[3.], [4.]]]], tf.float32)
b = tf.ones([1, 1, 1, 3], dtype=tf.float32)
c = tf.nn.conv2d(a, b, strides=[1, 1, 1, 1], padding='SAME')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
