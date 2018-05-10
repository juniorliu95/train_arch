import tensorflow as tf
import numpy as np

a = tf.Variable([1,2,3,4], tf.float32)
b = tf.Variable([5,6,7,8], tf.float32)
c = tf.where(a > 2, a, b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))


