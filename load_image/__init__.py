import tensorflow as tf
from load_image import read_and_decode
if __name__ == '__main__':
    with tf.Session() as sess:
        dataset_val = read_and_decode('../../dataset/pre_test.tfrecord', 1, 1)
        iter_val = dataset_val.make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, dataset_val.output_types, dataset_val.output_shapes)
        img_batch, label_batch, mask_batch, name = iterator.get_next()

        init = tf.global_variables_initializer()
        sess.run(init)

        handle_val = sess.run(iter_val.string_handle())
        print sess.run(name, feed_dict={handle: handle_val})