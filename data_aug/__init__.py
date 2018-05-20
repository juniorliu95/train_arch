import tensorflow as tf
import numpy as np
import cv2
import data_aug
if __name__ == '__main__':
    
#    path = '/home/eric/Desktop/CHNCXR_0327_1.jpg'
    path = '/home/eric/Desktop/218_214894_13871201_1.2.392.200046.100.2.1.24545396346.120131104832.3_1.jpeg'
    image = cv2.imread(path,0)
    image = tf.Variable(image,tf.float32)
    image = tf.expand_dims(image,-1)
#    img = data_aug.random_flip(image)
    img = data_aug.random_light(image)
    img = data_aug.random_rotation(img)
    img = data_aug.random_move(img)
    print img.get_shape().as_list()
    
#    image = img-image
    img = tf.squeeze(img)
#    img = data_aug.random_rotation(image)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img = sess.run(img)
    cv2.imshow('',np.uint8(img))
    cv2.waitKey()
    cv2.destroyAllWindows()
    print np.max(img)