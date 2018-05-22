import tensorflow as tf
import numpy as np
import cv2
import data_aug
if __name__ == '__main__':
    
#    path = '/home/eric/Desktop/CHNCXR_0327_1.jpg'
    path = '/home/eric/Desktop/218.jpeg'
    mask_path = '/home/eric/Desktop/218_mask.jpeg'

    image = cv2.imread(path,0)
    image = tf.Variable(image,tf.float32)
    mask = cv2.imread(mask_path,0)
    mask = tf.image.convert_image_dtype(mask,tf.float32)

    image = tf.expand_dims(image,-1)
    mask = tf.expand_dims(mask,-1)
#    img = data_aug.random_flip(image)
    #img = data_aug.random_light(image)
    img,mask = data_aug.random_rotation(image,mask)
    img,mask = data_aug.random_move(img,mask)
    #print img.get_shape().as_list()
    mask = tf.cast(mask,tf.float32)
    mask = tf.image.resize_images(mask,(1024,1024))
    img = tf.squeeze(img)
    msk = tf.squeeze(mask)
    msk = tf.cast(msk,tf.uint8)
    out = img * msk

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img = sess.run(out)
    cv2.imshow('',np.uint8(img))
    cv2.waitKey()
    cv2.destroyAllWindows()
    #print np.max(img)
