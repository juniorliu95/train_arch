
# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np  
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
#from skimage import exposure


## 完成图像的左右镜像
#def random_flip(image, random_flip=True):
#    if random_flip and np.random.choice([True, False]):
#        image = np.fliplr(image) # 左右
#    if random_flip and np.random.choice([True, False]):
#        image = np.flipud(image) # 上下
#    return image

# 改变光照
# 光照调节也可以用log, 参数调节和gamma相反；
# img = exposure.adjust_log(img, 1.3)

def random_light(image, random_exposure=True):
    if np.random.choice([True, False]) and random_exposure:
        delta = np.random.randint(-15, 15) / 100.
        image = tf.image.adjust_brightness(image,delta)
#    tf.image.random_contrast(image, 0.1,0.9, seed=None)
    return image
#    if random_exposure and np.random.choice([True, False]):
#        image = exposure.adjust_gamma(image, 1.3) # 调暗
#    if random_exposure and np.random.choice([True, False]):
#        image = exposure.adjust_gamma(image, 1.5) # 调暗
#    if random_exposure and np.random.choice([True, False]):
#        image = exposure.adjust_gamma(image, 0.9) # 调亮
#    if random_exposure and np.random.choice([True, False]):
#        image = exposure.adjust_gamma(image, 0.8) # 调亮
#    if random_exposure and np.random.choice([True, False]):
#        image = exposure.adjust_gamma(image, 0.7) # 调亮
#    if random_exposure and np.random.choice([True, False]):
#        image = exposure.adjust_gamma(image, 0.5) # 调亮

#def random_exposure(image, random_exposure=True):
#    if random_exposure and np.random.choice([True, False]):
#        e_rate = np.random.uniform(0.5,1.5)
#        image = exposure.adjust_gamma(image, e_rate)
#    return image

def random_rotation(image, mask, random_rot=True):
    if np.random.choice([True, False]) and random_rot:
        # 0-180随机产生旋转角度。
        angle = np.random.randint(-5,5) / 180. * np.pi
#        print angle
        image = tf.contrib.image.rotate(image, angle)  # this function need input of at least 3-d
        mask= tf.contrib.image.rotate(mask, angle)
    return image, mask
    
def random_rotation0(image, random_rot=True):
    if np.random.choice([True, False]) and random_rot:
        # 0-180随机产生旋转角度。
        angle = np.random.randint(-5,5) / 180. * np.pi
#        print angle
        image = tf.contrib.image.rotate(image, angle)  # this function need input of at least 3-d
    return image

def random_move(image, mask, random_move=True):
     if random_move and np.random.choice([True, False]):
        pix_x = np.random.randint(0, 150)
        pix_y = np.random.randint(0, 150)
#        print pix_x,pix_y
        img_size = image.get_shape().as_list()
        image = tf.image.resize_image_with_crop_or_pad(image,img_size[1] + 150, img_size[0]+ 150)
        image = tf.expand_dims(image, 0)
        image = tf.image.crop_to_bounding_box(image, pix_y, pix_x, img_size[1],img_size[0])
        image = tf.squeeze(image, 0)
        mask = tf.image.resize_image_with_crop_or_pad(mask,(img_size[1]+150)*400/img_size[1],(img_size[0]+150)*400/img_size[0])
        mask = tf.expand_dims(mask, 0)
        mask = tf.image.crop_to_bounding_box(mask,pix_y*400/img_size[1],pix_x*400/img_size[0],400,400)
        mask = tf.squeeze(mask, 0)
     return image,mask
 
def random_move0(image, random_move=True):
     if random_move and np.random.choice([True, False]):
        pix_x = np.random.randint(0, 150)
        pix_y = np.random.randint(0, 150)
#        print pix_x,pix_y
        img_size = image.get_shape().as_list()
        image = tf.image.resize_image_with_crop_or_pad(image,img_size[1] + 150, img_size[0]+ 150)
        image = tf.expand_dims(image, 0)
        image = tf.image.crop_to_bounding_box(image, pix_y, pix_x, img_size[1],img_size[0])
        image = tf.squeeze(image, 0)
        
     return image
def random_flip_horizonal(image, mask, random_flip=True):
    # left to right
    
    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped
    if np.random.choice([True, False]) and random_flip:
        image = _flip_image(image)
        mask = _flip_image(mask)
    return image, mask

def random_flip_horizonal0(image, random_flip=True):
    
    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped
    if np.random.choice([True, False]) and random_flip:
        image = _flip_image(image)
    return image


#def random_crop(image, crop_size=299, random_crop=True):
#    if random_crop and np.random.choice([True, False]):
#        if image.shape[1] > crop_size:
#            sz1 = image.shape[1] // 2
#            sz2 = crop_size // 2
#            diff = sz1 - sz2
#            (h, v) = (np.random.randint(0, diff + 1), np.random.randint(0, diff + 1))
#            image = image[v:(v + crop_size), h:(h + crop_size), :]
#
#    return image
