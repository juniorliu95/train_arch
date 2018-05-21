# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np  
import tensorflow as tf
#import os
from PIL import Image
#import cv2
from data_aug import data_aug
# 适用于二级目录 。。。/图片类别文件/图片（.png ,jpg等）

#def load_img_path(imgDir,imgFoldName, img_label):
#    imgs = os.listdir(imgDir+imgFoldName)
#    imgNum = len(imgs)
#    data = []
#    label = []
#    for i in range (imgNum):
#        img_path = imgDir+imgFoldName+"/"+imgs[i]
#        # 用来检测图片是否有效，放在这里会太费时间。
#        # img = cv2.imread(img_path)
#        # if img is not None:
#        data.append(img_path)
#        label.append(int(img_label))
#    return data,label
#
#def shuffle_train_data(train_imgs, train_labels):
#    index = [i for i in range(len(train_imgs))]
#    np.random.shuffle(index)
#    train_imgs = np.asarray(train_imgs)
#    train_labels = np.asarray(train_labels)
#    train_imgs = train_imgs[index]
#    train_labels = train_labels[index]
#    return train_imgs, train_labels
#
#def load_database_path(imgDir):
#    img_path = os.listdir(imgDir)
#    train_imgs = []
#    train_labels = []
#    for i, path in enumerate(img_path):
#        craterDir = imgDir + '/'
#        foldName = path
#        data, label = load_img_path(craterDir,foldName, i)
#        train_imgs.extend(data)
#        train_labels.extend(label)
#        print ("文件名对应的label:")
#        print (path, i)
#    #打乱数据集
#    train_imgs, train_labels = shuffle_train_data(train_imgs, train_labels)
#    return train_imgs, train_labels
#
#
#def get_next_batch_from_path(image_path, image_labels, pointer, IMAGE_HEIGHT=299, IMAGE_WIDTH=299, batch_size=64, is_train=True):
#    batch_x = np.zeros([batch_size, IMAGE_HEIGHT,IMAGE_WIDTH,3])
#    num_classes = len(image_labels[0])
#    batch_y = np.zeros([batch_size, num_classes]) 
#    for i in range(batch_size):  
#        image = cv2.imread(image_path[i+pointer*batch_size])
#        image = cv2.resize(image, (int(IMAGE_HEIGHT*1.5), int(IMAGE_WIDTH*1.5)))
#        if is_train:
#            image = random_flip(image)
#            image = random_rotation(image)
#            image = random_crop(image)
#            image = random_exposure(image)
#        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))  
#        # 选择自己预处理方式：
#        '''
#        m = image.mean()
#        s = image.std()
#        min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
#        std = max(min_s, s)
#        image = (image-m)/std'''
#        # image = (image-127.5)
#        image = image / 255.0
#        image = image - 0.5
#        image = image * 2
#        
#        batch_x[i,:,:,:] = image
#        # print labels[i+pointer*batch_size]
#        batch_y[i] = image_labels[i+pointer*batch_size]
#    return batch_x, batch_y

def read_and_decode(filename, epochs=None,batch_size=1,is_train=True, has_mask=True):
    
    def get_dataset(fname):  
        dataset = tf.data.TFRecordDataset(filename)  
        return dataset.map(parse_exmp)
    
    def parse_exmp(serial_exmp): 
        if has_mask:
            features = tf.parse_single_example(serial_exmp,
                                               features={
                                                   'label': tf.FixedLenFeature([], tf.int64),
                                                   'img_raw': tf.FixedLenFeature([], tf.string),
                                                   'mask': tf.FixedLenFeature([], tf.string)
                                               })  # 将image数据和label取出来
        else:
            features = tf.parse_single_example(serial_exmp,
                                               features={
                                                   'label': tf.FixedLenFeature([], tf.int64),
                                                   'img_raw': tf.FixedLenFeature([], tf.string),
                                               })  # 将image数据和label取出来
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img0 = tf.cast(tf.reshape(img, [1024, 1024, 1]),tf.float32)  # reshape为128*128的1通道图片
        
        mean = tf.reduce_mean(img0)
        std = tf.sqrt(tf.reduce_mean((img0-mean)**2))
        img_w = (tf.cast(img0, tf.float32) - mean) * (1./std)  # 白化
#        img_w = data_aug.random_light(img_w, is_train)
        label = features['label'] # 在流中抛出label张量
        label = tf.cast(label, tf.float32)
        if has_mask:
            mask = tf.decode_raw(features['mask'], tf.uint8)
            mask0 = tf.cast(tf.reshape(mask, [400, 400, 1]),tf.float32)  # reshape为128*128的1通道图片
            mask_w = mask0 / tf.reduce_max(mask0)
            img_w, mask_w = data_aug.random_rotation(img_w,mask_w, is_train)
            img_w, mask_w = data_aug.random_move(img_w,mask_w, is_train)
            img_w, mask_w = data_aug.random_flip_horizonal(img_w,mask_w, is_train)
            return img_w,label,mask_w
        img_w = data_aug.random_rotation0(img_w, is_train)
        img_w = data_aug.random_move0(img_w, is_train)
        img_w = data_aug.random_flip_horizonal0(img_w, is_train)
        return img_w, label
    dataset_train = get_dataset(filename)  
    dataset_train = dataset_train.repeat(epochs).shuffle(1000).batch(batch_size)
    return dataset_train


def test():
    img0, label, mask0 = read_and_decode('./train.tfrecord', epoch=None, is_train=True)

if __name__ == '__main__':
    test()

