'''
convert images and labels to tfrecords
labels are of shape [20,20,3]
images are of [160,160]
'''

# -*- coding = utf-8 -*-
import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
from PIL import ImageStat
from PIL import ImageMath
import numpy as np

# make tfrecord file
# cwd = '/home/higo/DataSet/1030/'
# cwd = 'E:/fingerprint/1030/'
# cwd = 'E:/fingerprint/1216/'
# cwd = 'E:/fingerprint/1222/'
cwd = './'
classes = ('train/', 'train_normal/', 'test/', 'test_normal/', 'mask/') #人为 设定 2 类

writer = tf.python_io.TFRecordWriter(cwd+"train.tfrecord") #要生成的文件
for i,img_name in enumerate(os.listdir(cwd+classes[0])):  # 生成训练集
    img_path = cwd+classes[0]+img_name #每一个图片的地址
    img = Image.open(img_path)
    img = img.resize((1024, 1024, 3))
    # ImgMean = ImageStat.Stat(img).mean
    # ImgStd = ImageStat.Stat(img).stddev
    # print(ImgMean, ImgStd)
    # img = (img-ImgMean)/ImgStd
    # print(img)
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()#将图片转化为二进制格式

    dirc = 1
    dirc_raw = dirc.tobytes()

    mask_path = cwd + classes[4] + img_name  # 每一个图片的地址
    mask = Image.open(mask_path)
    mask = mask.resize((400, 400))
    mask_raw = mask.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dirc_raw])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw]))
    })) #example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  #序列化为字符串
    print("this is num %d" % i)
for i, img_name in enumerate(os.listdir(cwd + classes[1])):  # 生成训练集
    img_path = cwd + classes[1] + img_name  # 每一个图片的地址
    img = Image.open(img_path)
    img = img.resize((1024, 1024, 3))
    # ImgMean = ImageStat.Stat(img).mean
    # ImgStd = ImageStat.Stat(img).stddev
    # print(ImgMean, ImgStd)
    # img = (img-ImgMean)/ImgStd
    # print(img)
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()  # 将图片转化为二进制格式

    dirc = 0
    dirc_raw = dirc.tobytes()

    mask_path = cwd + classes[4] + img_name  # 每一个图片的地址
    mask = Image.open(mask_path)
    mask = mask.resize((400, 400))
    mask_raw = mask.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dirc_raw])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw]))
    }))  # example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  # 序列化为字符串
    print("this is num %d" % i)
writer.close()

writer1 = tf.python_io.TFRecordWriter(cwd+"test.tfrecord") #要生成的文件
for i,img_name in enumerate(os.listdir(cwd + classes[2])):  # 生成测试集
    class_path = cwd + classes[2]
    img_path = class_path+img_name #每一个图片的地址
    img = Image.open(img_path)
    img = img.resize((1024, 1024, 3))
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()#将图片转化为二进制格式

    dirc = 1
    dirc_raw = dirc.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dirc_raw])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    }))  # example对象对label和image数据进行封装
    writer1.write(example.SerializeToString())  #序列化为字符串
    print("this is num %d" % i)

for i,img_name in enumerate(os.listdir(cwd + classes[3])):  # 生成测试集
    class_path = cwd + classes[3]
    img_path = class_path+img_name #每一个图片的地址
    img = Image.open(img_path)
    img = img.resize((1024, 1024, 3))
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()#将图片转化为二进制格式

    dirc = 0
    dirc_raw = dirc.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dirc_raw])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    }))  # example对象对label和image数据进行封装
    writer1.write(example.SerializeToString())  #序列化为字符串
    print("this is num %d" % i)
writer1.close()