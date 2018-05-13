'''
convert images and labels to tfrecords
labels are of shape [20,20,3]
images are of [160,160]
'''

# -*- coding = utf-8 -*-
import os
import tensorflow as tf
from PIL import Image
from PIL import ImageStat
from PIL import ImageMath
import numpy as np

# make tfrecord file
# cwd = '/home/higo/DataSet/1030/'
# cwd = 'E:/fingerprint/1030/'
# cwd = 'E:/fingerprint/1216/'
# cwd = 'E:/fingerprint/1222/'
cwd = '../dataset/'
classes = ('train/', 'train_normal/', 'test/', 'test_normal/', 'mask/')

writer = tf.python_io.TFRecordWriter(cwd+"train.tfrecord")
for i, img_name in enumerate(os.listdir(cwd+classes[0])):  # train
    img_path = cwd+classes[0]+img_name
    img = Image.open(img_path)
    img = img.resize((1024, 1024))
    # ImgMean = ImageStat.Stat(img).mean
    # ImgStd = ImageStat.Stat(img).stddev
    # print(ImgMean, ImgStd)
    # img = (img-ImgMean)/ImgStd
    # print(img)
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()

    mask_path = cwd + classes[4] + img_name
    mask = Image.open(mask_path)
    mask = mask.resize((400, 400))
    mask_raw = mask.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw]))
    })) 
    writer.write(example.SerializeToString())
    print("this is num %d" % i)
for i, img_name in enumerate(os.listdir(cwd + classes[1])):
    img_path = cwd + classes[1] + img_name
    img = Image.open(img_path)
    img = img.resize((1024, 1024))
    # ImgMean = ImageStat.Stat(img).mean
    # ImgStd = ImageStat.Stat(img).stddev
    # print(ImgMean, ImgStd)
    # img = (img-ImgMean)/ImgStd
    # print(img)
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()


    mask_path = cwd + classes[4] + img_name
    mask = Image.open(mask_path)
    mask = mask.resize((400, 400))
    mask_raw = mask.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw]))
    }))
    writer.write(example.SerializeToString())
    print("this is num %d" % i)
writer.close()

print('train.tfrecord finished!')



writer1 = tf.python_io.TFRecordWriter(cwd+"test.tfrecord")
for i, img_name in enumerate(os.listdir(cwd + classes[2])):  # test
    class_path = cwd + classes[2]
    img_path = class_path+img_name
    img = Image.open(img_path)
    img = img.resize((1024, 1024))
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    }))
    writer1.write(example.SerializeToString())
    print("this is num %d" % i)

for i, img_name in enumerate(os.listdir(cwd + classes[3])):
    class_path = cwd + classes[3]
    img_path = class_path+img_name
    img = Image.open(img_path)
    img = img.resize((1024, 1024))
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    }))
    writer1.write(example.SerializeToString())
    print("this is num %d" % i)
writer1.close()

print('test.tfrecord finished!')


writer2 = tf.python_io.TFRecordWriter(cwd+"pre_test.tfrecord")
for i, img_name in enumerate(os.listdir(cwd+classes[2])):  # pre_test
    img_path = cwd+classes[2]+img_name
    img = Image.open(img_path)
    img = img.resize((1024, 1024))
    # ImgMean = ImageStat.Stat(img).mean
    # ImgStd = ImageStat.Stat(img).stddev
    # print(ImgMean, ImgStd)
    # img = (img-ImgMean)/ImgStd
    # print(img)
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()

    mask_path = cwd + classes[4] + img_name
    mask = Image.open(mask_path)
    mask = mask.resize((400, 400))
    mask_raw = mask.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw]))
    }))
    writer2.write(example.SerializeToString())
    print("this is num %d" % i)
for i, img_name in enumerate(os.listdir(cwd + classes[3])):
    img_path = cwd + classes[3] + img_name
    img = Image.open(img_path)
    img = img.resize((1024, 1024))
    # ImgMean = ImageStat.Stat(img).mean
    # ImgStd = ImageStat.Stat(img).stddev
    # print(ImgMean, ImgStd)
    # img = (img-ImgMean)/ImgStd
    # print(img)
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()


    mask_path = cwd + classes[4] + img_name
    mask = Image.open(mask_path)
    mask = mask.resize((400, 400))
    mask_raw = mask.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw]))
    }))
    writer2.write(example.SerializeToString())
    print("this is num %d" % i)
writer2.close()

print('pre_test.tfrecord finished!')