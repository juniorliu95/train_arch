# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import argparse
import os
os.path.append('../net/')
from load_image.load_image import read_and_decode
import config
from PIL import Image
from datetime import datetime
import math
import time
# import cv2
from mask import model_copy
from keras.utils import np_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from load_image.load_image import load_database_path, get_next_batch_from_path, shuffle_train_data
# inception_v4
from net.inception_v4.inception_v4 import inception_v4_arg_scope, inception_v4
# resnet_v2_50, resnet_v2_101, resnet_v2_152, resnet_v2_200
from net.resnet_v2.resnet_v2 import resnet_arg_scope, resnet_v2_50, resnet_v2_101, resnet_v2_152, resnet_v2_200
# vgg16, vgg19
from net.vgg.vgg import vgg_arg_scope, vgg_16, vgg_19
# inception_resnet_v2
from net.inception_resnet_v2.inception_resnet_v2 import inception_resnet_v2_arg_scope,inception_resnet_v2


def arch_inception_v4(X, num_classes, dropout_keep_prob=0.8, is_train=False, mask=None):
    arg_scope = inception_v4_arg_scope()
    with slim.arg_scope(arg_scope):
        net, end_points = inception_v4(X, is_training=is_train,mask=mask)
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Logits_out'):
            # 8 x 8 x 1536
            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool_1a_out')
            # 1 x 1 x 1536
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out')
            net = slim.flatten(net, scope='PreLogitsFlatten_out')
            # 1536
            net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
            net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
    return net

def arch_resnet_v2(X, num_classes, dropout_keep_prob=0.8, is_train=False,name=50, mask=None):
    arg_scope = resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        if name == 50:
            net, end_points = resnet_v2_50(X, is_training=is_train, mask=mask)
        elif name == 101:
            net, end_points = resnet_v2_101(X, is_training=is_train, mask=mask)
        elif name == 152:
            net, end_points = resnet_v2_152(X, is_training=is_train, mask=mask)
        elif name == 200:
            net, end_points = resnet_v2_200(X, is_training=is_train, mask=mask)
        else:
            net, end_points = [], []
            assert("not exist layer num:", name)

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Logits_out'):
            net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out0')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out0')
            net = slim.conv2d(net, 200, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out1')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out1')
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out2')
            net = tf.squeeze(net,[1, 2], name='SpatialSqueeze')
    return net

def arch_vgg(X, num_classes, dropout_keep_prob=0.8, is_train=False, name=16, mask=None):
    arg_scope = vgg_arg_scope()
    with slim.arg_scope(arg_scope):
        if name == 16:
            net, end_points = vgg_16(X, is_training=is_train)
        elif name == 19:
            net, end_points = vgg_19(X, is_training=is_train)
        else:
            net, end_points = [], []
            assert ("not exist layer num:", name)

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Logits_out'):
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,normalizer_fn=None,scope='fc8')
            net = tf.squeeze(net,[1,2], name='fc8/squeezed')
    return net

def arch_inception_resnet_v2(X, num_classes, dropout_keep_prob=0.8, is_train=False, mask=None):
    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        net, end_points = inception_resnet_v2(X, is_training=is_train,num_classes=num_classes)
    return net

def g_parameter(checkpoint_exclude_scopes):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    print (exclusions)
    # 需要加载的参数。
    variables_to_restore = []
    # 需要训练的参数
    variables_to_train = []
    for var in slim.get_model_variables():
    # 切记不要用下边这个，这是个天大的bug，调试了3天。
    # for var in tf.trainable_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                variables_to_train.append(var)
                print ("ok")
                print (var.op.name)
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore,variables_to_train


def train(IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,batch_size=64,keep_prob=0.8,
           arch_model="arch_inception_v4",checkpoint_exclude_scopes="Logits_out", checkpoint_path="../ckpt/inception_v4/inception_v4.ckpt"):

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    #Y = tf.placeholder(tf.float32, [None, 4])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    MASK = tf.placeholder(tf.float32, [None, 400, 400, 1])
    is_training = tf.placeholder(tf.bool, name='is_training')
    k_prob = tf.placeholder(tf.float32) # dropout

    # 定义模型
    if arch_model == "arch_inception_v4":
        net = arch_inception_v4(X, num_classes, k_prob, is_training,mask=MASK)

    elif arch_model == "arch_resnet_v2_50":
        net = arch_resnet_v2(X, num_classes, k_prob, is_training, mask=MASK)
    elif arch_model == "arch_resnet_v2_101":
        net = arch_resnet_v2(X, num_classes, k_prob, is_training, name=101, mask=MASK)
    elif arch_model == "arch_resnet_v2_152":
        net = arch_resnet_v2(X, num_classes, k_prob, is_training,name=152, mask=MASK)
    elif arch_model == "arch_resnet_v2_200":
        net = arch_resnet_v2(X, num_classes, k_prob, is_training,name=200, mask=MASK)

    elif arch_model == "vgg_16":
        net = arch_vgg(X, num_classes, k_prob, is_training, mask=MASK)
    elif arch_model == "vgg_19":
        net = arch_vgg(X, num_classes, k_prob, is_training, name=19, mask=MASK)
    elif arch_model == "inception_resnet_v2":
        net = inception_resnet_v2(X, num_classes, is_training, k_prob, mask=MASK)
    else:
        net = []
        assert('model not expected:',arch_model)
    variables_to_restore,variables_to_train = g_parameter(checkpoint_exclude_scopes)

    # loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = net))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = net))

    var_list = variables_to_train
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=var_list)
    predict = tf.reshape(net, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(Y, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #------------------------------------------------------------------------------------#
    image_flow, label_flow, mask_flow = read_and_decode('../dataset/train.tfrecord', epoch)

    img_batch, label_batch,mask_batch = tf.train.shuffle_batch \
        ([image_flow, label_flow, mask_flow], batch_size=batch_size,
         capacity=config.capacity, min_after_dequeue=config.min_after_dequeue)

    if tf.shape(img_batch)[-1] == 1:
        img_batch = tf.concat([img_batch, img_batch, img_batch], axis=-1)

    label_batch = tf.one_hot(label_batch, num_classes, on_value=1, axis=0)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())

    saver2 = tf.train.Saver(tf.global_variables())
    model_path = '../model/fine-tune'

    net_vars = variables_to_restore
    saver_net = tf.train.Saver(net_vars)
    # checkpoint_path = 'pretrain/inception_v4.ckpt'
    # saver2.restore(sess, "model/fine-tune-1120")
    saver_net.restore(sess, checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0
    try:
        while not coord.should_stop():
    # for epoch_i in range(epoch):
    #     for batch_i in range(int(train_n/batch_size)):
            images_train, labels_train, masks_train = sess.run([img_batch, label_batch,mask_batch])
            # images_train, labels_train = get_next_batch_from_path(train_data, train_label, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=True)

            loss, _ = sess.run([loss,optimizer], feed_dict={X: images_train, Y: labels_train, k_prob:keep_prob, is_training:True, MASK:masks_train})
            # print (los)

            if i % 100 == 0:
                # images_valid, labels_valid = get_next_batch_from_path(valid_data, valid_label, batch_i%(int(valid_n/batch_size)), IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=False)
                ls, acc = sess.run([loss, accuracy], feed_dict={X: images_train, Y: labels_train, k_prob:1.0, is_training:False, MASK:masks_train})
                print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(i, ls, acc))
                #if acc > 0.90:
                #    saver2.save(sess, model_path, global_step=batch_i, write_meta_graph=False)
            # elif batch_i % 20 == 0:
            #     loss_, acc_ = sess.run([loss, accuracy], feed_dict={X: images_train, Y: labels_train, k_prob:1.0, is_training:False,MASK:masks_train})
            #     print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss_, acc_))
                
        # print('Epoch===================================>: {:>2}'.format(epoch_i))
        # valid_ls = 0
        # valid_acc = 0
        # for batch_i in range(int(valid_n/batch_size)):
        #     images_valid, labels_valid = get_next_batch_from_path(valid_data, valid_label, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=False)
        #     epoch_ls, epoch_acc = sess.run([loss, accuracy], feed_dict={X: images_valid, Y: labels_valid, k_prob:1.0, is_training:False})
        #     valid_ls = valid_ls + epoch_ls
        #     valid_acc = valid_acc + epoch_acc
        # print('Epoch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(epoch_i, valid_ls/int(valid_n/batch_size), valid_acc/int(valid_n/batch_size)))
        # if valid_acc/int(valid_n/batch_size) > 0.90:
            i += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
    # When done, ask the threads to stop.
    # test save
        saver2.save(sess, model_path, global_step=i, write_meta_graph=False)
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

def pre_test(IMAGE_HEIGHT, IMAGE_WIDTH, num_classes, epoch, batch_size=64,
          arch_model="arch_inception_v4", checkpoint_exclude_scopes="Logits_out",
          checkpoint_path="pretrain/inception_v4/inception_v4.ckpt"):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    MASK = tf.placeholder(tf.float32, [None, 400, 400, 1])
    k_prob = tf.placeholder(tf.float32)  # dropout

    # 定义模型
    if arch_model == "arch_inception_v4":
        net = arch_inception_v4(X, num_classes, k_prob,mask=MASK)

    elif arch_model == "arch_resnet_v2_50":
        net = arch_resnet_v2(X, num_classes, k_prob, mask=MASK)
    elif arch_model == "arch_resnet_v2_101":
        net = arch_resnet_v2(X, num_classes, k_prob, name=101, mask=MASK)
    elif arch_model == "arch_resnet_v2_152":
        net = arch_resnet_v2(X, num_classes, k_prob, name=152, mask=MASK)
    elif arch_model == "arch_resnet_v2_200":
        net = arch_resnet_v2(X, num_classes, k_prob, name=200, mask=MASK)

    elif arch_model == "vgg_16":
        net = arch_vgg(X, num_classes, k_prob, mask=MASK)
    elif arch_model == "vgg_19":
        net = arch_vgg(X, num_classes, k_prob, name=19, mask=MASK)
    elif arch_model == "inception_resnet_v2":
        net = inception_resnet_v2(X, num_classes, k_prob, mask=MASK)
    else:
        net = []
        assert ('model not expected:', arch_model)
    variables_to_restore, variables_to_train = g_parameter(checkpoint_exclude_scopes)

    var_list = variables_to_train
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    predict = tf.reshape(net, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(Y, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # ------------------------------------------------------------------------------------#
    image_flow, label_flow, mask_flow = read_and_decode('../dataset/pre_test.tfrecord', epoch)

    img_batch, label_batch, mask_batch = tf.train.shuffle_batch \
        ([image_flow, label_flow, mask_flow], batch_size=batch_size,
         capacity=config.capacity, min_after_dequeue=config.min_after_dequeue)

    if tf.shape(img_batch)[-1] == 1:
        img_batch = tf.concat([img_batch, img_batch, img_batch], axis=-1)

    label_batch = tf.one_hot(label_batch, num_classes, on_value=1, axis=0)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())

    net_vars = variables_to_restore
    saver_net = tf.train.Saver(net_vars)
    # checkpoint_path = 'pretrain/inception_v4.ckpt'
    # saver2.restore(sess, "model/fine-tune-1120")
    saver_net.restore(sess, checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0
    try:
        while not coord.should_stop():
            # for epoch_i in range(epoch):
            #     for batch_i in range(int(train_n/batch_size)):
            images_train, labels_train, masks_train = sess.run([img_batch, label_batch, mask_batch])

            acc = sess.run(accuracy, feed_dict={X: images_train, Y: labels_train, k_prob: 1.0, MASK: MASK})
            print('Batch: {:>2}: Validation accuracy: {:>3.5f}'.format(i, acc))
            i += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

def test(IMAGE_HEIGHT, IMAGE_WIDTH, num_classes, epoch, batch_size=64,
          arch_model="arch_inception_v4", checkpoint_exclude_scopes="Logits_out",
          checkpoint_path="pretrain/inception_v4/inception_v4.ckpt"):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
# TODO: get mask interface
    filename = "../mask_ckpt/"  # 修改
    chkpt_path = filename + "checkpoints/2018-04-19-1499"
    images_input = tf.split(X, 3, axis=-1)
    images_input_test = images_input[0]
    # images_input_test = X[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 0]
    images_input_test = tf.image.resize_images(images_input_test, [400, 400])
    mask = model_copy.predict(filename, images_input_test, chkpt_path, batch_size)
    MASK = mask
    k_prob = tf.placeholder(tf.float32)  # dropout

    # 定义模型
    with tf.device('gpu:0'):
        if arch_model == "arch_inception_v4":
            net = arch_inception_v4(X, num_classes, k_prob,mask=MASK)

        elif arch_model == "arch_resnet_v2_50":
            net = arch_resnet_v2(X, num_classes, k_prob, mask=MASK)
        elif arch_model == "arch_resnet_v2_101":
            net = arch_resnet_v2(X, num_classes, k_prob, name=101, mask=MASK)
        elif arch_model == "arch_resnet_v2_152":
            net = arch_resnet_v2(X, num_classes, k_prob, name=152, mask=MASK)
        elif arch_model == "arch_resnet_v2_200":
            net = arch_resnet_v2(X, num_classes, k_prob, name=200, mask=MASK)

        elif arch_model == "vgg_16":
            net = arch_vgg(X, num_classes, k_prob, mask=MASK)
        elif arch_model == "vgg_19":
            net = arch_vgg(X, num_classes, k_prob, name=19, mask=MASK)
        elif arch_model == "inception_resnet_v2":
            net = inception_resnet_v2(X, num_classes, k_prob, mask=MASK)
        else:
            net = []
            assert ('model not expected:', arch_model)
        variables_to_restore, variables_to_train = g_parameter(checkpoint_exclude_scopes)

        var_list = variables_to_train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        predicts = tf.reshape(net, [-1, num_classes])
        max_idx_p = tf.argmax(predicts, 1)
        max_idx_l = tf.argmax(Y, 1)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # ------------------------------------------------------------------------------------#
    image_flow, label_flow= read_and_decode('dataset/test.tfrecord', epoch)

    img_batch, label_batch = tf.train.shuffle_batch \
        ([image_flow, label_flow], batch_size=batch_size,
         capacity=config.capacity, min_after_dequeue=config.min_after_dequeue)

    if tf.shape(img_batch)[-1] == 1:
        img_batch = tf.concat([img_batch, img_batch, img_batch], axis=-1)

    label_batch = tf.one_hot(label_batch, num_classes, on_value=1, axis=0)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())

    net_vars = variables_to_restore
    saver_net = tf.train.Saver(net_vars)
    # checkpoint_path = 'pretrain/inception_v4.ckpt'
    # saver2.restore(sess, "model/fine-tune-1120")
    saver_net.restore(sess, checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0
    try:
        while not coord.should_stop():
            # for epoch_i in range(epoch):
            #     for batch_i in range(int(train_n/batch_size)):
            images_train, labels_train, masks_train = sess.run([img_batch, label_batch])

            acc = sess.run(accuracy, feed_dict={X: images_train, Y: labels_train, k_prob: 1.0})
            print('Batch: {:>2}: Validation accuracy: {:>3.5f}'.format(i, acc))
            i += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

if __name__ == '__main__':

    IMAGE_HEIGHT = 1024
    IMAGE_WIDTH = 1024
    num_classes = 2
    # epoch
    epoch = 100
    batch_size = 4
    # 模型的学习率
    learning_rate = 0.00001
    keep_prob = 0.8

    
    ##----------------------------------------------------------------------------##
    arch_model="arch_inception_v4"
    checkpoint_exclude_scopes = "Logits_out"
    checkpoint_path="pretrain/inception_v4/inception_v4.ckpt"
    print ("-----------------------------train.py start--------------------------")
    train(IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,batch_size,keep_prob,
          arch_model,checkpoint_exclude_scopes, checkpoint_path)
