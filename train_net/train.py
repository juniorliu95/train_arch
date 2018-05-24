# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: juniorliu95


"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
#import argparse
#import sys
#sys.path.append('../net/')
from load_image.load_image import read_and_decode
import config
import froc
#from PIL import Image
#from datetime import datetime
#import math
#import time
#import cv2
from mask import model_copy
#from keras.utils import np_utils
import time
import os
from heatmap import tb_map
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#from load_image.load_image import load_database_path, get_next_batch_from_path, shuffle_train_data
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
        net, end_points = inception_v4(X, dropout_keep_prob=dropout_keep_prob,is_training=is_train,mask=mask)  
        # inputs, num_classes=None, is_training=True,dropout_keep_prob=0.8,reuse=None,scope='InceptionV4',create_aux_logits=True,mask=None
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Logits_out'):
            # 8 x 8 x 1536
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out0')
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out1')
            net = slim.max_pool2d(net, [2, 2], scope='Logits_MaxPool_1a')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out2')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out3')
            net = slim.max_pool2d(net, [2, 2], scope='Logits_MaxPool_2a')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out4')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out5')
            net1 = net
            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                  scope='AvgPool_1a')
            # 1 x 1 x 1536
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out')
            net = slim.flatten(net, scope='PreLogitsFlatten_out')
            # 1536
#            net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
            
            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
            out = g_heatmap(net1)
            end_points['heatmap'] = out
    return net, end_points

def arch_resnet_v2(X, num_classes, dropout_keep_prob=0.8, is_train=False,name=50, mask=None):
    arg_scope = resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        if name == 50:
            net, end_points = resnet_v2_50(X, num_classes=num_classes,is_training=is_train, mask=mask)
#inputs,num_classes=None,is_training=True,global_pool=True,output_stride=None,spatial_squeeze=True,reuse=None,scope='resnet_v2_50',mask=None
        elif name == 101:
            net, end_points = resnet_v2_101(X, is_training=is_train, mask=mask)
        elif name == 152:
            net, end_points = resnet_v2_152(X, is_training=is_train, mask=mask)
        elif name == 200:
            net, end_points = resnet_v2_200(X, is_training=is_train, mask=mask)
        else:
            net = []
            end_points = []
            assert("not exist layer num:", name)

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Logits_out'):
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out0')
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out1')
            net = slim.max_pool2d(net, [2, 2], scope='Logits_MaxPool_1a')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out2')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out3')
            net = slim.max_pool2d(net, [2, 2], scope='Logits_MaxPool_2a')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out4')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out5')
            
            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                  scope='AvgPool_1a')
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out0')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out0')
            net = slim.conv2d(net, 256, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out1')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out1')
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out2')
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            # end_points['heatmap'] = net
    return net

def arch_vgg(X, num_classes, dropout_keep_prob=0.8, is_train=False, name=16, mask=None):
    arg_scope = vgg_arg_scope()
    with slim.arg_scope(arg_scope):
        if name == 16:
            net, end_points = vgg_16(X, is_training=is_train, dropout_keep_prob=dropout_keep_prob, mask=mask)
        elif name == 19:
            net, end_points = vgg_19(X, is_training=is_train, dropout_keep_prob=dropout_keep_prob, mask=mask)
        else:
            net = []
            end_points = []
            assert ("not exist layer num:", name)

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Logits_out'):
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out0')
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out1')
            net = slim.max_pool2d(net, [2, 2], scope='Logits_MaxPool_1a')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out2')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out3')
            net = slim.max_pool2d(net, [2, 2], scope='Logits_MaxPool_2a')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out4')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out5')
            net1 = net
            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                  scope='AvgPool_1a')
            net = slim.fully_connected(net, num_classes, activation_fn=None,normalizer_fn=None,scope='Logits')
            out = g_heatmap(net1)
            end_points['heatmap'] = out
            net = tf.squeeze(net,[1,2], name='fc8/squeezed')
    return net, end_points

def arch_inception_resnet_v2(X, num_classes, dropout_keep_prob=0.8, is_train=False, mask=None):
    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        net, end_points = inception_resnet_v2(X, num_classes=num_classes, is_training=is_train, dropout_keep_prob=dropout_keep_prob, mask=mask)
        with tf.variable_scope('Logits_out'):
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out0')
            net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out1')
            net = slim.max_pool2d(net, [2, 2], scope='Logits_MaxPool_1a')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out2')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out3')
            net = slim.max_pool2d(net, [2, 2], scope='Logits_MaxPool_2a')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out4')
            net = slim.conv2d(net, 512, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_conv_out5')
            
            net1 = net
            
            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                  scope='AvgPool_1a_8x8')
            net = slim.flatten(net)
    
            net = slim.dropout(net, dropout_keep_prob, is_training=is_train,
                               scope='Dropout')
            
            end_points['PreLogitsFlatten'] = net
            net = slim.fully_connected(net, num_classes, activation_fn=None,
                                          scope='Logits')
            end_points['Logits'] = net
#            end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
            out = g_heatmap(net1)
            end_points['heatmap'] = out
    return net, end_points


def g_heatmap(net):
    weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Logits_out/Logits/weights:0')
#    print weight
    w1,w2 = tf.split(weight,2,axis=-1)
    w2 = tf.squeeze(w2,axis=-1)
    out = net * w2
    return out

def g_parameter(checkpoint_exclude_scopes,retrain=True):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    print (exclusions)
    variables_to_restore = []
    variables_to_train = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                variables_to_train.append(var)
                if retrain:
                    variables_to_restore.append(var)
                print (var.op.name)
                print ("ok")
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore,variables_to_train


def train(IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,batch_size=64,keep_prob=0.8,
           arch_model="arch_inception_v4",checkpoint_exclude_scopes="Logits_out",
          checkpoint_path="../ckpt/inception_v4/inception_v4.ckpt", retrain=True):
    is_training = tf.placeholder_with_default(False, shape=(),name='is_training')
    k_prob = tf.placeholder('float') # dropout

    dataset_train = read_and_decode('../dataset/train.tfrecord', epoch,batch_size)
    nBatchs = config.nDatasTrain*epoch//batch_size
    iter_train = dataset_train.make_one_shot_iterator()
    handle = tf.placeholder(tf.string, shape=[])  
    iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)  
    _, img_batch, label_batch0, mask_batch, _ = iterator.get_next()
    
    dataset_val = read_and_decode('../dataset/val.tfrecord', 1,1)
    iter_val   = dataset_val.make_one_shot_iterator()
    
    if IMAGE_HEIGHT != img_batch.get_shape().as_list()[1] or IMAGE_WIDTH != img_batch.get_shape().as_list()[2]:
        img_batch = tf.image.resize_images(img_batch,[IMAGE_HEIGHT,IMAGE_WIDTH])
    if img_batch.get_shape().as_list()[-1] == 1:
        img_batch = tf.concat([img_batch, img_batch, img_batch], axis=-1)

    label_batch = tf.cast(tf.one_hot(tf.cast(label_batch0,tf.uint8), num_classes, on_value=1, axis=1),tf.float32)

    # setup models
    if arch_model == "arch_inception_v4":
        net, _ = arch_inception_v4(img_batch, num_classes, k_prob, is_training,mask=mask_batch)
        model_path = '../model/inception_v4/'

    elif arch_model == "arch_resnet_v2_50":
        net = arch_resnet_v2(img_batch, num_classes, k_prob, is_training,mask=mask_batch)
        model_path = '../model/resnet_v2_50/'
        
    elif arch_model == "arch_resnet_v2_101":
        net = arch_resnet_v2(img_batch, num_classes, k_prob, is_training,name=101, mask=mask_batch)
        model_path = '../model/resnet_v2_101/'
        
    elif arch_model == "arch_resnet_v2_152":
        net = arch_resnet_v2(img_batch, num_classes, k_prob, is_training,name=152,mask=mask_batch)
        model_path = '../model/resnet_v2_152/'
#        
#    elif arch_model == "arch_resnet_v2_200":
#        net = arch_resnet_v2(X, num_classes, k_prob, is_training,name=200, mask=MASK)
#        model_path = '../model/resnet_v2_200'
#
#    elif arch_model == "vgg_16":
#        net = arch_vgg(X, num_classes, k_prob, is_training, name=16, mask=MASK)
#        model_path = '../model/vgg_16'
        
    elif arch_model == "vgg_19":
        net, _ = arch_vgg(img_batch, num_classes, k_prob, is_training,name=19,mask=mask_batch)
        model_path = '../model/vgg_19/'
        
    elif arch_model == "inception_resnet_v2":
        net,_ = arch_inception_resnet_v2(img_batch, num_classes, k_prob, is_training,mask=mask_batch)
        model_path = '../model/inception_resnet_v2/'
        
    else:
        net = []
        model_path = '../model/'+arch_model
        assert(net == [], 'model not expected:'+ arch_model)

    variables_to_restore,variables_to_train = g_parameter(checkpoint_exclude_scopes, retrain)
    # loss function
    logits = tf.argmax(tf.nn.softmax(net, axis=-1), 1)
    loss = tf.reduce_mean(2 * label_batch0*tf.log(logits)+(1-label_batch0)*tf.log(1-logits))  # focal loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_batch, logits = net))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = net))

    var_list = variables_to_train
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=var_list)
    predict = tf.reshape(net, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(label_batch, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #------------------------------------------------------------------------------------#
    

    def summary_op(datapart='train'):  
        tf.summary.scalar(datapart + '-loss', loss)  
        tf.summary.scalar(datapart + '-eval', accuracy)  
        return tf.summary.merge_all() 
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    configgpu = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
    configgpu.gpu_options.allow_growth = True 
    sess = tf.Session(config=configgpu)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    saver2 = tf.train.Saver(tf.global_variables())

    net_vars = variables_to_restore
    saver_net = tf.train.Saver(net_vars)
    # checkpoint_path = 'pretrain/inception_v4.ckpt'
    # saver2.restore(sess, "model/fine-tune-1120")
    num_of_iteration = 0  # if retrained, add to i as the new name of ckpt.
    if checkpoint_path.find('ckpt') == -1:
        f = open(checkpoint_path + 'checkpoint')
        # if failed, try --retrain=False
        line = f.readline()
        model = line.split('"')
        checkpoint_path += model[1]
        num_of_iteration += int(checkpoint_path[checkpoint_path.rfind('-')+1:])
        print 'checkpoint path:'
        print checkpoint_path, 'num of iteration:', num_of_iteration
    saver_net.restore(sess, checkpoint_path)
    
    handle_train, handle_val = sess.run([x.string_handle() for x in [iter_train, iter_val]])  
    
    
    summary_op_train = summary_op() 
    summary_wrt = tf.summary.FileWriter(config.logdir,sess.graph)
    iterate = 0
    try:
        for i in range(0, nBatchs):
            _, cur_loss, cur_train_eval, summary = sess.run([train_op, loss, accuracy,summary_op_train],
                                                            feed_dict={handle: handle_train, is_training:True, k_prob: keep_prob} )  
            # log to stdout and eval validation set  
            if i % 100 == 0 or i == nBatchs-1:  
                saver2.save(sess, model_path+'model.ckpt', global_step=i+num_of_iteration) # save variables
                summary_wrt.add_summary(summary, global_step=i)
                start_time = time.time()
                cur_val_loss, cur_val_eval = sess.run([loss, accuracy],  
                    feed_dict={handle: handle_val, is_training:False, k_prob: 1.0}) 
                end_time = time.time()
                val_time = end_time-start_time
                summary_wrt.add_summary(summary, global_step=i)  
                print 'step %5d: time %.5f,loss %.5f, acc %.5f --- loss_val %0.5f, acc_val %.5f'%(i,   
                    val_time, cur_loss, cur_train_eval, cur_val_loss, cur_val_eval)  
                # sess.run(init_train)
            iterate = i
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        saver2.save(sess, model_path+'model.ckpt', global_step=iterate + num_of_iteration, write_meta_graph=False)
    sess.close()

def pre_test(IMAGE_HEIGHT, IMAGE_WIDTH, num_classes, batch_size=64,
          arch_model="arch_inception_v4", checkpoint_exclude_scopes="Logits_out",
          checkpoint_path="../model/inception_v4/inception_v4.ckpt",record=True):
    is_training = tf.placeholder_with_default(False, shape=(),name='is_training')
    k_prob = tf.placeholder('float') # dropout

    dataset_test = read_and_decode('../dataset/pre_test.tfrecord', 1,batch_size)
    nBatchs = config.nDatasTest//batch_size
    iter_test = dataset_test.make_one_shot_iterator()
    handle = tf.placeholder(tf.string, shape=[])  
    iterator = tf.data.Iterator.from_string_handle(handle, dataset_test.output_types, dataset_test.output_shapes)  
    img0_batch, img_batch, label_batch, mask_batch, name_batch = iterator.get_next()  
    # get a  new batch for each call
    
    if IMAGE_HEIGHT != img_batch.get_shape().as_list()[1] or IMAGE_WIDTH != img_batch.get_shape().as_list()[2]:
        img_batch = tf.image.resize_images(img_batch,[IMAGE_HEIGHT,IMAGE_WIDTH])
        
    if img_batch.get_shape().as_list()[-1] == 1:
        img_batch = tf.concat([img_batch, img_batch, img_batch], axis=-1)
        img0_batch = tf.concat([img0_batch, img0_batch, img0_batch], axis=-1)
        img0_batch = tf.squeeze(img0_batch,axis=0)

    label_batch = tf.cast(tf.one_hot(tf.cast(label_batch,tf.uint8), num_classes, on_value=1, axis=1),tf.float32)
    # setup models
    if arch_model == "arch_inception_v4":
        net, end_points = arch_inception_v4(img_batch, num_classes, k_prob, is_training,mask=mask_batch)

    elif arch_model == "arch_resnet_v2_50":
        net = arch_resnet_v2(img_batch, num_classes, k_prob, is_training,mask=mask_batch)
        
    elif arch_model == "arch_resnet_v2_101":
        net = arch_resnet_v2(img_batch, num_classes, k_prob, is_training,name=101,mask=mask_batch)
        
    elif arch_model == "arch_resnet_v2_152":
        net = arch_resnet_v2(img_batch, num_classes, k_prob, is_training,name=152,mask=mask_batch)
#        
#    elif arch_model == "arch_resnet_v2_200":
#        net = arch_resnet_v2(X, num_classes, k_prob, is_training,name=200, mask=MASK)
#
#    elif arch_model == "vgg_16":
#        net = arch_vgg(X, num_classes, k_prob, is_training, name=16, mask=MASK)
        
    elif arch_model == "vgg_19":
        net,end_points = arch_vgg(img_batch, num_classes, k_prob, is_training,name=19,mask=mask_batch)
        
    elif arch_model == "inception_resnet_v2":
        net,end_points = arch_inception_resnet_v2(img_batch, num_classes, k_prob, is_training,mask=mask_batch)
        
    else:
        net = []
        assert(net == [], 'model not expected:'+ arch_model)
        
        
    
    variables_to_restore, _ = g_parameter(checkpoint_exclude_scopes,True)
    
    predict = tf.reshape(net, [-1, num_classes])
    predict_s = tf.nn.softmax(predict)
    max_idx_p = tf.argmax(predict_s, 1)
    max_idx_l = tf.argmax(label_batch, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #------------------------------------------------------------------------------------#
    
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    configgpu = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
    configgpu.gpu_options.allow_growth = True 
    sess = tf.Session(config=configgpu)
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    net_vars = variables_to_restore
    saver_net = tf.train.Saver(net_vars)
    # checkpoint_path = 'pretrain/inception_v4.ckpt'
    # saver2.restore(sess, "model/fine-tune-1120")
    if checkpoint_path.find('ckpt') == -1:  # not the original ckpt path
        try:
            f = open(checkpoint_path + 'checkpoint')
        except:
            assert "no such file."
        line = f.readline()
        model = line.split('"')
        checkpoint_path += model[1]
        print 'checkpoint path:'
        print checkpoint_path
    saver_net.restore(sess, checkpoint_path)
    
    handle_test = sess.run(iter_test.string_handle())  
    
    
    threshold = np.divide(range(0,101), 100.)

    points = []
    gts = []
    acc = 0.
    try:
        for i in range(0, nBatchs):
            start_time = time.time()
            if arch_model.find("arch_resnet_v2_") == -1:
                img0_out, map_out,output,label_out, name_out = sess.run([img0_batch,end_points['heatmap'], predict_s,label_batch,name_batch],feed_dict={handle: handle_test, k_prob: 1.0} )
            else:
                output,label_out, name_out = sess.run([predict_s,label_batch,name_batch],feed_dict={handle: handle_test, k_prob: 1.0} )
                
            cur_test_eval, label_p = sess.run([accuracy,max_idx_l], feed_dict={predict_s: output, label_batch:label_out, is_training:False, k_prob: 1.0} )   # careful
            end_time = time.time()
            test_time = end_time - start_time
            print name_out[0], 'step %3d: acc %.5f, time:%.5f'%(i, cur_test_eval,test_time)
            if i == 0:
                acc = cur_test_eval
            else:
                acc = acc * i / (i+1.) + cur_test_eval/(i+1.)
            #TODO: heat map
            if arch_model.find("arch_resnet_v2_") == -1:  #no heatmap for ResNet
                tb_map.main(map_out, img0_out,name_out[0])
#            tb_map.main(map_out,img0_out,name_out)
#            img_out,label_out,mask_out = sess.run([img_batch,label_batch,mask_batch], feed_dict={handle:handle_test})
#            img_out = np.reshape(img_out,[400,400,1])
#            a = np.uint8(img_out)
#            cv2.imshow('',a)
#            cv2.waitKey()
#            cv2.destroyAllWindows()
#            print output[0][1]
#            print label_out[0][1]
            
            points.append(output[0][1])
            gts.append(label_p[0])
        
    except tf.errors.OutOfRangeError:
        print('Done testing -- epoch limit reached')
    finally:
        # P-R curve
        precision = []
        recall = []
        
        for j in range(len(threshold)):
            tp_temp = 0.
            tn_temp = 0.
            fn_temp = 0.
            fp_temp = 0.
            for k in range(len(points)):
                if points[k] > threshold[j]:
                    if gts[k] == 1:
                        tp_temp += 1
                    else: fp_temp += 1
                else:
                    if gts[k] == 0:
                        tn_temp += 1
                    else: fn_temp += 1
            pre = tp_temp/(tp_temp + fp_temp + 1e-6)
            rec = tp_temp/(tp_temp + fn_temp + 1e-6)
            if pre>0 and rec>0:
                precision.append(pre)
                recall.append(rec)
#            print tp_temp,tn_temp,fp_temp,fn_temp
            print 'threshold:', threshold[j], 'precision:',pre,'recall:', rec

        froc.plotFROC(recall,precision, np.divide(range(0,101), 100.), 'P-R.pdf', False, 'recall', 'precision')
        document = open('../results/' + arch_model+'_pr.txt','w+')
        document.write(str(precision))
        document.write('\n')
        document.write(str(recall))
        document.close()
        # fROC curve
        sensitivity = []
        fp_perframe = []
        for j in range(len(threshold)):
            tp_temp = 0.
            tn_temp = 0.
            fn_temp = 0.
            fp_temp = 0.
            for k in range(len(points)):
                if points[k] > threshold[j]:
                    if gts[k] == 1:
                        tp_temp += 1
                    else: fp_temp += 1
                else:
                    if gts[k] == 0:
                        tn_temp += 1
                    else: fn_temp += 1
            fp = fp_temp/(tp_temp + fp_temp + tn_temp + fn_temp + 1e-6)
            sen = tp_temp/(tp_temp + fn_temp + 1e-6)
            if sen > 0 :
                sensitivity.append(sen)
                fp_perframe.append(fp)
            print 'threshold:', threshold[j], 'sensitivity:', sen, 'fp per frame:', fp
#            print tp_temp,tn_temp,fp_temp,fn_temp
        froc.plotFROC(fp_perframe,sensitivity, np.divide(range(0,101), 100.), 'fROC.pdf', False)
        document = open('../results/' + arch_model+'_froc.txt','w+')
        document.write(str(sensitivity))
        document.write('\n')
        document.write(str(fp_perframe))
        document.close()
        # ROC curve
        sensitivity = []
        specificity = []
        
        for j in range(len(threshold)):
            tp_temp = 0.
            tn_temp = 0.
            fn_temp = 0.
            fp_temp = 0.
            for k in range(len(points)):
                if points[k] > threshold[j]:
                    if gts[k] == 1:
                        tp_temp += 1
                    else:
                        fp_temp += 1
                else:
                    if gts[k] == 0:
                        tn_temp += 1
                    else:
                        fn_temp += 1
            spec = fp_temp / (tn_temp + fp_temp + 1e-6)
            sen = tp_temp / (tp_temp + fn_temp + 1e-6)
            if sen > 0:
                sensitivity.append(sen)
                specificity.append(spec)
            print 'threshold:', threshold[j], 'sensitivity:', sen, 'specificity:', spec
            
            #            print tp_temp,tn_temp,fp_temp,fn_temp
        froc.plotFROC(specificity, sensitivity, np.divide(range(0, 101), 100.), 'ROC.pdf', False, 'specificity', 'sensitivity')
        document = open('../results/' + arch_model+'_roc.txt','w+')
        document.write(str(sensitivity))
        document.write('\n')
        document.write(str(specificity))
        document.close()

        print 'acc:', acc
    sess.close()


def test(IMAGE_HEIGHT, IMAGE_WIDTH, num_classes, batch_size=64,
          arch_model="arch_inception_v4", checkpoint_exclude_scopes="Logits_out",
          checkpoint_path="../model/inception_v4/inception_v4.ckpt",retrain=True):
    is_training = tf.placeholder_with_default(False, shape=(),name='is_training')
    k_prob = tf.placeholder('float') # dropout

    dataset_test = read_and_decode('../dataset/test.tfrecord', 1,batch_size,has_mask=False)
    nBatchs = config.nDatasTrain//batch_size
    iter_test = dataset_test.make_one_shot_iterator()
    handle = tf.placeholder(tf.string, shape=[])  
    iterator = tf.data.Iterator.from_string_handle(handle, dataset_test.output_types, dataset_test.output_shapes)  
    img_batch, label_batch = iterator.get_next()
    
    if IMAGE_HEIGHT != img_batch.get_shape().as_list()[1] or IMAGE_WIDTH != img_batch.get_shape().as_list()[2]:
        img_batch = tf.image.resize_images(img_batch,[IMAGE_HEIGHT,IMAGE_WIDTH])
    
    # TODO: get mask interface
    filename = "../mask_ckpt/"  # 修改
    chkpt_path = filename + "checkpoints/2018-04-19-1499"
    images_input = tf.split(img_batch, 3, axis=-1)
    images_input_test = images_input[0]
    # images_input_test = X[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 0]
    images_input_test = tf.image.resize_images(images_input_test, [400, 400])
    mask_batch = model_copy.predict(images_input_test, chkpt_path, batch_size)
    k_prob = tf.placeholder(tf.float32)  # dropout
    
    is_training = False
#    img_batch = tf.image.resize_images(img_batch,[224,224])
    if img_batch.get_shape().as_list()[-1] == 1:
        img_batch = tf.concat([img_batch, img_batch, img_batch], axis=-1)

    label_batch = tf.cast(tf.one_hot(tf.cast(label_batch,tf.uint8), num_classes, on_value=1, axis=1),tf.float32)
    # setup models
    if arch_model == "arch_inception_v4":
        net,_ = arch_inception_v4(img_batch, num_classes, k_prob, is_training,mask=mask_batch)

    elif arch_model == "arch_resnet_v2_50":
        net = arch_resnet_v2(img_batch, num_classes, k_prob, is_training,mask=mask_batch)
        
    elif arch_model == "arch_resnet_v2_101":
        net = arch_resnet_v2(img_batch, num_classes, k_prob, is_training,mask=mask_batch)
        
    elif arch_model == "arch_resnet_v2_152":
        net = arch_resnet_v2(img_batch, num_classes, k_prob, is_training,mask=mask_batch)
#        
#    elif arch_model == "arch_resnet_v2_200":
#        net = arch_resnet_v2(X, num_classes, k_prob, is_training,name=200, mask=MASK)
#
#    elif arch_model == "vgg_16":
#        net = arch_vgg(X, num_classes, k_prob, is_training, name=16, mask=MASK)
        
    elif arch_model == "vgg_19":
        net,_ = arch_vgg(img_batch, num_classes, k_prob, is_training,name=19,mask=mask_batch)
        
    elif arch_model == "inception_resnet_v2":
        net,_ = arch_inception_resnet_v2(img_batch, num_classes, k_prob, is_training,mask=mask_batch)
        
    else:
        net = None
        assert(net == None, 'model not expected:'+ arch_model)
        
        
    
    variables_to_restore, _ = g_parameter(checkpoint_exclude_scopes,retrain)
    
    predict = tf.reshape(net, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(label_batch, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #------------------------------------------------------------------------------------#
    
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    configgpu = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
    configgpu.gpu_options.allow_growth = True 
    sess = tf.Session(config=configgpu)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    net_vars = variables_to_restore
    saver_net = tf.train.Saver(net_vars)
    # checkpoint_path = 'pretrain/inception_v4.ckpt'
    # saver2.restore(sess, "model/fine-tune-1120")
    if checkpoint_path.find('ckpt') == -1:
        try:
            f = open(checkpoint_path + 'checkpoint')
        except:
            assert "no such file."
        line = f.readline()
        model = line.split('"')
        checkpoint_path += model[1]
        print 'checkpoint path:'
        print checkpoint_path
    saver_net.restore(sess, checkpoint_path)
    
    handle_test = sess.run(iter_test.string_handle())  

    
    try:
        for i in range(0, nBatchs):    
            cur_test_eval = sess.run(accuracy,feed_dict={handle: handle_test, is_training:False, k_prob: 1.0} )   
            print 'step %5d: acc %.5f'%(i, cur_test_eval)
    except tf.errors.OutOfRangeError:
        print('Done testing -- epoch limit reached')

    sess.close()


if __name__ == '__main__':

    IMAGE_HEIGHT = 1024
    IMAGE_WIDTH = 1024
    num_classes = 2
    # epoch
    epoch = 100
    batch_size = 1
    # 模型的学习率
    learning_rate = 0.00001
    keep_prob = 0.8

    
    ##----------------------------------------------------------------------------##
    arch_model="arch_inception_v4"
    checkpoint_exclude_scopes = "Logits_out"
    checkpoint_path="../ckpt/inception_v4.ckpt"
    print ("-----------------------------train.py start--------------------------")
    train(IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,batch_size,keep_prob,
          arch_model,checkpoint_exclude_scopes, checkpoint_path)
