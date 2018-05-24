# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

#import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
#import argparse
import os
#from PIL import Image
#from datetime import datetime
#import math
#import time
#from load_image.load_image import load_database_path, get_next_batch_from_path
from train_net.train import train, pre_test, test
#import os 
#from keras.utils import np_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import config
# TODO:rewrite

tf.app.flags.DEFINE_string('mode', 'train', "train,pre_test or test.")
tf.app.flags.DEFINE_bool('retrain', True, "if the model had been trained. For new layers reading.")
tf.app.flags.DEFINE_bool('record', True, "whether to record pr and roc in txt.")
tf.app.flags.DEFINE_string('arch_model', None, "arch_model")
tf.app.flags.DEFINE_string('gpu', '0', "gpu")
FLAGS = tf.app.flags.FLAGS

def main(_):

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    IMAGE_HEIGHT = config.IMAGE_HEIGHT
    IMAGE_WIDTH = config.IMAGE_WIDTH
    num_classes = config.num_classes
    # epoch
    epoch = config.epoch
    batch_size = config.batch_size
    test_batch_size = config.test_batch_size
    # 模型的学习率
    learning_rate = config.learning_rate
    keep_prob = config.keep_prob
    ##----------------------------------------------------------------------------##

    # 选择需要的模型
    # arch_model="arch_inception_v4";  arch_model="arch_resnet_v2_50"; arch_model="vgg_16"
    arch_model = config.arch_model
    if FLAGS.arch_model:
        arch_model = FLAGS.arch_model
    print 'arch_model:',arch_model
    # 设置要更新的参数和加载的参数，目前是非此即彼，可以自己修改哦
    checkpoint_exclude_scopes = config.checkpoint_exclude_scopes
    # 迁移学习模型参数
    model_path=config.checkpoint_path
    retrain = False
    if FLAGS.retrain:
        model_path = config.model_path
        print 'start from retrained model.'
        retrain = True
    
    # different mode
    if FLAGS.mode == 'train':
        print ("-----------------------------train start--------------------------")
        train(IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,batch_size,keep_prob,
              arch_model, checkpoint_exclude_scopes, model_path, retrain)
    elif FLAGS.mode == 'pre_test':
        print ("-----------------------------pre_test start--------------------------")
        pre_test(IMAGE_HEIGHT,IMAGE_WIDTH,num_classes,test_batch_size,
              arch_model, checkpoint_exclude_scopes, model_path, FLAGS.record)
    elif FLAGS.mode == 'test':
        print ("-----------------------------test start--------------------------")
        test(IMAGE_HEIGHT,IMAGE_WIDTH,num_classes,batch_size,
              arch_model, checkpoint_exclude_scopes, model_path, retrain)
if __name__ == '__main__':
    tf.app.run()
