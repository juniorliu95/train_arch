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
from PIL import Image
from datetime import datetime
import math
import time
from load_image.load_image import load_database_path, get_next_batch_from_path
from train_net.train import train
import os
from keras.utils import np_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import config
# TODO:rewrite
tf.app.flags.DEFINE_string('mode', 'train', "train or test")
tf.app.flags.DEFINE_integer('network', 'vgg_19', "choose basic network")
tf.app.flags.DEFINE_boolean('bool_name', False, "descript3")

FLAGS = tf.app.flags.FLAGS

def main(_):
    IMAGE_HEIGHT = config.IMAGE_HEIGHT
    IMAGE_WIDTH = config.IMAGE_WIDTH
    num_classes = config.num_classes
    # epoch
    epoch = config.epoch
    batch_size = config.batch_size
    # 模型的学习率
    learning_rate = config.learning_rate
    keep_prob = config.keep_prob
    ##----------------------------------------------------------------------------##

    # 选择需要的模型
    # arch_model="arch_inception_v4";  arch_model="arch_resnet_v2_50"; arch_model="vgg_16"
    arch_model=config.arch_model
    # 设置要更新的参数和加载的参数，目前是非此即彼，可以自己修改哦
    checkpoint_exclude_scopes = config.checkpoint_exclude_scopes
    # 迁移学习模型参数
    checkpoint_path=config.checkpoint_path
    

    print ("-----------------------------train.py start--------------------------")
    train(IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,batch_size,keep_prob,
          arch_model, checkpoint_exclude_scopes, checkpoint_path)

if __name__ == '__main__':
    tf.app.run()
