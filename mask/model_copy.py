#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np

import glob
import os, cv2
from mask.layers import *
import json
import datetime
from scipy import ndimage
import scipy.misc

from tensorflow.python.training.moving_averages import assign_moving_average
import skimage.io as io

from PIL import Image



#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"



def batch_norm(x, istraining):
    beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
    axises = [0,1,2]
    batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(istraining, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

class Model(object):
    def __init__(self, sess, batch_size):
        self.sess = sess
        self.batch_size = batch_size
        self.input_img_size = 400
        self.input_mask_size = 400
        self.total_loss = 0
        self.lr = 0.001
        self.beta1 = 0.9
        self.layers = 4
        self.in_channels=1
        self.out_channels=2
        self.root_features=32
        self.filter_size1=3
        self.filter_size2=5
        self.pool_size=2
        self.up_size=2
        self.class_prob=0.5
        self.trainloss = []
        self.convs = []
        self.pools = []
        self.concats = []
        self.output_map = None
        self.weights = None
        self.biases = None
        self.weights = []
        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=10)

    def build_model(self):  
        with tf.name_scope("inputs"):
            self.input_I = tf.placeholder(dtype=tf.float32, shape=[None, self.input_img_size, self.input_img_size, 1], name='inputI')
            self.input_gt = tf.placeholder(dtype=tf.float32, shape=[None, self.input_mask_size, self.input_mask_size, 1], name='target') 
            self.keep_prob = tf.placeholder(tf.float32)
            self.istraining = tf.placeholder(tf.bool)
            tf.summary.image("target", tf.reshape(self.input_gt[:,:,:,0], [-1,self.input_img_size,self.input_img_size,1]), max_outputs=6)
        
        with tf.device("/gpu:1"):
            with tf.name_scope("dwconv1"):
                dwconv11 = self.create_conv_net(self.input_I, in_channels=self.in_channels, out_channels=self.root_features, filter_size=self.filter_size1)
                dwconv12 = self.create_conv_net(dwconv11, self.root_features, self.root_features, filter_size=self.filter_size2)
                pool1 = pool_avg(dwconv12, self.pool_size)
            with tf.name_scope("dwconv2"):
                dwconv21 = self.create_conv_net(pool1, self.root_features, 2*self.root_features, filter_size=self.filter_size1)
                dwconv22 = self.create_conv_net(dwconv21, 2*self.root_features, 2*self.root_features, filter_size=self.filter_size2)
                pool2 = pool_avg(dwconv22, self.pool_size)
            with tf.name_scope("dwconv3"):
                dwconv31 = self.create_conv_net(pool2, 2*self.root_features, 2**2 * self.root_features, filter_size=self.filter_size1)
                dwconv32 = self.create_conv_net(dwconv31, 2**2 * self.root_features, 2**2 * self.root_features, filter_size=self.filter_size2)
                pool3 = pool_avg(dwconv32, self.pool_size)
            with tf.name_scope("dwconv4"):
                dwconv41 = self.create_conv_net(pool3, 2**2 * self.root_features, 2**3 * self.root_features, dropout=True, filter_size=self.filter_size1)
                dwconv42 = self.create_conv_net(dwconv41, 2**3 * self.root_features, 2**3 * self.root_features, dropout=True, filter_size=self.filter_size2)
                pool4 = pool_avg(dwconv42, self.pool_size)

        with tf.device("/gpu:1"):
            with tf.name_scope("upconv1"):
                upconv11 = self.create_conv_net(pool4, 2**3 * self.root_features, 2**4 * self.root_features, filter_size=self.filter_size1)
                upconv12 = self.create_conv_net(upconv11, 2**4 * self.root_features, 2**4 * self.root_features, filter_size=self.filter_size2)
                deconv1 = self.create_upconv_net(upconv12, 2**4 * self.root_features, 2**3 * self.root_features)
                concat1 = crop_and_concat(dwconv42, deconv1)
            with tf.name_scope("upconv2"):
                upconv21 = self.create_conv_net(concat1, 2**4 * self.root_features, 2**3 * self.root_features, filter_size=self.filter_size1)
                upconv22 = self.create_conv_net(upconv21, 2**3 * self.root_features, 2**3 * self.root_features, filter_size=self.filter_size2)
                deconv2 = self.create_upconv_net(upconv22, 2**3 * self.root_features, 2**2 * self.root_features)
                concat2 = crop_and_concat(dwconv32, deconv2)
            with tf.name_scope("upconv3"):
                upconv31 = self.create_conv_net(concat2, 2**3 * self.root_features, 2**2 * self.root_features, filter_size=self.filter_size1)
                upconv32 = self.create_conv_net(upconv31, 2**2 * self.root_features, 2**2 * self.root_features, filter_size=self.filter_size2)
                deconv3 = self.create_upconv_net(upconv32, 2**2 * self.root_features, 2**1 * self.root_features)
                concat3 = crop_and_concat(dwconv22, deconv3)
            with tf.name_scope("upconv4"):
                upconv41 = self.create_conv_net(concat3, 2**2 * self.root_features, 2**1 * self.root_features, filter_size=self.filter_size1)
                upconv42 = self.create_conv_net(upconv41, 2**1 * self.root_features, 2**1 * self.root_features, filter_size=self.filter_size2)
                deconv4 = self.create_upconv_net(upconv42, 2**1 * self.root_features, self.root_features)
                concat4 = crop_and_concat(dwconv12, deconv4)

            #twice normal convolutions and once 1*1 convolution
            with tf.name_scope("Outconv_layers"):
                outconv1 = self.create_conv_net(concat4, 2 * self.root_features, self.root_features)
                outconv2 = self.create_conv_net(outconv1, self.root_features, self.root_features)
                self.output_map = self.create_outconv_net(outconv2)

        with tf.name_scope("predict"):
            self.predictor = tf.clip_by_value(tf.nn.softmax(self.output_map), 1e-5, 1)
            tf.summary.image("output", tf.reshape(self.output_map[:,:,:,0], [-1,self.input_img_size,self.input_img_size,1]), max_outputs=6)

        with tf.name_scope("Loss"):
            loss0 = -1.2 * tf.reduce_mean(self.input_gt[:,:,:,0] * tf.log(self.predictor[:,:,:,0]))
            loss1 = -0.8 * tf.reduce_mean((1.0-self.input_gt[:,:,:,0]) * tf.log(self.predictor[:,:,:,1]))
            self.total_loss = loss0 + loss1
            self.train_step = self.total_loss
            tf.summary.scalar("loss",self.total_loss)
        self.merged = tf.summary.merge_all()

        print('initializing graph...')
        with tf.name_scope("Train"):
            self.u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.train_step)
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))

        self.sess.run(init_op)     
            
    def create_conv_net(self, x, in_channels, out_channels, Bn=True, dropout=False, filter_size=3):
        #variable
        stddev = np.sqrt(2/(filter_size**2 * in_channels))
        w1 = weight_variable([filter_size, filter_size, in_channels, out_channels], stddev)
        self.weights.append(w1)
        b1 = bias_variable([out_channels])
        #convolution
        conv1 = conv2d(x, w1, padding='SAME') + b1
        #BN
        if Bn == True:         
            conv1 = batch_norm(conv1, istraining=self.istraining)
        #relu
        conv1 = tf.nn.relu(conv1)
        #dropout
        if dropout == True:
            conv1 = tf.nn.dropout(conv1, self.keep_prob)
        return conv1

    def create_upconv_net(self, x, in_channels, out_channels, Bn=True, dropout=False, filter_size=3):
        up_size = self.up_size
        #variable
        stddevd = np.sqrt(2/(filter_size**2 * out_channels))
        wd = weight_variable([up_size, up_size, out_channels, in_channels], stddevd)
        self.weights.append(wd)
        bd = bias_variable([out_channels])
        #deconv       
        deconv = deconv2d(x, wd, stride=up_size)+ bd
        #BN
        if Bn == True:  
            deconv = batch_norm(deconv, istraining=self.istraining)
        #relu
        deconv = tf.nn.relu(deconv)
        #dropout
        if dropout == True:
            deconv = tf.nn.dropout(deconv, self.keep_prob)
        return deconv

    def create_outconv_net(self, x):
        stddev1 = np.sqrt(2/self.root_features)
        w1 = weight_variable([1, 1, self.root_features, self.out_channels], stddev1)
        self.weights.append(w1)
        b1 = bias_variable([self.out_channels])
        #output conv      
        return conv2d(x, w1, padding='SAME') + b1
    
    def get_accuracy(self, logits, targets):
        pred = np.where(logits<self.class_prob, 0 ,1)
        equal_array = np.equal(pred[:,:,:,0], targets[:,:,:,0])
        acc = np.mean(equal_array)
        return acc

    def train(self, images, masks, retval=False):    
        self.sess.run(self.u_optimizer, feed_dict={self.input_I: images, self.input_gt: masks, self.istraining: True, self.keep_prob:0.5})
        if retval:
            loss, train_pred = self.sess.run([self.total_loss, self.predictor], feed_dict={self.input_I: images, self.input_gt: masks, self.istraining: True, self.keep_prob:1})
            acc = self.get_accuracy(train_pred, masks)
            return loss, acc

    def eval(self, images, masks):   
        loss, eval_pred = self.sess.run([self.total_loss, self.predictor], feed_dict={self.input_I: images, self.input_gt: masks, self.istraining: False, self.keep_prob:1})
        acc = self.get_accuracy(eval_pred, masks)
        return loss, acc
    
    def save_summary(self, images, masks):
        result = self.sess.run(self.merged, feed_dict={self.input_I: images, self.input_gt: masks, self.istraining: True, self.keep_prob:1})
        return result

    def predict(self, x_test):       
        y_dummy = np.empty((x_test.shape[0], self.input_mask_size, self.input_mask_size, 1))
        prediction = self.sess.run(self.predictor, feed_dict={self.input_I: x_test, self.input_gt: y_dummy, self.istraining: False, self.keep_prob:1})
        prediction = np.where(prediction<self.class_prob, 0, 1)
        return prediction
    
    # saving checkpoint 
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # loading checkpoint file
    def load_chkpoint(self, checkpoint_path):
        print(" [*] Reading checkpoint...")
        self.saver.restore(self.sess, checkpoint_path)


def train_eval(num_files, epochs, eval_every, batch_size, images_arrays, masks_arrays, filename, chkpt_path=None):
    #train and eval sets
    train_indices = np.random.choice(num_files, int(num_files*0.9), replace=False)
    eval_indices = np.array(list(set(range(num_files))-set(train_indices)))
    x_train = images_arrays[train_indices]
    x_eval = images_arrays[eval_indices]
    y_train = masks_arrays[train_indices]
    y_eval = masks_arrays[eval_indices] 
     
    num_train = len(x_train)
    num_eval = len(x_eval)
    epochs_acc = []
    train_acc = []
    eval_acc = []
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    #save train and eval data
    np.save(filename+"npdata/"+date+"train_imgs.npy",x_train) 
    np.save(filename+"npdata/"+date+"train_masks.npy",y_train) 
    np.save(filename+"npdata/"+date+"eval_imgs.npy",x_eval) 
    np.save(filename+"npdata/"+date+"eval_masks.npy",y_eval) 
    
    #chcekpoints_path
    chkp_dir = filename+"checkpoints/"
    acc_dir = filename+"acc_data/"
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)
    if not os.path.exists(acc_dir):
        os.makedirs(acc_dir)

    #train and eval
    with tf.Session() as sess:
        mdl = Model(sess, batch_size)
        if chkpt_path != None:
            mdl.load_chkpoint(chkpt_path)
        train_writer = tf.summary.FileWriter(filename+"logs/train", sess.graph)
        eval_writer = tf.summary.FileWriter(filename+"logs/eval") 
        for epoch in range(epochs):
            rand_indices = np.random.choice(num_train, size=batch_size, replace=False)
            rand_x = x_train[rand_indices]
            rand_y = y_train[rand_indices]
            input_x = []
            input_y = []
            #data augmentation
            for i in range(batch_size):
                angle = np.random.uniform(-10.0, 10.0) + 90*np.random.randint(0,4)  
                input_x.append(ndimage.rotate(rand_x[i], angle, reshape=False, mode="constant", cval=0))
                input_y.append(ndimage.rotate(rand_y[i], angle, reshape=False, mode="constant", cval=0))
            #train step
            mdl.train(np.array(input_x), np.array(input_y)) 
            #eval
            if (epoch+1)%eval_every == 0:
                epochs_acc.append(epoch)
                #train loss summary
                rand_indices = np.random.choice(num_train, size=batch_size, replace=False)
                rand_x = x_train[rand_indices]
                rand_y = y_train[rand_indices]
                result = mdl.save_summary(rand_x, rand_y)
                train_writer.add_summary(result, epoch)
                tmp_loss, acc = mdl.train(rand_x, rand_y, retval=True)
                train_acc.append(acc) 
                print("train entropy loss: %f   train accurary: %f"%(tmp_loss, acc))
                #eval loss summary and eval
                rand_indices = np.random.choice(num_eval, size=batch_size, replace=False)
                rand_x = x_eval[rand_indices]
                rand_y = y_eval[rand_indices]        
                result = mdl.save_summary(rand_x, rand_y)
                eval_writer.add_summary(result, epoch)
                tmp_loss, acc = mdl.eval(rand_x, rand_y)
                eval_acc.append(acc)
                print("eval entropy loss: %f   eval accurary: %f"%(tmp_loss, acc))
                #save variables
                mdl.save_chkpoint(chkp_dir, model_name=date, step=epoch)

    #save acc value                      
    acc_file = acc_dir+date+'.txt'
    acc_data = [epochs_acc, train_acc, eval_acc]
    with open(acc_file, "w") as f:
        json.dump(acc_data,f)



def predict(filename, images_arrays, chkpt_path, batch_size):
    num_test = len(images_arrays)
    pred_masks = [] 
    with tf.Session() as sess:
        mdl = Model(sess, batch_size)
        
        mdl.load_chkpoint(chkpt_path)
        generations = np.ceil(num_test/batch_size).astype(int)     
        for generation in range(generations):
            start_index = batch_size * generation
            end_index = batch_size * (generation + 1) 
            if end_index >= num_test:
                end_index = num_test
            input_x = images_arrays[start_index:end_index]
            pred_mask = mdl.predict(input_x)
            pred_masks.append(pred_mask[:,:,:,0])     
    pred_masks = np.vstack(pred_masks)[:,:,:,np.newaxis]
    #return
    return pred_masks
        
    

