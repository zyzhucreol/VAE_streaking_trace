#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library of common log-prior distributions
@author: Zheyuan Zhu
"""

import tensorflow as tf

def l2(x):
    logp=tf.losses.mean_squared_error(labels=tf.zeros_like(x), predictions=x,scope='l2_prior')
    return logp

def l1(x):
    logp=tf.losses.absolute_difference(labels=tf.zeros_like(x), predictions=x,scope='l1_prior')
    return logp

def nonneg(x):
    logp=tf.losses.mean_squared_error(labels=tf.zeros_like(x), predictions=tf.nn.relu(-x),scope='noneg_prior')
    return logp

def TV(x):
    # x must be a tensor with batchXRowsXColsXchannels
    with tf.name_scope('TV_prior'):
        # differentiation along horizontal (col) direction
        x_H1=x[:,:-1,1:,:]
        x_H2=x[:,:-1,:-1,:]
        x_H3=x[:,-1:,1:,:]
        x_H4=x[:,-1:,:-1,:]
        # differentiation along vertical (row) direction
        x_V1=x[:,1:,:-1,:]
        x_V2=x[:,:-1,:-1,:]
        x_V3=x[:,1:,-1:,:]
        x_V4=x[:,:-1,-1:,:]
        logp=tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(x_H1-x_H2)+tf.square(x_V1-x_V2)),axis=1),axis=1)+\
            tf.reduce_mean(tf.reduce_mean(tf.abs(x_H3-x_H4),axis=1),axis=1)+\
            tf.reduce_mean(tf.reduce_mean(tf.abs(x_V3-x_V4),axis=1),axis=1)
        return tf.reduce_mean(logp)