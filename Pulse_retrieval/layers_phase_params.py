#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encoder/decoder structures for the pulse retrieval model

@author: zyzhu
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, Flatten
from tensorflow.contrib.rnn import BasicLSTMCell
from aux_layers import sampler

def encoderY(args):
    with tf.name_scope('encoderY'):
        y=tf.keras.Input(shape=(args.N_eng,args.N_tau,1),name='Input_encoderY')
        I_conv=Conv2D(args.d_channels,(11,7),(1,1),padding='same',kernel_initializer='glorot_normal')(y)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Conv2D(args.d_channels*2,(11,7),(2,2),padding='same',kernel_initializer='glorot_normal')(I_conv)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Conv2D(args.d_channels*4,(7,5),(2,2),padding='same',kernel_initializer='glorot_normal')(I_conv)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Conv2D(args.d_channels*8,(5,3),(2,2),padding='same',kernel_initializer='glorot_normal')(I_conv)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Conv2D(args.d_channels*16,(3,3),(2,2),padding='same',kernel_initializer='glorot_normal')(I_conv)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Conv2D(args.d_channels*32,(3,3),(2,2),padding='same',kernel_initializer='glorot_normal')(I_conv)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Conv2D(args.d_channels*64,(3,3),(2,2),padding='same',kernel_initializer='glorot_normal')(I_conv)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Conv2D(args.d_channels*64,(3,3),(2,2),padding='same',kernel_initializer='glorot_normal')(I_conv)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Conv2D(args.enc_size,(2,2),(2,2),padding='same',kernel_initializer='glorot_normal')(I_conv)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Flatten()(I_conv)
        I_conv=LeakyReLU()(I_conv)
        encodery_model=tf.keras.Model(inputs=y,outputs=I_conv,name='encoderY')
        return encodery_model

def encoderX(args):
    with tf.name_scope('encoderX'):
        x=tf.keras.Input(shape=(args.X_dim,),name='Input_encoderX')
        x_temp=Dense(args.enc_size,kernel_initializer='glorot_normal')(x)
        x_temp=LeakyReLU()(x_temp)
        x_temp=Dense(args.enc_size,kernel_initializer='glorot_normal')(x_temp)
        encoderx_model=tf.keras.Model(inputs=x,outputs=x_temp,name='encoderX')
        return encoderx_model

def decoder(args):
    with tf.name_scope('decoder'):
        y=tf.keras.Input(shape=(args.dec_size,),name='Input_decoder')
        I_out=Dense(args.X_dim,kernel_initializer='glorot_normal')(y)
        decoder_model=tf.keras.Model(inputs=y,outputs=I_out,name='decoder')
        return decoder_model

#%% building-block models: recogntion, conditional prior and generator
class recognition_encoder(tf.keras.Model):
    def __init__(self, args):
        super(recognition_encoder,self).__init__(name='recog_enc')
        with tf.name_scope('recog_enc'):
            self.encodeY = encoderY(args)
            self.encodeX = encoderX(args)
        
    def call(self,y,x):
        ry = self.encodeY(y)
        rx = self.encodeX(x)
        ryx= tf.concat((ry,rx),axis=-1)
        return ryx
    
class recognition_model(tf.keras.Model):
    def __init__(self, args):
        super(recognition_model,self).__init__(name='recognition')
        with tf.name_scope('recognition'):
            self.encodeY = encoderY(args)
            self.encodeX = encoderX(args)
            self.LSTM_enc_cell = BasicLSTMCell(args.enc_size,name='LSTM_encoder')
            self.sample=sampler(args.enc_size,args.z_size)
            #add LSTM_enc_cell to graphy with a dummy call to itself
#            h=tf.zeros_like(tf.keras.Input(shape=(2*args.enc_size,)))
#            state=self.LSTM_enc_cell.zero_state(args.batch_size,'float32')
#            state=self.LSTM_enc_cell.get_initial_state(dtype='float32',batch_size=args.batch_size)
#            self.LSTM_enc_cell(h,state)
        
    def call(self,y,x,state):
        ry = self.encodeY(y)
        rx = self.encodeX(x)
        ryx= tf.concat((ry,rx),axis=-1)
        h_enc,state_new = self.LSTM_enc_cell(ryx,state)
        z,mu,logsigma,sigma = self.sample(h_enc)
        dist_params = [mu,logsigma,sigma]
        return z,dist_params,ry,state_new
    
class cond_prior_model(tf.keras.Model):
    def __init__(self, args):
        super(cond_prior_model,self).__init__(name='cond_prior')
        with tf.name_scope('cond_prior'):
            self.encodeY = encoderY(args)
            self.LSTM_enc_cell = BasicLSTMCell(args.enc_size,name='LSTM_encoder')
            self.sample=sampler(args.enc_size,args.z_size)
#            #add LSTM_enc_cell to graph with a dummy call to itself
#            h=tf.zeros_like(tf.keras.Input(shape=(args.enc_size,)))
#            state=self.LSTM_enc_cell.zero_state(args.batch_size,'float32')
#            state=self.LSTM_enc_cell.get_initial_state(dtype='float32',batch_size=args.batch_size)
#            self.LSTM_enc_cell(h,state)
        
    def call(self,y,state):
        ry = self.encodeY(y)
        h_enc,state_new = self.LSTM_enc_cell(ry,state)
        z,mu,logsigma,sigma = self.sample(h_enc)
        dist_params = [mu,logsigma,sigma]
        return z,dist_params,ry,state_new

class generator_model(tf.keras.Model):
    def __init__(self, args):
        super(generator_model,self).__init__(name='generator')
        with tf.name_scope('generator'):
            self.decode = decoder(args)
            self.LSTM_dec_cell = BasicLSTMCell(args.dec_size,name='LSTM_decoder')
            #add LSTM_enc_cell to graph with a dummy call to itself
#            h=tf.zeros_like(tf.keras.Input(shape=(args.z_size+args.enc_size,)))
#            state=self.LSTM_dec_cell.zero_state(args.batch_size,'float32')
#            state=self.LSTM_dec_cell.get_initial_state(dtype='float32',batch_size=args.batch_size)
#            self.LSTM_dec_cell(h,state)
        
    def call(self,ry,z,state):
        ryz = tf.concat((ry,z),axis=-1)
        h_dec,state_new = self.LSTM_dec_cell(ryz,state)
        x = self.decode(h_dec)
        return x, state_new