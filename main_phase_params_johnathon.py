#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pulse retrieval example

@author: Zheyuan Zhu
"""
import numpy as np
from scipy.io import savemat
import tensorflow as tf
from aux_layers import KL_calc, sampler
from Pulse_retrieval.layers_phase_params_johnathon import recognition_encoder, encoderY, decoder#, LSTM_decoder
from tensorflow.contrib.rnn import BasicLSTMCell
from Pulse_retrieval.forward_model_phase_params_johnathon import A_fun, xuv_taylor_to_E, ir_from_params
from Pulse_retrieval.read_data_phase_params import read_data

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1';

#%% model parameters
args=type('',(),{})()
args.batch_size = 20 # training batch size
args.train_iters=50000 # number of training iterations
args.learning_rate=1e-4 # learning rate for optimizer

# encoder/decoder arguments
args.T=1 # number of recurrences
args.d_channels=4
args.enc_size = 256 # number of hidden units (channels) in LSTM_encode output
args.dec_size = 128 # number of hidden units (channels) in LSTM_decode output
args.z_size=128 # Sampler output size
# foward model parameters
args.N_xuv=5 # number of XUV phase parameters
args.N_ir=4 # number of IR parameters
args.X_dim=args.N_xuv+args.N_ir
args.N_eng=301
args.N_tau=38
args.counts=10 # Poission peak counts in experiment

# file I/O parameters
load_model=False
save_name='CVAE_pulse_phase_Poisson_johnathon_10counts'
save_model_path='./models/'+save_name # path to save trained model
load_model_path='./models/CVAE_pulse_phase_Poisson_4' # path to load pre-trained weights, if load_model=True
save_mat_folder='./RcVAE_results/'+save_name+'_' # path to save reconstruction examples
log_path='/home/zyzhu/Documents/FSUB/tensorboard/log_CVAE_pulse/'+save_name # Tensorboard path to log training process

#%% build RcVAE graph
x = tf.placeholder(tf.float32,shape=(None,args.X_dim),name='X')
y = tf.placeholder(tf.float32,shape=(None,args.N_eng,args.N_tau,1),name='Y')
batch_size_flexible=tf.shape(y)[0]

x_q,mu_q,logsigma_q,sigma_q=[0]*args.T,[0]*args.T,[0]*args.T,[0]*args.T
x_p,mu_p,logsigma_p,sigma_p=[0]*args.T,[0]*args.T,[0]*args.T,[0]*args.T

recog_encode=recognition_encoder(args)
recog_lstm=BasicLSTMCell(args.enc_size,name='recog_lstm')
recog_lstm_states=recog_lstm.zero_state(batch_size_flexible,'float32')
recog_sample=sampler(args.enc_size,args.z_size,name='recog_sampler')

cond_prior_encode=encoderY(args)
cond_prior_lstm=BasicLSTMCell(args.enc_size,name='cond_prior_lstm')
cond_prior_lstm_states=cond_prior_lstm.zero_state(batch_size_flexible,'float32')
cond_prior_sample=sampler(args.enc_size,args.z_size,name='cond_prior_sampler')

dec_lstm=BasicLSTMCell(args.dec_size,name='dec_lstm')
dec_lstm_state_q=dec_lstm.zero_state(batch_size_flexible,'float32')
dec_lstm_state_p=dec_lstm.zero_state(batch_size_flexible,'float32')

decode=decoder(args)

for t in range(args.T):
    # inference model
    x_q_prev = tf.zeros(shape=(),name='xq_init') if t==0 else x_q[t-1]
    delta_x = x if t==0 else x_q_prev
    deltay_q = y if t==0 else y-A_fun(x_q_prev)
    ryx = recog_encode(deltay_q,delta_x)
    h_enc_q,recog_lstm_states = recog_lstm(ryx,recog_lstm_states,scope='recog_lstm')
    z_q,mu_q[t],logsigma_q[t],sigma_q[t] = recog_sample(h_enc_q)
    h_dec_q,dec_lstm_state_q = dec_lstm(z_q,dec_lstm_state_q)
    dxq = decode(h_dec_q)
    x_q[t] = tf.add(x_q_prev,dxq,name='add_dxq{}'.format(str(t+1)))
    # retrieval model
    x_p_prev = tf.zeros(shape=(),name='xp_init') if t==0 else x_p[t-1]
    deltay_p = y if t==0 else y-A_fun(x_p_prev)
    rdy_p = cond_prior_encode(deltay_p)
    h_enc_p,cond_prior_lstm_states = cond_prior_lstm(rdy_p,cond_prior_lstm_states,scope='cond_prior_lstm')
    z_p,mu_p[t],logsigma_p[t],sigma_p[t] = cond_prior_sample(h_enc_p)
    h_dec_p,dec_lstm_state_p = dec_lstm(z_p,dec_lstm_state_p)
    dxp = decode(h_dec_p)
    x_p[t] = tf.add(x_p_prev,dxp,name='add_dxp{}'.format(str(t+1)))
    
#%% Define loss function and optimizer
Xq=x_q[-1]
Xp=x_p[-1]
Y_hat=A_fun(Xp)
KL_loss=KL_calc(mu_q,mu_p,logsigma_q,logsigma_p,sigma_q,sigma_p)
Lx=tf.losses.mean_squared_error(labels=x,predictions=Xq,scope='pf')
Ly=tf.reduce_mean(Y_hat*args.counts-y*tf.log(Y_hat*args.counts+1e-6),name='pg')
L_hybrid=tf.add(tf.add(Lx,0.001*KL_loss,'L'),Ly,'L_hybrid')

def phase2spectrum(X_batch):
    with tf.name_scope('phase2spectrum'):
        xuv_phase=X_batch[:,0:5]
        ir_coeff=X_batch[:,5:9]
        xuv_complex=xuv_taylor_to_E(xuv_phase)
        ir_complex=ir_from_params(ir_coeff)
        xuv_real=tf.real(xuv_complex['f_cropped'])
        xuv_imag=tf.imag(xuv_complex['f_cropped'])
        ir_real=tf.real(ir_complex['E_prop']['f_cropped'])
        ir_imag=tf.imag(ir_complex['E_prop']['f_cropped'])
        X_display=tf.concat((xuv_real,xuv_imag,ir_real,ir_imag),axis=1)
        return X_display
    
Xq_display=phase2spectrum(Xq)
Xp_display=phase2spectrum(Xp)
x_display=phase2spectrum(x)
# Tensorboard monitoring
tf.summary.scalar('MSE_X',Lx)
tf.summary.scalar('MSE_Y',Ly)
tf.summary.scalar('KL',KL_loss)
tf.summary.image('Xq',tf.reshape(Xq_display,[-1,11,40,1]))
tf.summary.image('Xp',tf.reshape(Xp_display,[-1,11,40,1]))
tf.summary.image('X_true',tf.reshape(x_display,[-1,11,40,1]))
tf.summary.image('Y_true',y)
tf.summary.image('Yp',Y_hat)
merged=tf.summary.merge_all()

# optimizer
with tf.name_scope('train'):
    optimizer1=tf.train.AdamOptimizer(args.learning_rate)
    train_op1=optimizer1.minimize(L_hybrid)

# Setup training iterations
fetches=[Lx,Ly,KL_loss,train_op1,merged]
fetches_test=[Lx,Ly,KL_loss,merged]

sess=tf.InteractiveSession()
saver = tf.train.Saver()

X_train,I_train,_,X_test,I_test,_=read_data(\
                                            '/media/HDD1/zyz/FROG_train_Poisson/XUV_IR_train_Poisson_johnathon_10counts_myA.mat',\
                                            '/media/HDD1/zyz/FROG_train_Poisson/XUV_IR_test_Poisson_johnathon_10counts_myA.mat')

# choose to initialize the weights or restore the weights from trained model
if load_model is True:
    saver.restore(sess,load_model_path)
else:
    tf.global_variables_initializer().run()

    model1_writer = tf.summary.FileWriter(log_path,sess.graph)
    model2_writer = tf.summary.FileWriter(log_path+'_test',sess.graph)
    
    #%% Run training iterations
    for i in range(40000,args.train_iters):
        ind_train = np.random.randint(0,np.size(I_train,axis=0),size=args.batch_size)
        Y_mb = I_train[ind_train,:]
        X_mb = X_train[ind_train,:]
        
        feed_dict={x:X_mb,y:Y_mb}
        results=sess.run(fetches,feed_dict)
        Lxs,Lys,Lzs,_,summary_md1=results
        if i%100==0:
            ind_test = np.random.randint(0,np.size(I_test,axis=0),size=args.batch_size)
            Y_mb_test = I_test[ind_test,:]
            X_mb_test = X_test[ind_test,:]
            feed_dict_test={x:X_mb_test,y:Y_mb_test}
            Lxs,Lys,Lzs,summary_test=sess.run(fetches_test,feed_dict_test)
            print("iter=%d : MSE: %f fidelity: %f KL: %f" % (i,Lxs,Lys,Lzs))
        model1_writer.add_summary(summary_md1, i)
        model2_writer.add_summary(summary_test, i)
    #save trained weights
    save_path=saver.save(sess,save_model_path)

#%% generate and save reconstruction examples
n_instance=10
n_samples=1000
# feed test data into retrieval network
for kk in range(n_instance):
    samples_test = sess.run(Xp,feed_dict={y:I_test})
    MSE_test=np.mean(np.mean(np.mean((samples_test-X_test[0:n_samples,:])**2,axis=1)))
    PSNR_test=10*np.log10(1/MSE_test)
    AX_array=sess.run(A_fun(samples_test))
    fidelity_test=np.mean(np.mean(np.mean((AX_array-I_test[0:n_samples,:])**2,axis=1)))
    X_display=sess.run(phase2spectrum(samples_test))
    savemat(save_mat_folder+'{}_test_branch2.mat'.format(str(kk)),\
            {'coefs_test':samples_test,\
             'sample_test':X_display,'AX_array':AX_array,'MSE_test':MSE_test,'fidelity_test':fidelity_test})
    
#%% load experiment trace
#from scipy.io import loadmat
#load=loadmat('Pulse_retrieval/measured_trace.mat')
#I_measure=np.float32(load['I_measure'])
#I_measure=np.expand_dims(np.expand_dims(I_measure,0),3)
#for kk in range(n_instance):
#    samples_test = sess.run(Xp,feed_dict={y:I_measure})
#    X_display=sess.run(phase2spectrum(samples_test))
#    AX_array=sess.run(A_fun(samples_test))
#    savemat(save_mat_folder+'experiment_trace_{}.mat'.format(str(kk)),\
#                {'coefs_test':samples_test,\
#                 'sample_test':X_display,'AX_array':AX_array})