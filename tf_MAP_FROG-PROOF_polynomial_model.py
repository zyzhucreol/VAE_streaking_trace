#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow implementation of maximum-a-posteriori (MAP) estimator
Demonstration of MAP estimation on a measurement and a known non-linear
forward model A_fun.
@author: Zheyuan Zhu
"""

import tensorflow as tf
import numpy as np
import shutil
shutil.rmtree('__pycache__',ignore_errors=True)
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import Pulse_retrieval.forward_model_phase_params_johnathon as fwdmd
import Pulse_retrieval.xuv_spectrum.spectrum as xuv_spectrum
import priors as prior
from scipy.io import loadmat
load=loadmat('Pulse_retrieval/measured_trace_sample4_301x98.mat')
I_measure=np.float32(load['I_measure'])
I_measure=np.expand_dims(np.expand_dims(I_measure,0),3)

save_model_path='./models/tf_FROG_johnathon_sample4' # path to save trained model
save_mat_folder='./results/tf_FROG_johnathon_sample4_' # path to save reconstruction examples
log_path='/home/zyzhu/Documents/FSUB/tensorboard/log_CVAE_pulse2/tf_FROG_johnathon' # path to log training process

#%% Define FROG layer and measurement

xuv_phase = tf.get_variable('XUV_phase',initializer=1.0*np.float32(np.random.randn(1,5)))
ir_param = tf.get_variable('ir_param',initializer=1.0*np.float32(np.random.randn(1,4)))
X = tf.concat((xuv_phase,ir_param),axis=1,name='all_coeff_assembly')

# FROG
I_FROG=fwdmd.A_fun(X)
# perform pulse retrieval on streak ind_test
counts=130.0
test_streak=I_measure/np.max(I_measure)*counts
# log likelihood l=log(p(y|x))
l = tf.reduce_mean(I_FROG*counts-test_streak*tf.log(I_FROG*counts+1e-6),name='MSEY')

# log prior distribution log_prior=log(p(x))
log_prior = prior.l2(xuv_phase)

# log-posterior distribution L=log(p(x|y))
#alpha=0.001
#L = tf.add(l,alpha*log_prior,name='posterior')
L = l # ignore prior distribution

#tf.summary.image('xuv_phase',tf.reshape(xuv_phase,(-1,10,20,1)))
tf.summary.image('streak_recon',I_FROG)
tf.summary.scalar('log_likelihood',l)
#tf.summary.scalar('log_prior',log_prior)

#%% setup TensorFlow optimization
learning_rate=tf.placeholder(tf.float32,shape=[],name='learning_rate')
optimizer=tf.train.AdamOptimizer(learning_rate)
train_op=optimizer.minimize(L)

sess=tf.InteractiveSession()
model1_writer = tf.summary.FileWriter(log_path,sess.graph)
merged=tf.summary.merge_all()
tf.global_variables_initializer().run()

#%% run optimization
for i in range(10000):
    if i<=7000:
        loss,summary,_=sess.run([L,merged,train_op],feed_dict={learning_rate:1e-2})
    if i>7000:
        loss,summary,_=sess.run([L,merged,train_op],feed_dict={learning_rate:1e-3})
    model1_writer.add_summary(summary,i)
    if i % 100 ==0:
        print(loss)

#%% plot the optimization result
import matplotlib.pyplot as plt
xuv_complex=sess.run(fwdmd.xuv_taylor_to_E(xuv_phase))
xuv_amp=np.abs(xuv_complex['f_cropped'][0])
xuv_angle=np.angle(xuv_complex['f_cropped'][0])
xuv_t=np.abs(xuv_complex['t'][0])

plt.figure();
plt.plot(xuv_amp,color='blue')
plt.plot(xuv_angle,color='red')

plt.figure();
plt.plot(xuv_spectrum.tmat,np.abs(xuv_t))

#xuv_real_test=X_test[0,0:200]
#xuv_imag_test=X_test[0,200:400]
#xuv_phase_test=np.unwrap(np.arctan2(xuv_imag_test,xuv_real_test))
#plt.figure();
#plt.plot(xuv_phase_test)
#plt.plot(xuv_phase_recon[0])
#plt.figure();
#plt.plot(xuv_phase_test-xuv_phase_recon[0])
#plt.figure();
#plt.plot(xuv_real_test)
#plt.plot(xuv_imag_test)

FROG_result=sess.run(I_FROG*counts)
plt.figure();
plt.subplot(1,2,1)
plt.imshow(FROG_result[0,:,:,0])
plt.subplot(1,2,2)
plt.imshow(I_measure[0,:,:,0])
