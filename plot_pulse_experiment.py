#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot spectrum and time-domain pulse from parameters

@author: zyzhu
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import Pulse_retrieval.xuv_spectrum.spectrum as xuv_spectrum
from Pulse_retrieval.forward_model import xuv_taylor_to_E, ir_from_params
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1';

def MSE(X,X0):
    max_X=np.max(X0)
    X_normalized=X/max_X
    X0_normalized=X0/max_X
    NMSE=np.mean(np.square(np.abs(X_normalized-X0_normalized)))
    return NMSE

def calc_fwhm(tmat, I_t):
    half_max = np.max(I_t)/2
    index1 = 0
    index2 = len(I_t) - 1

    while I_t[index1] < half_max:
        index1 += 1
    while I_t[index2] < half_max:
        index2 -= 1

    t1 = tmat[index1]
    t2 = tmat[index2]
    fwhm = t2 - t1
    return fwhm, t1, t2, half_max

load=loadmat('/media/HDD1/zyz/FROG_train_Poisson/XUV_IR_test_Poisson_hard2_myA.mat')
I_test=load['I_test']
counts_test=load['counts_test']
xuv_coefs_test=np.float32(load['xuv_coefs_test'])
ir_values_test=np.float32(load['ir_values_test'])

sess=tf.InteractiveSession()

n_instances=10
ind_plot=48
MSE_trace_experiment=np.zeros((n_instances))
fwhm_experiment=np.zeros((n_instances))
for ii in range(n_instances):
    load=loadmat('./RcVAE_results/CVAE_pulse_phase_Poisson_3_experiment_trace_{}.mat'.format(str(ii)))
    phase_params_recon=load['coefs_test']
    I_recon=load['AX_array']*120
    xuv_coefs_recon=phase_params_recon[:,0:6]
    ir_values_recon=phase_params_recon[:,6:10]
    load=loadmat('Pulse_retrieval/measured_trace.mat')
    I_measure=np.float32(load['I_measure'])
    
    xuv_pulse_recon=sess.run(xuv_taylor_to_E(xuv_coefs_recon))
    ir_puls_recone=sess.run(ir_from_params(ir_values_recon))
    xuv_Ef_recon=xuv_pulse_recon['f']
    xuv_Ef_recon_cropped=xuv_pulse_recon['f_cropped']
    xuv_Et_recon=xuv_pulse_recon['t']
    
    ind_plot=0
    
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.pcolormesh(I_recon[ind_plot,:,:,0],cmap='jet')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.pcolormesh(I_measure,cmap='jet')
    plt.colorbar()
    plt.savefig('./output/I_measure_'+str(ii)+'.png')
    plt.close()
    MSE_trace_experiment[ii]=MSE(I_recon[ind_plot,:,:,0],I_measure)
    
    plt.figure(2)
    fig,ax1=plt.subplots()
    ax1.plot(xuv_spectrum.fmat_cropped,np.abs(xuv_Ef_recon_cropped[ind_plot,:]),color='blue')
    ax2=ax1.twinx()
    ax2.plot(xuv_spectrum.fmat_cropped,np.unwrap(np.angle(xuv_Ef_recon_cropped[ind_plot,:])),color='red')
    plt.savefig('./output/Ef_recon_experiment_'+str(ii)+'.png')
    plt.close()
    
    plt.figure(3)
    fig,ax1=plt.subplots()
    ax1.plot(xuv_spectrum.tmat,np.abs(xuv_Et_recon[ind_plot,:]),color='blue')
    ax2=ax1.twinx()
    ax2.plot(xuv_spectrum.tmat,np.unwrap(np.angle(xuv_Et_recon[ind_plot,:])),color='red')
    plt.savefig('./output/Et_recon_experiment_'+str(ii)+'.png')
    plt.close()
    fwhm_experiment[ii],_,_,_=calc_fwhm(xuv_spectrum.tmat,np.abs(xuv_Et_recon[ind_plot,:]))