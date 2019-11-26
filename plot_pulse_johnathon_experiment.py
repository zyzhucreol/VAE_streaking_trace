#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot spectrum and time-domain pulse from parameters

@author: zyzhu
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat,savemat
import Pulse_retrieval.xuv_spectrum.spectrum as xuv_spectrum
from Pulse_retrieval.forward_model_phase_params_johnathon import xuv_taylor_to_E, ir_from_params
from Pulse_retrieval.forward_model_phase_params_johnathon import A_fun
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0';

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

load=loadmat('Pulse_retrieval/measured_trace_sample4_301x98.mat')
I_ideal=np.float32(load['I_measure'])
I_ideal=I_ideal/np.max(I_ideal)
I_ideal=np.expand_dims(np.expand_dims(I_ideal,0),3)

xuv_coefs_in = tf.placeholder(tf.float32, shape=[None,5])
xuv_E_prop = xuv_taylor_to_E(xuv_coefs_in)
ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
ir_E_prop = ir_from_params(ir_values_in)
trace_image=A_fun(tf.concat((xuv_coefs_in,ir_values_in),axis=1))
sess=tf.InteractiveSession()

save_fig=False
loadname='CVAE_pulse_phase_mixed_Poisson_johnathon_sample4_no_MSEY_10-100'
n_instances=10
ind_plot_list=np.array([0])
MSE_trace=np.zeros((n_instances,np.size(ind_plot_list)))
fwhm=np.zeros((n_instances,np.size(ind_plot_list)))
fwhm_true=np.zeros((n_instances,np.size(ind_plot_list)))
I_recon_group=np.zeros((n_instances,np.size(ind_plot_list),np.shape(I_ideal)[1],np.shape(I_ideal)[2]))
I_ideal_group=np.zeros((n_instances,np.size(ind_plot_list),np.shape(I_ideal)[1],np.shape(I_ideal)[2]))
Et_recon_group=np.zeros((n_instances,2048),dtype='complex64')
Ef_recon_group=np.zeros((n_instances,232),dtype='complex64')
phase_params_group=np.zeros((n_instances,9))

for ii in range(n_instances):
    load=loadmat('./RcVAE_results/'+loadname+'_experiment_trace_{}.mat'.format(str(ii)))
    phase_params_recon=load['coefs_test']
    I_recon=load['AX_array']
    xuv_coefs_recon=phase_params_recon[ind_plot_list,0:5]
    ir_values_recon=phase_params_recon[ind_plot_list,5:9]
    
    feed_dict_recon = {xuv_coefs_in: xuv_coefs_recon,ir_values_in: ir_values_recon}
    xuv_Ef_recon_cropped=sess.run(xuv_E_prop['f_cropped'], feed_dict=feed_dict_recon)
    xuv_Et_recon=sess.run(xuv_E_prop['t_photon'], feed_dict=feed_dict_recon)
    Et_recon_group[ii]=xuv_Et_recon
    Ef_recon_group[ii]=xuv_Ef_recon_cropped
    phase_params_group[ii]=phase_params_recon
    
    for jj in range(np.size(ind_plot_list)):
        ind_plot=ind_plot_list[jj]
        #%% plot ind_plot-th pulse and reconstructed trace
        if save_fig:
            filename=loadname+'_ind_plot_{}'.format(str(ind_plot))
            os.makedirs('./output/'+filename,exist_ok=True)
            plt.figure(1)
            plt.pcolormesh(I_ideal[ind_plot,:,:,0],cmap='jet')
            plt.colorbar()
            plt.savefig('./output/'+filename+'/I_ideal_'+str(ii)+'.png')
            plt.close()
            
            plt.figure(2)
            plt.subplot(1,2,1)
            plt.pcolormesh(I_recon[ind_plot,:,:,0],cmap='jet')
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.pcolormesh(I_ideal[ind_plot,:,:,0],cmap='jet')
            plt.colorbar()
            plt.savefig('./output/'+filename+'/I_recon_'+str(ii)+'.png')
            plt.close()
        
        MSE_trace[ii,jj]=MSE(I_recon[ind_plot,:,:,0],I_ideal[ind_plot,:,:,0])
        I_recon_group[ii,jj]=I_recon[ind_plot,:,:,0]
        I_ideal_group[ii,jj]=I_ideal[ind_plot,:,:,0]
        
        if save_fig:
            plt.figure(3)
            fig,ax1=plt.subplots()
            ax1.plot(xuv_spectrum.fmat_cropped,np.abs(xuv_Ef_recon_cropped[ind_plot,:]),color='blue')
            ax2=ax1.twinx()
            ax2.plot(xuv_spectrum.fmat_cropped,np.unwrap(np.angle(xuv_Ef_recon_cropped[ind_plot,:])),color='red')
            plt.savefig('./output/'+filename+'/XUV_Ef_recon_'+str(ii)+'.png')
            plt.close()
            
            plt.figure(4)
            fig,ax1=plt.subplots()
            ax1.plot(xuv_spectrum.tmat,np.abs(xuv_Et_recon[ind_plot,:]),color='blue')
            ax2=ax1.twinx()
            ax2.plot(xuv_spectrum.tmat,np.unwrap(np.angle(xuv_Et_recon[ind_plot,:])),color='red')
            plt.savefig('./output/'+filename+'/XUV_Et_recon_'+str(ii)+'.png')
            plt.close()
        fwhm[ii,jj],_,_,_=calc_fwhm(xuv_spectrum.tmat,np.abs(xuv_Et_recon[ind_plot,:]))

    print(ii)   
savemat('./RcVAE_results/'+loadname+'_experiment_statistics.mat',\
        {'MSE_trace':MSE_trace,
         'I_recon_group':I_recon_group,'I_ideal_group':I_ideal_group,
         'fwhm':fwhm,
         'xuv_Ef_recon':Ef_recon_group,
         'xuv_Et_recon':Et_recon_group,
         'tmat':xuv_spectrum.tmat})