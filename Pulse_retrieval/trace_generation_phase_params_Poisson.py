import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from forward_model_phase import xuv_taylor_to_E, ir_from_params, A_fun
from generate_data3 import add_shot_noise
import hdf5storage
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1';
N_eng,N_tau=256,38

#%% batch generate XUV, IR pulses and streaking trace
xuv_coefs_in = tf.placeholder(tf.float32, shape=[None,6])
xuv_E_prop = xuv_taylor_to_E(xuv_coefs_in)
ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
ir_E_prop = ir_from_params(ir_values_in)
#image = A_fun(tf.concat((xuv_real_batch,xuv_imag_batch,ir_real_batch,ir_imag_batch),axis=1))

#   np.array([[0.0, 0.5, 0.0, 0.0, 0.0],[1,0.5,0,0,1]]),
    #                           1st, 2nd, 3rd,4th, 5th 
    #                             -1 < x < 1
#   np.array([[1.0, 0.0, 0.0, 0.0],[0.5,0,-1,0]])
    #               CEP, central wavelength, Pulse Duration, Intensity 
    #                             -1 < x < 1

# linear phase XUV offset by CEP of IR
#xuv_coefs_train_batch=np.array([[1.0, 0.5, 0.2, 0.3, 0.2],[-1.0,0.5,0.2,0.3,0.2]])
#ir_values_train_batch=np.array([[0.5, 0.0, 0.0, 0.0],[0.4688,0,0,0]])
#xuv_coefs_train_batch=np.array([[1.0, 0.5, 0., 0., 0.],[0.0,0.5,0,0,0]])
#ir_values_train_batch=np.array([[0.5, 0.0, 0.0, 0.0],[0.4844,0,0,0]])
#xuv_coefs_train_batch=np.array([[1.0, 0.5, 0., 0., 0.],[0.5,0.5,0,0,0]])
#ir_values_train_batch=np.array([[0.5, 0.0, 0.0, 0.0],[0.4922,0,0,0]])
#xuv_coefs_train_batch=np.array([[0.5, 0.2, 0.2, 0.3, 0.2],[0.0,0.2,0.2,0.3,0.2]])
#ir_values_train_batch=np.array([[0.8, 0.0, 0.0, 1.0],[0.7922,0,0,1]])
# reference: MSEy between linear XUV phase shift
#xuv_coefs_train_batch=np.array([[1.0, 0.5, 0., 0., 0.],[-1.0,0.5,0,0,0]])
#ir_values_train_batch=np.array([[0.5, 0., 0.0, 0.0],[0.5,0,0,0]])

# generate independent copys
xuv_coefs_batch=np.random.uniform(low=-0.50,high=0.50,size=(1100,1,6))
ir_values_batch=np.random.uniform(low=-0.50,high=0.50,size=(1100,1,4))
# generate 10 ambiguities for each independent pulse
xuv_coefs_batch=np.tile(xuv_coefs_batch,(1,10,1))
ir_values_batch=np.tile(ir_values_batch,(1,10,1))
xuv_shift=np.zeros((1100,10,2))
ir_shift=ir_values_batch[:,:,0:1]
xuv_coefs_mod_batch=np.concatenate((xuv_shift,xuv_coefs_batch[:,:,2:6]),axis=2)
ir_values_mod_batch=np.concatenate((ir_shift,ir_values_batch[:,:,1:4]),axis=2)
xuv_coefs_batch=np.reshape(xuv_coefs_mod_batch,(-1,6))
ir_values_batch=np.reshape(ir_values_mod_batch,(-1,4))
xuv_coefs_train_batch=xuv_coefs_batch[0:10000,:]
ir_values_train_batch=ir_values_batch[0:10000,:]
xuv_coefs_test_batch=xuv_coefs_batch[10000:11000,:]
ir_values_test_batch=ir_values_batch[10000:11000,:]
feed_dict_train = {xuv_coefs_in: xuv_coefs_train_batch,ir_values_in: ir_values_train_batch}
feed_dict_test = {xuv_coefs_in: xuv_coefs_test_batch,ir_values_in: ir_values_test_batch}

sess=tf.InteractiveSession()
xuv_pulse_train = sess.run(xuv_E_prop, feed_dict=feed_dict_train)
ir_pulse_train = sess.run(ir_E_prop, feed_dict=feed_dict_train)
xuv_pulse_test = sess.run(xuv_E_prop, feed_dict=feed_dict_test)
ir_pulse_test = sess.run(ir_E_prop, feed_dict=feed_dict_test)

#%% generated xuv, ir pulses, and streaking traces
xuv_real_train=np.real(xuv_pulse_train)
xuv_imag_train=np.imag(xuv_pulse_train)
ir_real_train=np.real(ir_pulse_train)
ir_imag_train=np.imag(ir_pulse_train)
I_train = sess.run(A_fun(tf.concat((np.float32(xuv_coefs_train_batch),np.float32(ir_values_train_batch)),axis=1)))
xuv_real_test=np.real(xuv_pulse_test)
xuv_imag_test=np.imag(xuv_pulse_test)
ir_real_test=np.real(ir_pulse_test)
ir_imag_test=np.imag(ir_pulse_test)
I_test = sess.run(A_fun(tf.concat((np.float32(xuv_coefs_test_batch),np.float32(ir_values_test_batch)),axis=1)))

#%% add Poisson noise to generated training and test streaks
counts_train=np.kron(np.ones(1000,),np.array([2,3,5,10,12,16,32,60,80,100]))
counts_test=np.kron(np.ones(100,),np.array([2,3,5,10,12,16,32,60,80,100]))
#counts_train=2*np.ones(10000)
#counts_test=2*np.ones(1000)
for kk in range(np.shape(I_train)[0]):
    I_train[kk]=add_shot_noise(I_train[kk],counts_train[kk])
for kk in range(np.shape(I_test)[0]):
    I_test[kk]=add_shot_noise(I_test[kk],counts_test[kk])

#%% plot generated trace
ind1=5
ind2=1
plt.close('all')
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(np.real(xuv_pulse_train[ind1]), color="blue")
plt.plot(np.imag(xuv_pulse_train[ind1]), color="red")
plt.subplot(1,2,2)
plt.plot(np.real(xuv_pulse_train[ind2]), color="blue")
plt.plot(np.imag(xuv_pulse_train[ind2]), color="red")
plt.figure(2)
plt.subplot(1,2,1)
plt.plot(np.real(ir_pulse_train[ind1]), color="blue")
plt.plot(np.imag(ir_pulse_train[ind1]), color="red")
plt.subplot(1,2,2)
plt.plot(np.real(ir_pulse_train[ind2]), color="blue")
plt.plot(np.imag(ir_pulse_train[ind2]), color="red")
plt.figure(3)
plt.subplot(1,2,1)
plt.pcolormesh(I_train[ind1,:,:,0])
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolormesh(I_train[ind2,:,:,0])
plt.colorbar()
MSEy=np.mean(np.mean((I_train[ind1,:,:]-I_train[ind2,:,:])**2,axis=0),axis=0)
MSEx=np.mean(np.abs(xuv_pulse_train[ind1]-xuv_pulse_train[ind2])**2,axis=0)
print(MSEx)
print(MSEy)
# measured trace
#plt.figure(4)
#plt.pcolormesh(measured_trace.delay, measured_trace.energy, measured_trace.trace)

#%% save training and test data
filename='/media/HDD1/zyz/FROG_train_Poisson/XUV_IR_train_Poisson_hard_mixed_2-100_myA.mat'
savemat(filename,\
        {'counts_train':counts_train,\
         'xuv_coefs_train':xuv_coefs_train_batch,'ir_values_train':ir_values_train_batch,\
         'xuv_real_train':xuv_real_train,'xuv_imag_train':xuv_imag_train,'ir_real_train':ir_real_train,'ir_imag_train':ir_imag_train,'I_train':I_train})
filename='/media/HDD1/zyz/FROG_train_Poisson/XUV_IR_test_Poisson_hard_mixed_2-100_myA.mat'
savemat(filename,\
        {'counts_test':counts_test,\
        'xuv_coefs_test':xuv_coefs_test_batch,'ir_values_test':ir_values_test_batch,\
         'xuv_real_test':xuv_real_test,'xuv_imag_test':xuv_imag_test,'ir_real_test':ir_real_test,'ir_imag_test':ir_imag_test,'I_test':I_test})

#%% load generated training data
load=hdf5storage.loadmat('/media/HDD1/zyz/FROG_train_Poisson/XUV_IR_train_Poisson_hard_mixed_2-100_myA.mat')
xuv_coefs_train=np.float32(load['xuv_coefs_train'])
ir_values_train=np.float32(load['ir_values_train'])
Er_train=np.float32(load['xuv_real_train'])
Ei_train=np.float32(load['xuv_imag_train'])
Gr_train=np.float32(load['ir_real_train'])
Gi_train=np.float32(load['ir_imag_train'])
I_train=np.float32(load['I_train'])
load=hdf5storage.loadmat('/media/HDD1/zyz/FROG_train_Poisson/XUV_IR_test_Poisson_hard_mixed_2-100_myA.mat')
Er_test=np.float32(load['xuv_real_test'])
Ei_test=np.float32(load['xuv_imag_test'])
Gr_test=np.float32(load['ir_real_test'])
Gi_test=np.float32(load['ir_imag_test'])
I_test=np.float32(load['I_test'])
tf.reset_default_graph()
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':1}))
X_batch=np.concatenate((np.float32(xuv_coefs_train[0:128,:]),np.float32(ir_values_train[0:128,:])),axis=1)
I2_train=sess.run(A_fun(X_batch))
I2_train=np.reshape(I2_train,[-1,N_eng,N_tau])
ind1=0
ind2=9
plt.close('all')
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(Er_train[ind1,:], color="blue")
plt.plot(Ei_train[ind1,:], color="red")
plt.subplot(1,2,2)
plt.plot(Er_train[ind2,:], color="blue")
plt.plot(Ei_train[ind2,:], color="red")
plt.figure(2)
plt.subplot(1,2,1)
plt.plot(Gr_train[ind1,:], color="blue")
plt.plot(Gi_train[ind1,:], color="red")
plt.subplot(1,2,2)
plt.plot(Gr_train[ind2,:], color="blue")
plt.plot(Gi_train[ind2,:], color="red")
plt.figure(3)
plt.subplot(1,2,1)
plt.pcolormesh(I_train[ind1,:,:,0])
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolormesh(I_train[ind2,:,:,0])
plt.colorbar()
plt.figure(4)
plt.subplot(1,2,1)
plt.plot(20*np.sqrt(np.square(Er_train[ind1,:])+np.square(Ei_train[ind1,:])), color="blue")
plt.plot(np.unwrap(np.arctan2(Ei_train[ind1,:],Er_train[ind1,:])), color="red")
plt.subplot(1,2,2)
plt.plot(20*np.sqrt(np.square(Er_train[ind2,:])+np.square(Ei_train[ind2,:])), color="blue")
plt.plot(np.unwrap(np.arctan2(Ei_train[ind2,:],Er_train[ind2,:])), color="red")
MSEy=np.mean(np.mean((I_train[ind1,:,:,0]-I_train[ind2,:,:,0])**2,axis=0),axis=0)
MSEx=np.mean((Er_train[ind1,:]-Er_train[ind2,:])**2+(Ei_train[ind1,:]-Ei_train[ind2,:])**2,axis=0)
print(MSEx)
print(MSEy)