"""
This script is an add-on to main_pulse_phase_params_mixed_noise.py to 
visualize the latent space. Prior run of the main file required.

@author: Zheyuan Zhu
"""
import numpy as np
from scipy.io import savemat
from Pulse_retrieval.read_data_phase_params import read_data
X_train,I_train,counts_train,X_test,I_test,counts_test=read_data(\
                                            '/media/HDD1/zyz/FROG_train_Poisson/XUV_IR_train_Poisson_hard_mixed_myA.mat',\
                                            '/media/HDD1/zyz/FROG_train_Poisson/XUV_IR_test_Poisson_hard_mixed_myA.mat')

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1';

#%% feed test data of different noise level
ind_20=np.nonzero(counts_test==5)[0]
ind_20=ind_20[0:100]
ind_40=np.nonzero(counts_test==21)[0]
ind_40=ind_40[0:100]
ind_110=np.nonzero(counts_test==100)[0]
ind_110=ind_110[0:100]
ind_test=np.concatenate((ind_20,ind_40,ind_110))
Y_mb_test=I_test[ind_test,:]
feed_dict_test={y:Y_mb_test}
mup0,sigmap0=sess.run([mu_p[0],sigma_p[0]],feed_dict_test)
savemat('./RcVAE_results/'+save_name+'_zp.mat',
        {'mup0':mup0,'sigmap0':sigmap0})