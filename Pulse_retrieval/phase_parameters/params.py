import numpy as np
import scipy.constants as sc
import pickle
import os
import sys
sys.path.append('./Pulse_retrieval')
import measured_trace.get_trace as measured_trace

#infrared params
# for sample 2
# ir_param_amplitudes = {}
# ir_param_amplitudes["phase_range"] = (0, 2 * np.pi)
# ir_param_amplitudes["clambda_range"] = (1.6345, 1.6345)
# # ir_param_amplitudes["clambda_range"] = (1.0, 1.6345)
# ir_param_amplitudes["pulseduration_range"] =  (7.0, 12.0)
# ir_param_amplitudes["I_range"] = (0.4, 1.0)

# for sample 4
central_wavelength = measured_trace.lam0*1e6
# for sample 3
#central_wavelength = 1.6345

ir_param_amplitudes = {}
#ir_param_amplitudes["linear_phase"] = (-1., 1.)
ir_param_amplitudes["phase_range"] = (0, 2 * np.pi)
ir_param_amplitudes["clambda_range"] = (central_wavelength, central_wavelength)
#ir_param_amplitudes["clambda_range"] = (1.0, 1.6345)
ir_param_amplitudes["pulseduration_range"] = (11.0, 16.0)
ir_param_amplitudes["I_range"] = (0.10, 0.92)

# xuv phase parameters
xuv_pulse = {}
xuv_pulse["N"] = int(2 * 1024)
xuv_pulse["tmax"] = 1600e-18
# this scaled the applied spectral phase
# these values have to be tuned to keep the xuv pulse within the time range
#                                             1st   2nd  3rd   4th   5th order
xuv_pulse["coef_phase_amplitude"] = np.array([0.0, 1.3, 0.15, 0.03, 0.01])
# includes linear
xuv_phase_coefs=5
# phase amplitude
amplitude=20.0

#---------------------------
#--STREAKING TRACE PARAMS---
#---------------------------
Ip_eV = 24.587
Ip = Ip_eV * sc.electron_volt  # joules
Ip = Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.
# sample = 2
delay_values = measured_trace.delay
# delay_values_fs = delay_values * sc.physical_constants['atomic unit of time'][0] * 1e15
delay_values_fs = delay_values * 1e15
K = measured_trace.energy

# threshold scaler for the generated pulses
threshold_scaler = 0.03

threshold_min_index = 100
# threshold_min_index = 50
threshold_max_index = (2*1024) - 100
# threshold_max_index = 1024 - 50







