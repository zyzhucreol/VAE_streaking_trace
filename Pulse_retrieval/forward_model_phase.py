import sys
sys.path.append('./Pulse_retrieval')
import tensorflow as tf
import xuv_spectrum.spectrum
import ir_spectrum.ir_spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
import scipy.constants as sc
import phase_parameters.params
import measured_trace.get_trace as measured_trace
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1';
N_eng,N_tau = 256,38 # streak trace width,height

def tf_ifft(tensor, shift, axis=0):
    shifted = tf.manip.roll(tensor, shift=shift, axis=axis)
    # fft
    time_domain_not_shifted = tf.ifft(shifted)
    # shift again
    time_domain = tf.manip.roll(time_domain_not_shifted, shift=shift, axis=axis)

    return time_domain

def tf_fft(tensor, shift, axis=0):
    shifted = tf.manip.roll(tensor, shift=shift, axis=axis)
    # fft
    freq_domain_not_shifted = tf.fft(shifted)
    # shift again
    freq_domain = tf.manip.roll(freq_domain_not_shifted, shift=shift, axis=axis)

    return freq_domain

def ir_from_params(ir_param_values):
    with tf.name_scope('IR_from_params'):
        amplitudes = phase_parameters.params.ir_param_amplitudes
    
        # construct tf nodes for middle and half range of inputs
        parameters = {}
        for key in ["phase_range", "clambda_range", "pulseduration_range", "I_range"]:
            parameters[key] = {}
    
            # get the middle and half the range of the variables
            parameters[key]["avg"] = (amplitudes[key][0] + amplitudes[key][1])/2
            parameters[key]["half_range"] = (amplitudes[key][1] - amplitudes[key][0]) / 2
    
            # create tensorflow constants
            parameters[key]["tf_avg"] = tf.constant(parameters[key]["avg"], dtype=tf.float32)
            parameters[key]["tf_half_range"] = tf.constant(parameters[key]["half_range"], dtype=tf.float32)
    
    
        # construct param values from normalized input
        scaled_tf_values = {}
    
        for i, key in enumerate(["phase_range", "clambda_range", "pulseduration_range", "I_range"]):
            scaled_tf_values[key.split("_")[0]] = parameters[key]["tf_avg"] + ir_param_values[:,i] * parameters[key]["tf_half_range"]
    
        # convert to SI units
        W = 1
        cm = 1e-2
        um = 1e-6
        fs = 1e-15
    
        scaled_tf_values_si = {}
        scaled_tf_values_si["I"] = scaled_tf_values["I"] * 1e13 * W / cm ** 2
        scaled_tf_values_si["f0"] = sc.c / (um * scaled_tf_values["clambda"])
        scaled_tf_values_si["t0"] =  scaled_tf_values["pulseduration"] * fs
    #    scaled_tf_values_si["tau0"] =  scaled_tf_values["linear"] * fs
    
        # calculate ponderomotive energy in SI units
        Up = (sc.elementary_charge ** 2 * tf.abs(scaled_tf_values_si["I"])) / (2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * scaled_tf_values_si["f0"]) ** 2)
    
        # convert to AU
        values_au = {}
        values_au["Up"] = Up / sc.physical_constants['atomic unit of energy'][0]
        values_au["f0"] = scaled_tf_values_si["f0"] * sc.physical_constants['atomic unit of time'][0]
        values_au["t0"] = scaled_tf_values_si["t0"] / sc.physical_constants['atomic unit of time'][0]
    #    values_au["tau0"] = scaled_tf_values_si["tau0"] / sc.physical_constants['atomic unit of time'][0]
    
        # calculate driving amplitude in AU
        E0 = tf.sqrt(4 * values_au["Up"] * (2 * np.pi * values_au["f0"]) ** 2)
    
        # set up the driving IR field amplitude in AU
        tf_tmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.tmat, dtype=tf.float32), [1, -1])
        # tf_fmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.fmat, dtype=tf.float32), [1, -1])
    #    tf_fmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.fmat, dtype=tf.float32), [1, -1])
    
        # slow oscilating envelope
        Et_slow_osc = tf.reshape(E0, [-1, 1]) * tf.exp(-2*np.log(2) * (tf_tmat / tf.reshape(values_au["t0"], [-1, 1]))**2)
    
        # fast oscilating envelope
        phase = 2 * np.pi * tf.reshape(values_au["f0"], [-1, 1]) * tf_tmat
        Et_fast_osc = tf.exp(tf.complex(imag=phase, real=tf.zeros_like(phase)))
    
        # Pulse before phase applied
        Et = tf.complex(real=Et_slow_osc, imag=tf.zeros_like(Et_slow_osc)) * Et_fast_osc
    
        # Fourier transform
        Ef = tf_fft(Et, shift=int(len(ir_spectrum.ir_spectrum.tmat)/2), axis=1)
    
        # apply phase angle
        phase = tf.reshape(scaled_tf_values["phase"], [-1, 1])
        Ef_phase = Ef * tf.exp(tf.complex(imag=phase, real=tf.zeros_like(phase)))
    #    Ef_phase = Ef_phase * tf.exp(tf.complex(imag=tf_fmat*scaled_tf_values_si["tau0"], real=tf.zeros_like(phase)))
    
        # inverse fourier transform
        Et_phase = tf_ifft(Ef_phase, shift=int(len(ir_spectrum.ir_spectrum.tmat) / 2), axis=1)
    
        # crop the phase
        Ef_phase_cropped = Ef_phase[:, ir_spectrum.ir_spectrum.start_index:ir_spectrum.ir_spectrum.end_index]
    
        E_prop = {}
        E_prop["f"] = Ef_phase
        E_prop["f_cropped"] = Ef_phase_cropped
        E_prop["t"] = Et_phase
    
        return E_prop["f_cropped"]

def xuv_taylor_to_E(coefficients_in):
#    assert int(coefficients_in.shape[1]) == phase_parameters.params.xuv_phase_coefs

    amplitude = phase_parameters.params.amplitude

    Ef = tf.constant(xuv_spectrum.spectrum.Ef, dtype=tf.complex64)
    Ef = tf.reshape(Ef, [1, -1])

    fmat_taylor = tf.constant(xuv_spectrum.spectrum.fmat-xuv_spectrum.spectrum.f0, dtype=tf.float32)

    # create factorials
    factorials = tf.constant(factorial(np.array(range(coefficients_in.shape[1]))), dtype=tf.float32)
    factorials = tf.reshape(factorials, [1, -1, 1])

    # create exponents
    exponents = tf.constant(np.array(range(coefficients_in.shape[1])), dtype=tf.float32)

    # reshape the taylor fmat
    fmat_taylor = tf.reshape(fmat_taylor, [1, 1, -1])

    # reshape the exponential matrix
    exp_mat = tf.reshape(exponents, [1, -1, 1])

    # raise the fmat to the exponential power
    exp_mat_fmat = tf.pow(fmat_taylor, exp_mat)

    # scale the coefficients
    amplitude_mat = tf.constant(amplitude, dtype=tf.float32)
    amplitude_mat = tf.reshape(amplitude_mat, [1, -1, 1])

    # amplitude scales with exponent
    amplitude_scaler = tf.pow(amplitude_mat, exp_mat)

    # additional scaler
    # these are arbitrary numbers that were found to keep the field in the time window


    # for sample 2
    # scaler_2 = tf.constant(np.array([1.0, 1.0, 0.2, 0.06, 0.04]).reshape(1,-1,1), dtype=tf.float32)

    # for sample 3
    scaler_2 = tf.constant(np.array([1.0, 1.0, 1.3, 0.15, 0.03, 0.01]).reshape(1,-1,1), dtype=tf.float32)



    # reshape the coef values and scale them
    coef_values = tf.reshape(coefficients_in, [tf.shape(coefficients_in)[0], -1, 1]) * amplitude_scaler * scaler_2

    # divide by the factorials
    coef_div_fact = tf.divide(coef_values, factorials)

    # multiply by the fmat
    taylor_coefs_mat = coef_div_fact * exp_mat_fmat

    # this is the phase angle, summed along the taylor terms
    phasecurve = tf.reduce_sum(taylor_coefs_mat, axis=1)

    # apply the phase angle to Ef
    Ef_prop = Ef * tf.exp(tf.complex(imag=phasecurve, real=tf.zeros_like(phasecurve)))

    # fourier transform for time propagated signal
    Et_prop = tf_ifft(Ef_prop, shift=int(xuv_spectrum.spectrum.N/2), axis=1)

    # return the cropped E
    Ef_prop_cropped = Ef_prop[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]

    # return cropped phase curve
    phasecurve_cropped = phasecurve[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]

    E_prop = {}
    E_prop["f"] = Ef_prop
    E_prop["f_cropped"] = Ef_prop_cropped
    E_prop["t"] = Et_prop
    E_prop["phasecurve_cropped"] = phasecurve_cropped
    #E_prop["coefs_divided_by_int"] = coefs_divided_by_int

    return E_prop["f_cropped"]

def streaking_trace(xuv_cropped_f_in, ir_cropped_f_in):
    # ionization potential
    Ip = phase_parameters.params.Ip

    #-----------------------------------------------------------------
    # zero pad the spectrum of ir and xuv input to match the full original f matrices
    #-----------------------------------------------------------------
    N_xuv=(xuv_spectrum.spectrum.indexmax-xuv_spectrum.spectrum.indexmin)
    xuv_time_domain = tf_ifft(tensor=xuv_cropped_f_in, shift=int(N_xuv/ 2))
    # add linear phase to match the cropped fft to the padded fft
    xuv_time_domain = xuv_time_domain*tf.exp(tf.complex(imag=-2*np.pi*44/N_xuv*np.arange(-N_xuv/2,N_xuv/2,dtype='float32'),real=tf.zeros_like(xuv_cropped_f_in,dtype='float32')))
    
    #------------------------------------------------------------------
    #------ zero pad ir in frequency space to match xuv timestep-------
    #------------------------------------------------------------------
    # calculate N required to match timestep
    N_req = int(1 / (xuv_spectrum.spectrum.dt *xuv_spectrum.spectrum.N/N_xuv * ir_spectrum.ir_spectrum.df))
    # this additional zeros need to be padded to each side
    pad_2 = int((N_req - ir_spectrum.ir_spectrum.N) / 2)

    # ------------------------------------------------------------------
    # ---------------------make ir and xuv t axis-----------------------
    # ------------------------------------------------------------------
#    ir_taxis = xuv_spectrum.spectrum.dt * np.arange(-N_req/2, N_req/2, 1)
    
    ir_taxis = xuv_spectrum.spectrum.dt*xuv_spectrum.spectrum.N/N_xuv * np.arange(-N_req/2, N_req/2, 1)
    xuv_tmat = xuv_spectrum.spectrum.dt*xuv_spectrum.spectrum.N/N_xuv * np.arange(-N_xuv/2, N_xuv/2, 1)
    xuv_tmat = np.float32(np.reshape(xuv_tmat,(-1, 1)))
    
    # ------------------------------------------------------------------
    # ---------------------find indexes of tau values-------------------
    # ------------------------------------------------------------------
    center_indexes = []
    delay_vals_au = phase_parameters.params.delay_values/sc.physical_constants['atomic unit of time'][0]
    for delay_value in delay_vals_au:
        index = np.argmin(np.abs(delay_value - ir_taxis))
        center_indexes.append(index)
    center_indexes = np.array(center_indexes)
    rangevals = np.array(range(N_xuv)) - int((N_xuv/2))
    delayindexes = center_indexes.reshape(1, -1) + rangevals.reshape(-1, 1)

    # ------------------------------------------------------------------
    # ----------generate required ir time-domain integrand--------------
    # ------------------------------------------------------------------
    min_ir_t_domain_ind=np.int(np.min(delayindexes))
    max_ir_t_domain_ind=np.int(np.max(delayindexes))
    # Fourier transform of ir spectrum onto required time-domain integrand
    w=pad_2+np.arange(ir_spectrum.ir_spectrum.start_index,ir_spectrum.ir_spectrum.end_index)-N_req/2
    t=2*np.pi*(np.arange(min_ir_t_domain_ind,max_ir_t_domain_ind+1)-N_req/2)/N_req
    wt=np.matmul(np.reshape(w,[-1,1]),np.reshape(t,[1,-1]))
    cos_mat=tf.constant(np.cos(wt),dtype=tf.float32)
    sin_mat=tf.constant(np.sin(wt),dtype=tf.float32)
    scale_factor = np.float32(1.0/ ir_spectrum.ir_spectrum.N)
    ir_t_matched_dt_scaled=(tf.matmul(tf.real(tf.reshape(ir_cropped_f_in,[1,-1])),cos_mat)-tf.matmul(tf.imag(tf.reshape(ir_cropped_f_in,[1,-1])),sin_mat)) * scale_factor
    ir_t_matched_dt_scaled=tf.reshape(ir_t_matched_dt_scaled,[-1])
    
    #------------------------------------------------------------------
    # ---------------------integrate ir pulse--------------------------
    #------------------------------------------------------------------
    A_t = tf.constant(-1.0 * xuv_spectrum.spectrum.dt*xuv_spectrum.spectrum.N/N_xuv, dtype=tf.float32) * tf.cumsum(tf.real(ir_t_matched_dt_scaled))

    # integrate A_L(t)
    flipped1 = tf.reverse(A_t, axis=[0])
    flipped_integral = tf.constant(-1.0 * xuv_spectrum.spectrum.dt*xuv_spectrum.spectrum.N/N_xuv, dtype=tf.float32) * tf.cumsum(flipped1, axis=0)
    A_t_integ_t_phase = tf.reverse(flipped_integral, axis=[0])

    # integrate A_L(t)^2
    flipped1_2 = tf.reverse(A_t**2, axis=[0])
    flipped_integral_2 = tf.constant(-1.0 * xuv_spectrum.spectrum.dt*xuv_spectrum.spectrum.N/N_xuv, dtype=tf.float32) * tf.cumsum(flipped1_2, axis=0)
    A_t_integ_t_phase_2 = tf.reverse(flipped_integral_2, axis=[0])
    
    # ------------------------------------------------------------------
    # ------------gather values from integrated array-------------------
    # ------------------------------------------------------------------
    ir_values = tf.gather(A_t_integ_t_phase, delayindexes.astype(np.int)-min_ir_t_domain_ind)
    # for the squared integral
    ir_values_2 = tf.gather(A_t_integ_t_phase_2, delayindexes.astype(np.int)-min_ir_t_domain_ind)
    
    #------------------------------------------------------------------
    #-------------------construct streaking trace----------------------
    #------------------------------------------------------------------
#    xuv_tmat=np.float32(np.reshape(xuv_spectrum.spectrum.tmat,(-1, 1)))
    # convert K to atomic units
    
    K = phase_parameters.params.K * sc.electron_volt  # joules
    K = K[0:256]
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    K = np.reshape(K,(-1,1,1))
    p = np.sqrt(2 * K).reshape(-1, 1, 1)
    # convert to tensorflow
    p_tf = tf.constant(p, dtype=tf.float32)
    # 3d ir mat
    p_A_t_integ_t_phase3d = p_tf * ir_values + 0.5 * ir_values_2
    ir_phi = tf.exp(tf.complex(imag=(p_A_t_integ_t_phase3d), real=tf.zeros_like(p_A_t_integ_t_phase3d)))
#    print(tf.dimension_value(ir_phi))
    # add fourier transform term
    e_fft = np.exp(-1j * (K + Ip) * xuv_tmat.reshape(1, -1, 1))
    e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)
    # add xuv to integrate over
    xuv_time_domain_integrate = tf.reshape(xuv_time_domain, [1, -1, 1])
    # multiply elements together
    integration= tf.reduce_sum(xuv_time_domain_integrate * ir_phi * e_fft_tf,axis=1)
    # integrate over the xuv time
    integration = tf.constant(xuv_spectrum.spectrum.dt*xuv_spectrum.spectrum.N/N_xuv, dtype=tf.complex64) * integration
    # absolute square the matrix
    image_not_scaled = tf.square(tf.abs(integration))
    
    scaled = image_not_scaled - tf.reduce_min(image_not_scaled)
    image = scaled / tf.reduce_max(scaled)

    return image
    
def apply_A_concat(X):
    xuv_real=X[0:200]
    xuv_imag=X[200:400]
    ir_real=X[400:420]
    ir_imag=X[420:440]
    xuv_complex=tf.complex(xuv_real,xuv_imag)
    ir_complex=tf.complex(ir_real,ir_imag)
    img_concat=streaking_trace(xuv_complex,ir_complex)
    return img_concat

def A_fun(X_batch):
    with tf.name_scope('forward_A'):
        xuv_phase=X_batch[:,0:6]
        ir_coeff=X_batch[:,6:10]
        xuv_complex=xuv_taylor_to_E(xuv_phase)
        ir_complex=ir_from_params(ir_coeff)
        xuv_real=tf.real(xuv_complex)
        xuv_imag=tf.imag(xuv_complex)
        ir_real=tf.real(ir_complex)
        ir_imag=tf.imag(ir_complex)
        X_concat=tf.concat((xuv_real,xuv_imag,ir_real,ir_imag),axis=1)
        fn=lambda X: apply_A_concat(X)
        image_batch=tf.map_fn(fn,X_concat,name='AX')
        image_batch=tf.reshape(image_batch,[-1,N_eng,N_tau,1])
    return image_batch

if __name__ == "__main__":
    # construct placeholders
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None,
                phase_parameters.params.xuv_phase_coefs])
    xuv_E_prop = xuv_taylor_to_E(xuv_coefs_in)
    ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
    ir_E_prop = ir_from_params(ir_values_in)
    image = A_fun(tf.concat((xuv_coefs_in,ir_values_in),axis=1))

    feed_dict = {
#        xuv_coefs_in: np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],[1,0.5,1.0,0,0,0.0],
#                                [0.5,0.5,0.5,0.5,0.5,0.5],[1.0,1.0,1.0,1.0,1.0,1.0]]),
        xuv_coefs_in: np.ones((20,6)),
        #                           1st, 2nd, 3rd,4th, 5th 
        #                             -1 < x < 1
                    
#        ir_values_in: np.array([[1.0, 0.0, 0.0, 0.0],[0.5,0,-1,0],
#                                [0.5,0.5,0.5,0.5],[1.0,1.0,1.0,1.0]])
        ir_values_in: np.ones((20,4))
        #               CEP, central wavelength, Pulse Duration, Intensity 
        #                             -1 < x < 1
        }

    sess = tf.InteractiveSession()
    xuv_pulse = sess.run(xuv_E_prop[0], feed_dict=feed_dict)
    ir_pulse = sess.run(ir_E_prop[0], feed_dict=feed_dict)
    xuv_pulse_tf = tf.convert_to_tensor(xuv_pulse)
    ir_pulse_tf = tf.convert_to_tensor(ir_pulse)
    # generated trace
    trace = streaking_trace(xuv_pulse_tf,ir_pulse_tf)
    trace = sess.run(trace)
    plt.figure(1)
    plt.plot(np.abs(xuv_pulse), color="blue")
    plt.plot(np.angle(xuv_pulse)/(2*np.pi), color="red")
    plt.figure(2)
    plt.plot(np.abs(ir_pulse), color="blue")
    plt.plot(np.angle(ir_pulse)/(2*np.pi), color="red")
    plt.figure(3)
    plt.pcolormesh(trace)
    # measured trace
    plt.figure(4)
    plt.pcolormesh(measured_trace.delay, measured_trace.energy, measured_trace.trace)

#%%
#sess=tf.Session()
#xuv_cropped_f_in=tf.constant(sess.run(xuv_E_prop['f_cropped'][0],feed_dict))
#ir_cropped_f_in=tf.constant(sess.run(ir_E_prop['f_cropped'][0],feed_dict))