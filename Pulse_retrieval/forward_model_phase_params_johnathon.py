import sys
sys.path.append('./Pulse_retrieval')
import tensorflow as tf
import xuv_spectrum.spectrum
import ir_spectrum.ir_spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial, gamma
import scipy.constants as sc
import phase_parameters.params
import measured_trace.get_trace as measured_trace
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1,0';
N_eng = np.size(phase_parameters.params.K) # streak trace width,height
N_tau = np.size(phase_parameters.params.delay_values)

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

def hz_to_au_energy(vector_hz):
    vector_joules = vector_hz * sc.h # joules
    vector_energy_au = vector_joules / sc.physical_constants["atomic unit of energy"][0] # a.u. energy
    return vector_energy_au

def calc_streaking_phase_term(photon_energy, Ip):
    # take absolute value because cant square negative
    # photon_energy = np.abs(photon_energy)

    # --------------------------------------------------------
    # ---- phase accmulated from coulomb potential -----------
    # --------------------------------------------------------
    term1 =  2 - (1j / np.sqrt(2*( np.abs(photon_energy-Ip) )))
    # gamma function
    term1 = gamma(term1)
    # natural log
    term1 = np.log(term1)
    # imaginary part
    term1 = np.imag(term1)

    # --------------------------------------------------------
    # ---- phase accunulated from ir driving laser field -----
    # --------------------------------------------------------
    # approximate the wavelength as the average IR wavelength
    ir_wl_max = phase_parameters.params.ir_param_amplitudes['clambda_range'][0]
    ir_wl_min = phase_parameters.params.ir_param_amplitudes['clambda_range'][1]
    avg_ir_wavelength_um = (ir_wl_min + ir_wl_max) / 2
    Tlaser_sec = ((avg_ir_wavelength_um)*1e-6 / sc.c) # seconds
    # Tlaser_sec = (1.7*1e-6 / sc.c) # seconds
    Tlaser_au = Tlaser_sec / sc.physical_constants["atomic unit of time"][0] # a.u
    # cycle of laser # a.u
    # Tlaser = 1.7 / (0.3 * 24.2 *10**-3)
    x_integral_start = (5/27.2)
    dx = np.max(photon_energy-Ip) / 10000
    term2 = []
    for x_integral_end in (photon_energy-Ip):
        # calculate summation
        x = np.arange(x_integral_start, x_integral_end, dx)
        y = 1/(( 2*x )**( 3/2 ))*(2 - np.log(x * Tlaser_au))
        term2.append(dx * np.sum(y))
    term2 = np.array(term2)

    # accumulated phase from both ir pulse and coulomb potential of atom (hydrogen)
    phi_streak = term1 + term2

    return phi_streak

def ir_from_params(ir_param_values):
  with tf.name_scope('ir_from_params'):
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
        scaled_tf_values[key.split("_")[0]] = parameters[key]["tf_avg"] + ir_param_values[:, i] * parameters[key]["tf_half_range"]

    # convert to SI units
    W = 1
    cm = 1e-2
    um = 1e-6
    fs = 1e-15

    scaled_tf_values_si = {}
    scaled_tf_values_si["I"] = scaled_tf_values["I"] * 1e13 * W / cm ** 2
    scaled_tf_values_si["f0"] = sc.c / (um * scaled_tf_values["clambda"])
    scaled_tf_values_si["t0"] =  scaled_tf_values["pulseduration"] * fs

    # calculate ponderomotive energy in SI units
    Up = (sc.elementary_charge ** 2 * tf.abs(scaled_tf_values_si["I"])) / (2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * scaled_tf_values_si["f0"]) ** 2)

    # convert to AU
    values_au = {}
    values_au["Up"] = Up / sc.physical_constants['atomic unit of energy'][0]
    values_au["f0"] = scaled_tf_values_si["f0"] * sc.physical_constants['atomic unit of time'][0]
    values_au["t0"] = scaled_tf_values_si["t0"] / sc.physical_constants['atomic unit of time'][0]

    # calculate driving amplitude in AU
    E0 = tf.sqrt(4 * values_au["Up"] * (2 * np.pi * values_au["f0"]) ** 2)

    # set up the driving IR field amplitude in AU
    tf_tmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.tmat, dtype=tf.float32), [1, -1])
    # tf_fmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.fmat, dtype=tf.float32), [1, -1])

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

    # inverse fourier transform
    Et_phase = tf_ifft(Ef_phase, shift=int(len(ir_spectrum.ir_spectrum.tmat) / 2), axis=1)

    # crop the phase
    Ef_phase_cropped = Ef_phase[:, ir_spectrum.ir_spectrum.start_index:ir_spectrum.ir_spectrum.end_index]

    E_prop = {}
    E_prop["f"] = Ef_phase
    E_prop["f_cropped"] = Ef_phase_cropped
    E_prop["t"] = Et_phase

    out = {}
    out["scaled_values"] = scaled_tf_values
    out["E_prop"] = E_prop

    return out

def xuv_taylor_to_E(coefficients_in):
  with tf.name_scope('xuv_taylor_to_E'):
    assert int(coefficients_in.shape[1]) == phase_parameters.params.xuv_phase_coefs

    amplitude = phase_parameters.params.amplitude
    coef_amp = phase_parameters.params.xuv_pulse["coef_phase_amplitude"]

    Ef = tf.constant(xuv_spectrum.spectrum.Ef, dtype=tf.complex64)
    Ef = tf.reshape(Ef, [1, -1])
    Ef_photon = tf.constant(xuv_spectrum.spectrum.Ef_photon, dtype=tf.complex64)
    Ef_photon = tf.reshape(Ef_photon, [1, -1])

    fmat_taylor = tf.constant(xuv_spectrum.spectrum.fmat-xuv_spectrum.spectrum.f0, dtype=tf.float32)

    # create factorials
    factorials = tf.constant(factorial(np.array(range(coefficients_in.shape[1]))+1), dtype=tf.float32)
    factorials = tf.reshape(factorials, [1, -1, 1])

    # create exponents
    exponents = tf.constant(np.array(range(coefficients_in.shape[1]))+1, dtype=tf.float32)

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
    scaler_2 = tf.constant(coef_amp.reshape(1,-1,1), dtype=tf.float32)

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
    Ef_photon_prop = Ef_photon * tf.exp(tf.complex(imag=phasecurve, real=tf.zeros_like(phasecurve)))


    # ------------------------
    # additional ir phase term
    # Ep - Ip (a.u) ----------
    # ------------------------

    # Ep (photon energy)
    # Ip
    Ip = phase_parameters.params.Ip # a.u. energy
    # xuv_spectrum.spectrum.fmat # a.u. (frequency)
    E_photon_au = hz_to_au_energy(xuv_spectrum.spectrum.fmat_hz) # atomic units of energy
    # calc_streaking_phase_term(E_photon_au - Ip)
    phi_streak = calc_streaking_phase_term(E_photon_au, Ip)
    streaking_phase_term = phi_streak.astype(np.float32)
    streaking_phase_term_exp = tf.exp(tf.complex(imag=streaking_phase_term, real=tf.zeros_like(streaking_phase_term)))

    # fourier transform for time propagated signal
    Et_prop = tf_ifft(Ef_prop * streaking_phase_term_exp, shift=int(xuv_spectrum.spectrum.N/2), axis=1)
    Et_photon_prop = tf_ifft(Ef_photon_prop, shift=int(xuv_spectrum.spectrum.N/2), axis=1)

    # return the cropped E
    Ef_prop_cropped = Ef_prop[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]
    Ef_photon_prop_cropped = Ef_photon_prop[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]

    # return cropped phase curve
    phasecurve_cropped = phasecurve[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]

    E_prop = {}
    E_prop["streaking_phase_term_exp"] = streaking_phase_term_exp
    E_prop["f"] = Ef_prop
    E_prop["f_cropped"] = Ef_prop_cropped
    E_prop["f_photon_cropped"] = Ef_photon_prop_cropped
    E_prop["t"] = Et_prop
    E_prop["t_photon"] = Et_photon_prop
    E_prop["phasecurve_cropped"] = phasecurve_cropped
    #E_prop["coefs_divided_by_int"] = coefs_divided_by_int

    return E_prop

def streaking_trace(xuv_in, ir_in):
  with tf.name_scope('streak_trace'):
    # define the angle for streaking trace collection
    theta_max = np.pi/2
#    N_theta = 10
#    N_theta = 3
#    angle_in = tf.constant(np.linspace(0, theta_max, N_theta), dtype=tf.float32)
    angle_in=theta_max/2
    Beta_in =  1


    # this is the second version of streaking trace generator which also includes
    # the A^2 term in the integral

    # ionization potential
    Ip = phase_parameters.params.Ip

    # padded xuv in time domain, from xuv_taylor_to_E
    xuv_time_domain = xuv_in["t"][0]


    #------------------------------------------------------------------
    #------ zero pad ir in frequency space to match xuv timestep-------
    #------------------------------------------------------------------
    # calculate N required to match timestep
    N_req = int(1 / (xuv_spectrum.spectrum.dt * ir_spectrum.ir_spectrum.df))
    # this much needs to be padded to each side
    pad_2 = int((N_req - ir_spectrum.ir_spectrum.N) / 2)
    # pad the IR to match dt of xuv
    paddings_ir_2 = tf.constant([[pad_2, pad_2]], dtype=tf.int32)
    padded_ir_2 = tf.pad(ir_in["E_prop"]["f"][0], paddings_ir_2)
    # calculate ir with matching dt in time
    ir_t_matched_dt = tf_ifft(tensor=padded_ir_2, shift=int(N_req / 2))
    # match the scale of the original
    scale_factor = tf.constant(N_req/ ir_spectrum.ir_spectrum.N, dtype=tf.complex64)
    ir_t_matched_dt_scaled = ir_t_matched_dt * scale_factor


    #------------------------------------------------------------------
    # ---------------------integrate ir pulse--------------------------
    #------------------------------------------------------------------
    A_t = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(tf.real(ir_t_matched_dt_scaled))

    # integrate A_L(t)
    flipped1 = tf.reverse(A_t, axis=[0])
    flipped_integral = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1, axis=0)
    A_t_integ_t_phase = tf.reverse(flipped_integral, axis=[0])

    # integrate A_L(t)^2
    flipped1_2 = tf.reverse(A_t**2, axis=[0])
    flipped_integral_2 = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1_2, axis=0)
    A_t_integ_t_phase_2 = tf.reverse(flipped_integral_2, axis=[0])



    # ------------------------------------------------------------------
    # ---------------------make ir t axis-------------------------------
    # ------------------------------------------------------------------
    ir_taxis = xuv_spectrum.spectrum.dt * np.arange(-N_req/2, N_req/2, 1)



    # ------------------------------------------------------------------
    # ---------------------find indexes of tau values-------------------
    # ------------------------------------------------------------------
    center_indexes = []
    delay_vals_au = phase_parameters.params.delay_values/sc.physical_constants['atomic unit of time'][0]
    for delay_value in delay_vals_au:
        index = np.argmin(np.abs(delay_value - ir_taxis))
        center_indexes.append(index)
    center_indexes = np.array(center_indexes)
    rangevals = np.array(range(xuv_spectrum.spectrum.N)) - int((xuv_spectrum.spectrum.N/2))
    delayindexes = center_indexes.reshape(1, -1) + rangevals.reshape(-1, 1)


    # ------------------------------------------------------------------
    # ------------gather values from integrated array-------------------
    # ------------------------------------------------------------------
    ir_values = tf.gather(A_t_integ_t_phase, delayindexes.astype(np.int))
    ir_values = tf.expand_dims(tf.expand_dims(ir_values, axis=0), axis=3)
    # for the squared integral
    ir_values_2 = tf.gather(A_t_integ_t_phase_2, delayindexes.astype(np.int))
    ir_values_2 = tf.expand_dims(tf.expand_dims(ir_values_2, axis=0), axis=3)



    #------------------------------------------------------------------
    #-------------------construct streaking trace----------------------
    #------------------------------------------------------------------
    # convert K to atomic units
    K = phase_parameters.params.K * sc.electron_volt  # joules
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    K = K.reshape(-1, 1, 1, 1)
    p = np.sqrt(2 * K).reshape(-1, 1, 1, 1)
    # theta_max = np.pi # 90 degrees
    # angle_in = np.linspace(0, theta_max, 10)

    spec_angle = tf.reshape(tf.cos(angle_in), [1, 1, 1, -1])
    # convert to tensorflow
    p_tf = tf.constant(p, dtype=tf.float32)

    p_A_t_integ_t_phase3d = spec_angle * p_tf * ir_values + 0.5 * ir_values_2
    ir_phi = tf.exp(tf.complex(imag=(p_A_t_integ_t_phase3d), real=tf.zeros_like(p_A_t_integ_t_phase3d)))
    # add fourier transform term
    e_fft = np.exp(-1j * (K + Ip) * xuv_spectrum.spectrum.tmat.reshape(1, -1, 1, 1))
    e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)
    # add xuv to integrate over
    xuv_time_domain_integrate = tf.reshape(xuv_time_domain, [1, -1, 1, 1])
    # multiply elements together

    # axes:
    # (301, 2048, 98)
    # (K, xuv_time, tau_delay)
    # --> expand dimmension for angle -->
    # (301, 2048, 98, ??)
    # (K, xuv_time, tau_delay, angle)
    # angular distribution term calculated from equation
    angular_distribution = 1 + (Beta_in / 2)  * (3 * (tf.cos(angle_in))**2 - 1)
    angular_distribution = tf.reshape(angular_distribution, [1, 1, 1, -1])
    angular_distribution = tf.complex(imag=tf.zeros_like(angular_distribution), real=angular_distribution)
#    with tf.device('/gpu:1'):
    product = angular_distribution * xuv_time_domain_integrate * ir_phi * e_fft_tf
    # integrate over the xuv time
    integration = tf.constant(xuv_spectrum.spectrum.dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)
    # absolute square the matrix
    image_not_scaled = tf.square(tf.abs(integration))
    image_not_scaled = image_not_scaled * tf.reshape(tf.sin(angle_in), [1, 1, -1])

    # integrate along the theta axis
    dtheta = angle_in[1] - angle_in[0] if np.size(angle_in)>1 else 0.16
#    dtheta=0.16
    theta_integration = dtheta * tf.reduce_sum(image_not_scaled, axis=2)

    scaled = theta_integration - tf.reduce_min(theta_integration)
    image = scaled / tf.reduce_max(scaled)

    return image
    
def apply_A_concat(X):
    xuv_phase=tf.expand_dims(X[0:5],0)
    ir_coeff=tf.expand_dims(X[5:9],0)
    xuv_complex=xuv_taylor_to_E(xuv_phase)
    ir_complex=ir_from_params(ir_coeff)
    img_concat=streaking_trace(xuv_complex,ir_complex)
    return img_concat

def A_fun(X_batch):
    with tf.name_scope('forward_A'):
        fn=lambda X: apply_A_concat(X)
        image_batch=tf.map_fn(fn,X_batch,name='AX')
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
#        xuv_coefs_in: np.array([[0.5, 0.0, 0.0, 0.0, 0.0],[0.5,0,0,1,0.0],
#                                [0.5,0.5,0.5,0.5,0.5],[1.0,1.0,1.0,1.0,1.0]]),
        #                           1st, 2nd, 3rd,4th, 5th 
        #                             -1 < x < 1
        xuv_coefs_in: np.concatenate((np.array([[0.,-0.5,-0.6,-0.2,-0.5]]),0.5*np.ones((19,5))),0),
                    
#        ir_values_in: np.array([[1.0, 0.0, 0.0, 0.0],[0.5,0,-1,0],
#                                [0.5,0.5,0.5,0.5],[1.0,1.0,1.0,1.0]])
        #               CEP, central wavelength, Pulse Duration, Intensity 
        #                             -1 < x < 1
        ir_values_in: np.concatenate((np.array([[-0.1,0,-0.5,0]]),0.5*np.ones((19,4))),0)
        }

    sess = tf.InteractiveSession()
    xuv_pulse = sess.run(xuv_E_prop['f_cropped'][0], feed_dict=feed_dict)
    xuv_pulse_t = sess.run(xuv_E_prop['t'][0], feed_dict=feed_dict)
    ir_pulse = sess.run(ir_E_prop['E_prop']['f_cropped'][0], feed_dict=feed_dict)
    # generated trace
    trace = sess.run(image[0,:,:,0], feed_dict=feed_dict)
    plt.figure(1)
    plt.plot(np.abs(xuv_pulse), color="blue")
    plt.plot(np.angle(xuv_pulse)/(2*np.pi), color="red")
    plt.figure(2)
    plt.plot(xuv_spectrum.spectrum.tmat,np.abs(xuv_pulse_t), color="blue")
    plt.figure(3)
    plt.plot(np.abs(ir_pulse), color="blue")
    plt.plot(np.angle(ir_pulse)/(2*np.pi), color="red")
    plt.figure(4)
    plt.pcolormesh(measured_trace.delay, measured_trace.energy,trace)
    # measured trace
    plt.figure(5)
    plt.pcolormesh(measured_trace.delay, measured_trace.energy, measured_trace.trace)

#%%
#sess=tf.Session()
#xuv_cropped_f_in=tf.constant(sess.run(xuv_E_prop['f_cropped'][0],feed_dict))
#ir_cropped_f_in=tf.constant(sess.run(ir_E_prop['f_cropped'][0],feed_dict))