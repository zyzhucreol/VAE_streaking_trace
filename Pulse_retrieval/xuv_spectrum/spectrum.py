import csv
import pickle
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import scipy.interpolate
import os
import sys
sys.path.append('./Pulse_retrieval')
import phase_parameters.params


def open_data_file(filepath):
    x = []
    y = []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            values = line.rstrip().split(",")
            values = [float(e) for e in values]
            x.append(values[0])
            y.append(values[1])
    return x, y

def interp_measured_data_to_linear(electronvolts_in, intensity_in, plotting=False):
    # convert eV to joules
    joules = np.array(electronvolts_in) * sc.electron_volt  # joules
    hertz = np.array(joules / sc.h)
    Intensity = np.array(intensity_in)

    # define tmat and fmat
    N = phase_parameters.params.xuv_pulse["N"]
    tmax = phase_parameters.params.xuv_pulse["tmax"] # seconds

    dt = 2 * tmax / N
    tmat = dt * np.arange(-N / 2, N / 2, 1)
    df = 1 / (N * dt)
    fmat = df * np.arange(-N / 2, N / 2, 1)

    # get rid of any negative points if there are any
    Intensity[Intensity < 0] = 0

    # set the edges to 0
    Intensity[-1] = 0
    Intensity[0] = 0

    # for plotting later, reference the values which contain the non-zero Intensity
    f_index_min = hertz[0]
    f_index_max = hertz[-1]

    # add zeros at the edges to match the fmat matrix
    hertz = np.insert(hertz, 0, np.min(fmat))
    Intensity = np.insert(Intensity, 0, 0)
    hertz = np.append(hertz, np.max(fmat))
    Intensity = np.append(Intensity, 0)

    # get the carrier frequency
    f0 = hertz[np.argmax(Intensity)]
    # square root the intensity to get electric field amplitude
    Ef = np.sqrt(Intensity)

    # map the spectrum onto linear fmat
    interpolator = scipy.interpolate.interp1d(hertz, Ef, kind='linear')
    Ef_interp = interpolator(fmat)

    # calculate signal in time
    linear_E_t = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef_interp)))

    # set the indexes for cropped input
    indexmin = np.argmin(np.abs(fmat - f_index_min))
    indexmax = np.argmin(np.abs(fmat - f_index_max))

    if plotting:
        plt.figure(1)
        plt.plot(hertz, Intensity, color='red')
        plt.plot(fmat, np.zeros_like(fmat), color='blue')

        plt.figure(2)
        plt.plot(fmat, Ef_interp, label='|E|')
        plt.plot(hertz, Intensity, label='I')
        plt.xlabel('frequency [Hz]')
        plt.legend(loc=1)

        plt.figure(3)
        plt.plot(tmat, np.real(linear_E_t))

        plt.figure(4)
        plt.plot(fmat, Ef_interp, color='red', alpha=0.5)
        plt.plot(fmat[indexmin:indexmax], Ef_interp[indexmin:indexmax], color='red')
        plt.show()

    output = {}
    output["hertz"] = hertz
    output["linear_E_t"] = linear_E_t
    output["tmat"] = tmat
    output["fmat"] = fmat
    output["Ef_interp"] = Ef_interp
    output["indexmin"] = indexmin
    output["indexmax"] = indexmax
    output["f0"] = f0
    output["N"] = N
    output["dt"] = dt
    return output
    # fmat [hz]
    # return hertz, linear_E_t, tmat, fmat, Ef_interp, indexmin, indexmax, f0, N, dt

def retrieve_spectrum(plotting=False):
    electron_volts, intensity = open_data_file(os.path.dirname(__file__)+'/sample4/spectrum4_electron.csv')
    # add the ionization potential to the electron volts
    electron_volts = [e+phase_parameters.params.Ip_eV for e in electron_volts]

    # normalize intensity
    intensity = np.array(intensity)
    intensity = intensity / np.max(intensity)

    electron_interp = interp_measured_data_to_linear(electronvolts_in=electron_volts, intensity_in=intensity, plotting=plotting)

    # open the cross section
    electron_volts_cs, cross_section = open_data_file(os.path.dirname(__file__)+'/sample4/HeliumCrossSection.csv')

    # interpolate the cross section to match the electron spectrum
    interpolator = scipy.interpolate.interp1d(electron_volts_cs, cross_section, kind='linear')
    cross_sec_interp = interpolator(electron_volts)

    # calculate the photon spectrum by diving by the cross section
    photon_spec_I = intensity / cross_sec_interp

    # normalize photon spec intensity
    photon_spec_I = photon_spec_I / np.max(photon_spec_I)

    # interpolate the photon spectrum
    photon_interp = interp_measured_data_to_linear(electronvolts_in=electron_volts, intensity_in=photon_spec_I, plotting=plotting)

    # convert the xuv params to atomic units
    params = {}
    params['tmat'] = electron_interp["tmat"]/sc.physical_constants['atomic unit of time'][0]
    params['fmat'] = electron_interp["fmat"]*sc.physical_constants['atomic unit of time'][0] # 1 / time [a.u.]
    params['Ef'] = electron_interp["Ef_interp"]
    params['Ef_photon'] = photon_interp["Ef_interp"]
    params['indexmin'] = electron_interp["indexmin"]
    params['indexmax'] = electron_interp["indexmax"]
    params['f0'] = electron_interp["f0"]*sc.physical_constants['atomic unit of time'][0] + 0.2
    params['N'] = electron_interp["N"]
    params['dt'] = electron_interp["dt"]/sc.physical_constants['atomic unit of time'][0]

    return params


params = retrieve_spectrum()
tmat = params['tmat']
tmat_as = params['tmat'] * sc.physical_constants['atomic unit of time'][0] * 1e18 # attoseconds
fmat = params['fmat']
fmat_hz = params['fmat'] / sc.physical_constants['atomic unit of time'][0] # hz
fmat_ev = (fmat_hz * sc.h) / sc.electron_volt # electron volts
Ef = params['Ef']
Ef_photon = params['Ef_photon']
indexmin = params['indexmin']
indexmax = params['indexmax']
f0 = params['f0']
N = params['N']
dt = params['dt']
fmat_cropped = fmat[indexmin: indexmax]
fmat_hz_cropped = fmat_hz[indexmin: indexmax]
fmat_ev_cropped = fmat_ev[indexmin: indexmax]

if __name__ == "__main__":

    # spectrum 3
    params = retrieve_spectrum(plotting=True)






