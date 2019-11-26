import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import scipy.constants as sc


def find_f0(x, y):
    x = np.array(x)
    y = np.array(y)

    maxvals = []

    for _ in range(3):
        max_index = np.argmax(y)
        maxvals.append(x[max_index])

        x = np.delete(x, max_index)
        y = np.delete(y, max_index)

    maxvals = np.delete(maxvals, np.argmin(np.abs(maxvals)))

    return maxvals[np.argmax(maxvals)]

def find_central_frequency_from_trace(trace, delay, energy, plotting=False):
    # make sure delay is even
    assert len(delay) % 2 == 0

    N = len(delay)
    # print('N: ', N)
    dt = delay[-1] - delay[-2]
    df = 1 / (dt * N)
    freq_even = df * np.arange(-N / 2, N / 2)
    # plot the streaking trace and ft

    trace_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(trace, axes=1), axis=1), axes=1)

    # summation along vertical axis
    integrate = np.sum(np.abs(trace_f), axis=0)

    # find the maximum values
    f0 = find_f0(x=freq_even, y=integrate)  # seconds

    lam0 = sc.c / f0

    if plotting:
        # find central frequency
        _, ax = plt.subplots(3, 1)
        ax[0].pcolormesh(delay, energy, trace, cmap='jet')
        ax[1].pcolormesh(freq_even, energy, np.abs(trace_f), cmap='jet')
        ax[2].plot(freq_even, integrate)

    return f0, lam0

def retrieve_trace(find_f0=False):
    # get trace
    trace = []
    with open(os.path.dirname(__file__)+"/sample4/trace.csv", "r") as file:
        for line in file.readlines():
            line = line.rstrip()
            line = line.split(",")
            line = [float(e) for e in line]
            trace.append(line)

    trace = np.array(trace)

    delay_vals = []
    with open(os.path.dirname(__file__)+"/sample4/delay.csv", "r") as file:
        for line in file.readlines():
            line = line.rstrip()
            delay_vals.append(float(line))
    delay_vals = np.array(delay_vals)


    energy_vals = []
    with open(os.path.dirname(__file__)+"/sample4/energy.csv", "r") as file:
        for line in file.readlines():
            line = line.rstrip()
            energy_vals.append(float(line))
    energy_vals = np.array(energy_vals)


    # remove the last delay step so the delay axis is an even number for fourier transform
    # make sure the dimmensions match
    assert np.shape(energy_vals)[0] == np.shape(trace)[0]
    assert np.shape(delay_vals)[0] == np.shape(trace)[1]

    # if the delay axis is odd, make it even for fourier transforms
    if np.shape(delay_vals)[0] % 2 == 1:
        trace = trace[:, :-1]
        delay_vals = delay_vals[:-1]

    # convert to seconds
    delay_vals = delay_vals * 1e-15

    # normalize trace
#    trace = trace / np.max(trace)

    f0, lam0 = find_central_frequency_from_trace(trace=trace, delay=delay_vals, energy=energy_vals, plotting=False)

    return delay_vals, energy_vals, trace, lam0


def retrieve_trace3(find_f0=False):
    trace = []

    for line in open(os.path.dirname(__file__)+"/sample3/53as_trace.dat", "r"):
        line = line.rstrip()
        line = line.split("\t")
        line = [float(e) for e in line]
        trace.append(line)
    trace = np.transpose(np.array(trace))

    delay_min = -5.47 #fs
    delay_max = 5.44 #fs
    delay = np.linspace(delay_min, delay_max, np.shape(trace)[1])

    e_min = 50
    e_max = 350
    energy = np.linspace(e_min, e_max, np.shape(trace)[0])

    # remove the last delay step so the delay axis is an even number for fourier transform
    trace = trace[:, :-1]
    delay = delay[:-1]

    # convert to seconds
    delay = delay * 1e-15

    # normalize trace
#    trace = trace / np.max(trace)

    if find_f0:
        f0, lam0 = find_central_frequency_from_trace(trace=trace, delay=delay, energy=energy, plotting=True)
        print(f0)  # in seconds
        print(lam0)

    return delay, energy, trace

def retrieve_trace2(find_f0=False):
    filepath = './measured_trace/sample2/MSheet1_1.csv'
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        matrix = np.array(list(reader))

        Energy = matrix[1:, 0].astype('float') # eV
        Delay = matrix[0, 1:].astype('float') # fs
        Values = matrix[1:, 1:].astype('float')

    #print(Delay)
    # print('len(Energy): ', len(Energy))
    # print('Energy: ', Energy)

    # construct frequency axis with even number for fourier transform
    values_even = Values[:, :-1]
    Delay_even = Delay[:-1]
    Delay_even = Delay_even * 1e-15  # convert to seconds

    if find_f0:
        f0, lam0 = find_central_frequency_from_trace(trace=values_even, delay=Delay_even, energy=Energy, plotting=True)
        print(f0)  # in seconds
        print(lam0)

    return Delay_even, Energy, values_even

trace_num = 4


if trace_num == 2:

    delay, energy, trace = retrieve_trace2()

elif trace_num == 3:

    delay, energy, trace = retrieve_trace3()
    
elif trace_num == 4:

    delay, energy, trace, lam0 = retrieve_trace()


if __name__ == "__main__":

    delay, energy, trace = retrieve_trace(find_f0=True)

    plt.figure()
    plt.pcolormesh(delay, energy, trace, cmap="jet")
    plt.show()




