# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:25:02 2021

@author: TweeSr

Note
----

This is the library that can generate the waveform

A(t) = Sigma A_i * cos(omega_i * t + phi_i(t))
phi_i(t) = phi_i(t=0) + omega (c_s/c)*(1/f) Int delta x(t) dt
"""


import numpy as np
import math


#This method utilize the idea of frequency modulation method to generate signal
def freq_mod(freq_spacing,tones_num,MinFreq):
    """
    This method utilize the idea of frequency modulation method to generate signal.

    Note
    -----
    Frequencies are in unit of MHz.
    
    """
    freq_step = freq_spacing*10**6
    #freq_step = 0.5*10**6
    T = 1/freq_step
    min_freq = MinFreq*10**6
    max_freq = (MinFreq+freq_spacing*(tones_num-1))*10**6
    sample_rate = 625*10**6
    initial_phi = -np.pi/2
    num_tones = int( max_freq/freq_step)
    amplitude = np.ones(num_tones)
    def signal(peak, t):
        phase_list = []
        for i in range(1,num_tones+1):
            phi_i = phase_cal(peak_list,initial_phi,i)
            phase_list.append(phi_i)
        A = 0
        for k in range (1,num_tones+1):
            A = A + (peak[k-1]/2)**(1/2)*np.cos(2*np.pi*k*freq_step*t+phase_list[k-1]*np.ones(t.size))
        return A
    def phase_cal(peak,phi_init,n):
        theta = 0
        for i in range(1,n):
            theta = theta + (n-i)*peak[i-1]
        theta_n = phi_init -theta*2*np.pi
        #theta_n = theta_n+np.random.uniform(-0.1, 0.1)
        #print(theta_n)
        return theta_n

    peak_bound = math.floor(min_freq/freq_step)-1
    peak_count = num_tones - peak_bound
    peak_zero = np.zeros(peak_bound)
    peak_one = np.ones(peak_count)
    print(peak_count)
    #peak_one = np.load("C:/Users/TweeSr/Documents/python/optimal_amplitude_y.npy")

    peak_list = np.append(peak_zero,peak_one)
    peak_list = peak_list/np.sum(peak_list)

    sample_points = int(T*sample_rate)
    t_step = np.linspace(0,T,sample_points, endpoint=False)
    #print(phase_list)
    print(peak_one)
    x = signal(peak_list,t_step)

    x_final = x

    for i in range(100):
        x_final = np.append(x_final,x)
    return x_final


def RF_sig(t,array_tones,amp,phase):

    A = 0
    for i in range(array_tones.size):
        A += amp[i]*np.sin(array_tones[i]*2*np.pi*t+phase[i])
    normalized_factor = np.max(A)
    return A/normalized_factor


def sin_sum(parameters, length= 1*10**(-3)):
    """

    Parameters
    ----------
    parameters : dictionary
        for giving the frequencies, the amplitudes and the phases.

    Returns
    -------
    signal : s

    """
    
    freq_array = np.array(parameters["freq"])*10**6
    amp_array = np.array(parameters["amp"])
    phase_array = np.array(parameters["phase"])
    random_phase =np.random.uniform(0,2*np.pi+0.00000000001,freq_array.size)+phase_array
    #print(random_phase)
    t = np.linspace(0,length,int(625*10**6*length),endpoint= False)
    signal = RF_sig(t,freq_array,amp_array,random_phase)

    return signal

if __name__ == "__main__":
    parameters = dict([("freq", np.arange(50,60,1)*10**6),("amp",np.ones(10)),("phase",np.zeros(10))])

    test = sin_sum(parameters)
    np.save("test_signal_x.npy", test)
    print(test)
