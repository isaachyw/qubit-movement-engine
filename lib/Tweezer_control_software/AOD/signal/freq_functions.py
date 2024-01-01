"""
The freq function generator for ramping the frequencies.

Note
----

a. This is the library that can generate the integral part of the waveform,

    A(t) = Sigma A_i * cos(omega_i * t + phi_i(t))
    phi_i(t) = phi_i(t=0) + omega (c_s/c)*(1/f) Int delta x(t) dt

    that is, Int delta x(t) dt.

Contributors
------------
Chun-Wei (cl762)
"""

# Std imports
import time
import numpy as np
import matplotlib.pyplot as plt

# Scipy imports
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import trapezoid


def acceleration(t, t_moving, a):
    """
    Constant accleration.

    Parameters
    ----------
    t : float
        The input time data.
    t_moving : float
        The assigned moving time for this step.
    a : float
        The assigned accleration for this step.
    """
    return (1/2) * a * t**2


def decelleration(t, t_moving, a):
    """
    Constant accleration.
    Parameters
    ----------
    t : float
        The input time data.
    t_moving : float
        The assigned moving time for this step.
    a : float
        The assigned accleration for this step.
    """
    return -(1/2) * a * t**2 + a*t*t_moving - (1/4)*a*t_moving**2


def data_stack(base_data, new_data):
    """
    Stacking the next iteration of ramping data to the old ones.
    """
    # 2D case
    try: 
        return np.hstack((base_data, base_data[:, -1].reshape((2, 1)) + new_data))
    # 1D case
    except IndexError:
        return np.hstack((base_data, base_data[-1] + new_data))


def accleration_solver(init_freq, target_freq, t_moving):
    """
    The analytical solution to the accleration of this process.

    init_freq : tuple
        The initial point of the ramping process.
    target_freq : tuple
        The target point of the ramping process.
    t_moving : float
        The assigned moving time.
    """

    return 4*(target_freq.reshape((2, 1)) - init_freq.reshape((2, 1)))/(t_moving **2)


def moving_time_solver(init_freq, target_freq, ave_velocity, trap_spacing):
    """
    To be the optimal time mapping based on initial and target frequencies(positions).
    To be upgraded when we are really trapping atoms.
    Will return 1 just for now.

    Parameters
    ----------
    ave_velocity : float
        In unit of nm/microsec.
    trap_spacing : float
        In unit of micro meter.
    """
    return trap_spacing / (ave_velocity)


def freq_func_constant_ramp(freq_list, ave_velocity, trap_spacing, awg_resolution):
    """
    The constant acceleration ramping freq function itself.
    To be transform into integrals.

    Parameters
    ----------
    freq_list : list
        The atom position we want to scan through.
    awg_resolution : float
        The resolution of the awg card.
    """
    i = 0
    while i < len(freq_list) - 1:
        # Extract position data
        init_freq = np.asarray(freq_list[i]) * 2 * np.pi * 1e06
        target_freq = np.asarray(freq_list[i+1]) * 2 * np.pi * 1e06

        # Compute the optimum moving time and accleration in this iteration
        t_moving = moving_time_solver(init_freq, target_freq, ave_velocity, trap_spacing)
        a = accleration_solver(init_freq, target_freq, t_moving)

        # Generating the time data for this iteration
        # t_data_accleration and t_data_deccleration should be fixed in each iteration
        t_data_accleration, t_spacing = np.linspace(0, t_moving/2, int(awg_resolution * t_moving/2), endpoint=False, retstep=True)
        t_data_deccleration = np.linspace(t_moving/2, t_moving, int(awg_resolution * t_moving/2), endpoint=False)
        
        # Generating the raming data for this iteration
        # For the first iteration, there is no data to stack on.
        if i == 0:
            t_data = np.append(t_data_accleration, t_data_deccleration)
            data = np.hstack((acceleration(t_data_accleration, t_moving, a), decelleration(t_data_deccleration, t_moving, a)))
        # For the i>0 iteration, we will need to stack the data on previous one and add some offsets.
        else:
            t_data = data_stack(base_data = t_data, new_data = np.append(t_data_accleration, t_data_deccleration) + t_spacing)
            #TODO: by doing this, we might waste time on the boundary points, since there are duplicated data points on the connections
            data = data_stack(base_data = data, new_data = np.hstack((acceleration(t_data_accleration, t_moving, a), decelleration(t_data_deccleration, t_moving, a))))
        i+=1
    
    return t_data, data, t_spacing


def moving_tweezer_ramping(freq_list, awg_resolution):
    """
    Taking the frequency function and generate its integral based on interpolation.
    
    Parameters
    ----------
    freq_list : list
        The atom position we want to scan through.
    awg_resolution : float
        The resolution of the awg card.

    Notes
    -----
    For the integration of freq function, there are three possible directions.
    
    1. Use InterpolatedUnivariateSpline to get the freq function and then integral.
    2. Use scipy.integral.cumulative_trapezoid to get the integral
    3. Work on the analytical solution of that integral if neccessary.

    All above will give the correct restult and the benchmarks for ramping 
    between two points are like,

    ramp data cal time = 0.06514325000000554s
    InterpolatedUnivariateSplin cal time = 0.04459370900002568s
    cumulative_trapezoid cal time = 0.004300666999995428s.

    Another thing no notice is that if we are sampling with 625 MHz,
    then even freq function cal time will be forever. Luckily, the known
    tweezer moving time is in micro sec scale, so we don't need to worry about
    this issue. (Just for now)
    """
    # Generate the freq ramping data based on 
    # constant accleration
    #ramp_start = time.perf_counter()
    t_data, data = freq_func_constant_ramp(freq_list, awg_resolution)
    #ramp_end = time.perf_counter()

    # Since the data points might not be a trivial function for one to integrate
    # we use scipy model to help us better integrate the freq function with some 
    # numerical techniques
    
    # Interpolation method
    # interpolate_start = time.perf_counter()
    # interpolated_freq_function = InterpolatedUnivariateSpline(t_data, data, k=1)
    # integral_val = interpolated_freq_function.integral(t_data[0], t_data[-1])
    # interpolate_end = time.perf_counter()

    # Direct integration, will return the integration among all discrete data
    #inter_start = time.perf_counter()
    integral_val = trapezoid(data, t_data)
    # For cumulated data, we use cumulative_trapezoid
    #inter_end = time.perf_counter()

    # print(f'ramp data cal time = {ramp_end - ramp_start}')
    # print(f'interpolation cal time = {interpolate_end - interpolate_start}')
    # print(f'integral cal time = {inter_end - inter_start}')

    return integral_val

## Useful tools
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
    # Test block for freq_functions.py

    # Constant ramp
    print(moving_tweezer_ramping([85, 89, 83, 84, 87], awg_resolution = 625e6))
