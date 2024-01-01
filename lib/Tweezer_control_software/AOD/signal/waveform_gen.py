"""
Note
----

This is the library that can generate the waveform

A(t) = Sigma A_i * cos(omega_i * t + phi_i(t))
phi_i(t) = phi_i(t=0) + omega (c_s/c)*(1/f) Int delta x(t) dt

Contributors
------------
Chun-Wei Liu (cl3762)
"""

# Std imports
import os
import time
import h5py

# Numerical imports
import numpy as np
import math
from scipy.integrate import cumulative_trapezoid # integral discrete func
import networkx as nx

# Freq function imports
from lib.TrapGeometry.geometry import GeometeryGenerator
from lib.Tweezer_control_software.AOD.signal.freq_functions import freq_func_constant_ramp
from utils import check_folder

# Custom imports
from tqdm import tqdm


def precal_alignment_wave_gen():
    """
    The signal for automatic calibration static and dynamic traps
    """
    raise NotImplementedError

def cal_parallel_tweezer_wave_gen(freq_path, awg_resolution, grab_drop_time, ave_velocity, trap_spacing, save_folder = "./Experiments/"):
    """
    Generating the rf signal that drive multipe tweezers to the assigned location.

    Parameters
    ----------
    freq_path : list
        The path data that assign tweezer locations.
    lattice : str
        The classification of the lattice.
    save_folder : str
        Where the precalculated data file located.

    Notes
    -----
    The formation of freq_path is different than single tweezer. 
    Should be in the form of freq_path[row][move][trap]

    For FIFO mode in practice, maybe we should send new rf signal to the card every time we 
    complete the calculation for each row, instead of calculating the entire rearranging process.
    
    But for a template, we just send the complete rearranging signals to the card.
    Possible control sequence:
        Reservior moving -> target moving layers ...

    # TODO: This is a template
    """
    
    ## Some general parameters
    ## AWG card related
    
    ## Trap depth
    Amp_static = 1 # micro kelvin
    Amp_mt = 0.1 * Amp_static
    
    ## Trap related
    # Grap/Drop time
    t_grab = grab_drop_time * 1e-6 # sec
    t_drop = grab_drop_time * 1e-6 # sec
    t_static = 1e3 * 1e-6 # sec

    signal = np.zeros((2, 1))
    
    ## Calculate the moving data for each row
    for row in freq_path:
        move_signal = 0
        for move in row:
            freq_list = [move[0], move[-1]]
            phase_num = np.random.uniform(0, 2*np.pi + 1e-11)
            phase = np.array([phase_num, phase_num]).reshape(2, 1)
            ## Each move signal should be the same length, the same time data.
            ## TODO: Need to compute the moving time
            grab_signal = grab_atom_wave_gen(freq_list[0], Amp_mt, phase, t_grab, awg_resolution)
            drop_signal = drop_atom_wave_gen(freq_list[-1], Amp_mt, phase, t_drop, awg_resolution)
            moving_signal= move_atom_wave_gen(freq_list, Amp_mt, phase, ave_velocity, trap_spacing, awg_resolution = 625e06)[0]
            
            ## Stack all three types of signals that form one move signal
            move_signal += moving_signal# np.hstack((grab_signal, moving_signal, drop_signal))
        
        ## Append the the row signal to the total signal
        signal = np.hstack((signal, move_signal[:, 1:])) ## Get rid of np.zeros((2, 1))

    ## Might need to normalize
    signal[0, :] = signal[0, :]/np.max(signal[0, :])
    signal[1, :] = signal[1, :]/np.max(signal[1, :])

    return signal[:, 1:] ## Get rid of np.zeros((2, 1))

def precal_tweezer_wave_gen(freq_graph,
                             lattice, 
                             awg_resolution, 
                             grab_drop_time, 
                             ave_velocity, 
                             trap_spacing, 
                             save_folder = "./Experiments/"):
    """
    Generating the rf signal that drive tweezer to the assigned location.

    Parameters
    ----------
    freq_graph : networkx graph object
        The lattice graph object in rf frequency basis. Contains the information of 
        trap locations (nodes) and interactions (edges).
    lattice : str
        The classification of the lattice.
    save_folder : str
        Where the precalculated data file located.

    Notes
    -----
    a. Process of moving tweezers (MT):      
        1. Grab, ramp up MT amplitude, should be 10 times deeper than static trap.
        2. Move the tweezer
        3. Put, ramp down MT amplitude.

    b. Process of precalculating moving data. (Based on the current alignment of static traps (metasurface) and AODs.)
        1. Calculate the grab and drop signals, time complexity is O(2N).
        2. Calculate the moving data, can be up to [eN! - 1]

    c. Required information for defining the signal:
        1. Tone, the frequency that can map to atom position
        2. Phase, the phase term can prevent higher order expansions of the waveform (trap) interference or overlapping. 
            Naively randomly assigned.

    d. Remind the keys for accessing the hdf5 data.

    e. The data tree of the hdf5 file,
        -------- grab - signal - np.array
                      - index - int
            ---- drop - signal - np.array
                      - index - int
            ---- move - signal - np.array
                      - index - int

    f. Frequencies are transformed into angular frequencies w = 2 * pi * f and unit as MHz.
    """
    file_time = time.strftime("%Y%m%d_%H%M%S")
    precal_folder = os.path.join(save_folder, 'AWG/precal_signals/')
    check_folder(precal_folder)
    file_path = os.path.join(precal_folder, f'MovingTweezerSignal_{lattice}_{file_time}.hdf5')
    data_file = h5py.File(file_path,'w')
    grab_group = data_file.create_group("grab")
    drop_group = data_file.create_group("drop")
    move_group = data_file.create_group("move")
    static_group = data_file.create_group("static")
    

    ## Create signal and board memory segment index group
    grab_signal_group  = grab_group.create_group("signal")
    grab_seg_index_group  = grab_group.create_group("index")
    drop_signal_group  = drop_group.create_group("signal")
    drop_seg_index_group  = drop_group.create_group("index")
    move_signal_group  = move_group.create_group("signal")
    move_seg_index_group  = move_group.create_group("index")
    static_signal_group  = static_group.create_group("signal")
    static_seg_index_group  = static_group.create_group("index")


    ## Get the geometry (Extract edge and vertices)
    trap_location = list(freq_graph.nodes)
    
    ## Hoppings
    hoppings = []
    for e in freq_graph.edges:
        hoppings.append(e)
        hoppings.append(tuple(reversed(e)))

    print(f"Total trap numbers: {len(trap_location)}")
    print(f"Total allowed hopping in this graph: {len(hoppings)}")

    ## construct phase data
    ## Each frequency in each direction should have the same phase term

    phase_dict = {}
    phase_data = np.random.uniform(0, 2*np.pi + 1e-11, np.unique(np.array(trap_location)).size)
    for f, freq in enumerate(np.unique(np.array(trap_location))):
        phase_dict[freq] = phase_data[f]

    ## Some general parameters
    ## AWG card related
    #awg_resolution = 625*1e6 # in Hz, corresponding to the time sampling rate
    
    ## Trap depth
    ## The moving trap depth should be 2 ~ 10 times larger than one static trap;
    ## Light intensity I is porportional to sqrt(P), the square root of acoustic power
    
    Amp_static = 1
    Amp_mt = Amp_static / 4 # For DEMO: Amp_mt = 0.1 * Amp_static
    
    
    ## Process 1: grab/drop atom
    ## Trap related
    ## Grap/Drop time
    t_grab = grab_drop_time # sec
    t_drop = grab_drop_time # sec
    ## TODO: Should make it go through entire period
    t_static = 1e3 * 1e-6 # sec, this is a hard code number for best static signal performance

    ## Generating signals
    i = 0
    print("Generating static signals...")
    static_signal_group.create_dataset("static", data = static_wave_gen(trap_location, Amp_static, phase_dict, t_static, awg_resolution))
    static_seg_index_group.create_dataset("static", data = int(i))
    print("Done\n")
    i += 1

    print("Generating grab and drop signals...")
    for trap in tqdm(trap_location):
        grab_signal_group.create_dataset(str(trap), data = grab_atom_wave_gen(trap, Amp_mt, phase(phase_dict, trap), t_grab, awg_resolution))
        grab_seg_index_group.create_dataset(str(trap), data = int(i))
        i += 1

        drop_signal_group.create_dataset(str(trap), data = drop_atom_wave_gen(trap, Amp_mt, phase(phase_dict, trap), t_drop, awg_resolution))
        drop_seg_index_group.create_dataset(str(trap), data = int(i))
        i += 1
    print("Done\n")


    print("Generating moving signals...")
    start_time = time.time()
    for hopping in tqdm(hoppings):
        move_signal_group.create_dataset(str(hopping), data = move_atom_wave_gen(hopping, Amp_mt, phase(phase_dict, trap_location[trap_location.index(hopping[0])])[0], ave_velocity, trap_spacing, awg_resolution))
        move_seg_index_group.create_dataset(str(hopping), data = int(i))
        i += 1
    end_time = time.time()
    print("Done\n")
    print(f"Moving calculation time: {end_time - start_time}s, Average: {(end_time - start_time)/i * 1e3} ms")

    # close the data file
    data_file.close()

    ## for test
    return file_time

def phase(phase_dict, trap):
    """
    Obtaining phase data in the shape of N channels or coordinates. Phase data will be column data after this transformation.

    Note
    ----
    Be sure if the output is correct when updating the algorithm behind "move_atom_wave_gen", "grab_atom_wave_gen", "drop_atom_wave_gen",
    and "static_wave_gen"
    """
    return np.array([phase_dict[trap[0]], phase_dict[trap[1]]]).reshape(2, 1)

def move_atom_wave_gen(freq_list, Amp_mt, phase, ave_velocity, trap_spacing, awg_resolution = 625e06):
    """
    Generating the rf signal that move atom from one location to another.
    
    Parameters
    ----------
    freq_list : list
        The atom position (rf frequencies) that we want to ramp through.
        Format : [[wx, wy]_i]
    Amp_mt : float
        The amplitude of the moving tweezer. Typically 10 time larger than the static traps.
    phase : float
        The phase data of the initial atom position.
    awg_resolution : float
        The resolution of the AWG card. Typically 625 MHz.
    """
    # First get the ramping freq function (discrete data points)
    t_data, data, t_spacing = freq_func_constant_ramp(freq_list, ave_velocity, trap_spacing, awg_resolution)

    
    ## Calculate the rf signals for moving tweezers
    ## cumulative_trapezoid will return the "cumulated" integral of time t
    ## And then normalize it.
    signal = Amp_mt * np.cos(np.array(freq_list[0]).reshape((2, 1)) * 2 * np.pi * 1e06 * t_data + np.repeat(phase, t_data.shape[0], axis=1) + cumulative_trapezoid2D(data, t_data, initial=0))
    ## Normalize data
    # signal[0, :] = signal[0, :]/np.max(signal[0, :])
    # signal[1, :] = signal[1, :]/np.max(signal[1, :])

    return signal, data, t_data, t_spacing

def grab_atom_wave_gen(init_freq, Amp_mt, phase, t_grab, awg_resolution):
    """
    Generating the rf signal that grab atom at the assigned location.

    Parameters
    ----------
    init_freq : float
        The atom location that we want to migrate.
        Format : [wx, wy]
    Amp_mt : float
        Trap depth of the moving tweezer.
    t_grab : float
        The time that we want to spend on grabbing sigle atom.
    """

    # Define data
    if int(awg_resolution * t_grab) < 1:
        raise ValueError('Zero sample point obtained with current awg_resolution * t_grab value (less then 1)')

    t_grab_data = np.linspace(0, t_grab, int(awg_resolution * t_grab))
    Amp_ramp_data = np.linspace(0, Amp_mt, int(awg_resolution * t_grab))
    
    # Returning (signal_x, signal_y)
    ## Compute the signals
    signal = Amp_ramp_data * np.cos(np.array(init_freq).reshape((2, 1)) * 2 * np.pi * 1e06 * t_grab_data + np.repeat(phase, t_grab_data.shape[0], axis=1))
    
    ## Normalize data
    # signal[0, :] = signal[0, :]/abs(np.max(signal[0, :]))
    # signal[1, :] = signal[1, :]/abs(np.max(signal[1, :]))

    return signal

def drop_atom_wave_gen(end_freq, Amp_mt, phase, t_drop, awg_resolution):
    """
    Generating the rf signal that grab atom at the assigned location.

    Parameters
    ----------
    end_freq : float
        The atom location that we want to migrate to.
    Amp_mt : float
        Trap depth of the moving tweezer.
    t_grab : float
        The time that we want to spend on grabbing sigle atom.
    """

    # Define data
    if int(awg_resolution * t_drop) < 1:
        raise ValueError('Zero sample point obtained with current awg_resolution * t_grab value (less then 1)')

    t_drop_data = np.linspace(0, t_drop, int(awg_resolution * t_drop))
    Amp_ramp_data = np.linspace(Amp_mt, 0, int(awg_resolution * t_drop))

    ## Compute the signals
    signal = Amp_ramp_data * np.cos(np.array(end_freq).reshape((2, 1)) * 2 * np.pi * 1e06 * t_drop_data + np.repeat(phase, t_drop_data.shape[0], axis=1))

    ## Normalize data
    # signal[0, :] = signal[0, :]/abs(np.max(signal[0, :]))
    # signal[1, :] = signal[1, :]/abs(np.max(signal[1, :]))

    return signal
    
def position_to_freq_mapping(trap_location_EMCCD, calibration_data):
    """
    Match atom locations to AOD frequencies after EMCCD&AOD calibration.

    Parameters
    ----------
    trap_location_EMCCD : list
        The target trap and reservior trap locations from EMCCD pixels
    calibration_data : TBD
        The calibration data generated during calibration process.
    
    """
    # TODO: create a mapping
    freq_AOD = [[[85, 85], [86, 86]]]
    return freq_AOD

def cumulative_trapezoid2D(data, t_data, initial = 0):
    """
    The 2D version of scipy cumulative_trapezoid integration.

    Note
    ----
    The initial argument will give the set the initial response to x[0] to be a certain value.
    Making the output data the same length as t_data.
    """
    return np.vstack((cumulative_trapezoid(data[0], t_data, initial = initial), cumulative_trapezoid(data[1], t_data, initial = initial)))

def static_wave_gen(trap_location, Amp_static, phase_dict, t_static, awg_resolution):
    """
    The rf signal of the static lattice.

    Note
    ----
    Need to work with the data from geometry.py
    ## TODO: Need to make the signal go back to the t=0 at t=t_end
    """

    ## Signal time span
    t_data = np.linspace(0, t_static, int(awg_resolution * t_static), endpoint= False)

    ## Calculate the signal
    data = 0
    for freq in tqdm(trap_location, disable=True):
        data += Amp_static * np.sin(np.array(freq).reshape((2, 1)) * 2 * np.pi * 1e06 * t_data + np.repeat(phase(phase_dict, freq), t_data.shape[0], axis=1))
    
    data[0, :] = data[0, :]/abs(np.max(data[0, :]))
    data[1, :] = data[1, :]/abs(np.max(data[1, :]))

    return data

def main():
    """
    The test block for waveform_gen.
    """
    ## Define graph
    print('Defining the graph')
    freq_graph = nx.Graph()
    freq_graph.add_node(tuple([85, 85]))
    freq_graph.add_node(tuple([86, 85]))
    freq_graph.add_node(tuple([87, 84]))
    freq_graph.add_node(tuple([83, 86]))
    freq_graph.add_edge(u_of_edge = tuple([85, 85]), v_of_edge = tuple([86, 85]), weight = 5)
    print(f'Done\n')
    #precal_tweezer_wave_gen(freq_graph, lattice = 'square', save_folder = '.')
    print(static_wave_gen(freq_graph, t_static = 1 * 1e-6, awg_resolution = 625 * 1e6))


if __name__ == "__main__":
    main()
