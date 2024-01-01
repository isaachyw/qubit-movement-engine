"""
Main function for operating FIFO, again
"""
import time
import multiprocessing as mp

from AWG import awg_interface, waveform_gen

def static_tweezers():
    """Create static tweezer array."""

    dim = 10
    freq_spacing = 1
    center_freq = 82
    start_index = -1*(dim//2)
    freq_list = []
    for i in range(dim):
        freq_list.append(center_freq+(start_index+i)*freq_spacing)
    freq_array = freq_list
    amp = len(freq_array)*[1]
    phase = [0]*len(freq_array)

    a_data = dict([("freq",freq_array),("amp",amp),("phase",phase)])

    new_wave_x = waveform_gen.sin_sum(a_data)
    new_wave_y = waveform_gen.sin_sum(a_data)

    wave_form = dict([("x_data",new_wave_x),("y_data",new_wave_y)])

    return wave_form



def main():
    q = mp.Queue()
    p = mp.Process(target=awg_interface.Static2D_mode, args=(q,))
    p.start()

    print("Generating new waveforms")
    while True:
        wave_form = static_tweezers()
        q.put(wave_form)
        time.sleep(10)

if __name__ == "__main__":
    main()