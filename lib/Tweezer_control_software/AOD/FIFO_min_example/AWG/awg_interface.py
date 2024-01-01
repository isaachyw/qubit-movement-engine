# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:26:21 2021

@author: Weijun Yuan
"""

import numpy as np
import matplotlib.pyplot as plt
from AWG.M4i_FIFO_simple_rep_2D import *

def Static2D_mode(in_q):
    x_data = np.zeros(625*10**3)
    y_data = np.zeros(625*10**3)
    # freq_spacing_x = 1
    # freq_spacing_y = 1
    # tones_num_x = 30
    # tones_num_y = 1
    # MinFreq_x = 60
    # MinFreq_y = 60
    # x_data = waveform_gen.freq_mod(freq_spacing_x,tones_num_x,MinFreq_x)
    # y_data = waveform_gen.freq_mod(freq_spacing_y,tones_num_y,MinFreq_y)
    # print(x_data)
    # print(y_data)
    combined_data = [x_data]+[y_data]
    ###combine the data of two channels into one buffer
    precalculated_data = np.column_stack(combined_data).flatten()

    #setup the timer
    #setup the timer
    #define the card
    M4iFIFO = M4i_FIFO_simple_rep_2D(in_q,channelNum=2,sampleRate=625)

    #setup the buffer
    setBuffer = M4iFIFO.setSoftwareBuffer()

    #laod the data
    M4iFIFO.loadData(precalculated_data)

    #setup the card
    M4iFIFO.setupCard()


    #start the card
    M4iFIFO.startCard()

    #stop the card if the interrpution is given by the user
    #stop the card if the interrpution is given by the user

    r = M4iFIFO.stop()

    print("Card has been stopped with error code: ",str(r))
