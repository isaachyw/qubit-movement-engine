#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Templates form Atomize.
https://github.com/Anatoly1010/Atomize

"""
import os
import sys
import random
###AWG
#sys.path.append('/home/anatoly/AWG/spcm_examples/python')
sys.path.append('/home/anatoly/awg_files/python')
#sys.path.append('C:/Users/User/Desktop/Examples/python')
from math import sin, pi, exp, log2
from itertools import groupby, chain
from copy import deepcopy
import numpy as np
import atomize.device_modules.config.config_utils as cutil
import atomize.general_modules.general_functions as general

from pyspcm import *
from spcm_tools import *

#### Inizialization
# setting path to *.ini file
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, 'config','Spectrum_M4I_6631_X8_config.ini')

# configuration data
#config = cutil.read_conf_util(path_config_file)
specific_parameters = cutil.read_specific_parameters(path_config_file)

# Channel assignments
#ch0 = specific_parameters['ch0'] # TRIGGER

timebase_dict = {'ms': 1000000, 'us': 1000, 'ns': 1, }
channel_dict = {'CH0': 0, 'CH1': 1, }
function_dict = {'SINE': 0, 'GAUSS': 1, 'SINC': 2, 'BLANK': 3, }

# Limits and Ranges (depends on the exact model):
#clock = float(specific_parameters['clock'])
max_pulse_length = int(float(specific_parameters['max_pulse_length'])) # in ns
min_pulse_length = int(float(specific_parameters['min_pulse_length'])) # in ns
max_freq = int(float(specific_parameters['max_freq'])) # in MHz
min_freq = int(float(specific_parameters['min_freq'])) # in MHz
phase_shift_ch1_seq_mode = int(float(specific_parameters['ch1_phase_shift'])) # in radians

# Delays and restrictions
# MaxDACValue corresponds to the amplitude of the output signal; MaxDACValue - Amplitude and so on
# lMaxDACValue = int32 (0)
# spcm_dwGetParam_i32 (hCard, SPC_MIINST_MAXADCVALUE, byref(lMaxDACValue))
# lMaxDACValue.value = lMaxDACValue.value - 1
maxCAD = 32767 # MaxCADValue of the AWG card - 1
amplitude_max = 2500 # mV
amplitude_min = 80 # mV
sample_rate_max = 1250 # MHz
sample_rate_min = 50 # MHz
sample_ref_clock_max = 1000 # MHz
sample_ref_clock_min = 10 # MHz
loop_max = 100000
delay_max = 8589934560
delay_min = 0 

# Test run parameters
# These values are returned by the modules in the test run 
if len(sys.argv) > 1:
    test_flag = sys.argv[1]
else:
    test_flag = 'None'

test_amplitude = 600
test_channel = 1
test_sample_rate = '1250 MHz'
test_clock_mode = 'Internal'
test_ref_clock = 100
test_card_mode = 'Single'
test_trigger_ch = 'External'
test_trigger_mode = 'Positive'
test_loop = 0
test_delay = 0
test_channel = 'CH0'
test_amplitude = '600 mV'
test_num_segments = 1

class Spectrum_M4I_6631_X8:
    def __init__(self):
        if test_flag != 'test':

            # Collect all parameters for AWG settings
            self.sample_rate = 1250 # MHz
            self.clock_mode = 1 # 1 is Internal; 32 is External
            self.reference_clock = 100 # MHz
            self.card_mode = 32768 # 32768 is Single; 512 is Multi; 262144 is Sequence
            self.trigger_ch = 2 # 1 is Software; 2 is External
            self.trigger_mode = 1 # 1 is Positive; 2 is Negative; 8 is High; 10 is Low
            self.loop = 0 # 0 is infinity
            self.delay = 0 # in sample rate; step is 32; rounded
            self.channel = 3 # 1 is CH0; 2 is CH1; 3 is CH0 + CH1
            self.enable_out_0 = 1 # 1 means ON; 0 means OFF
            self.enable_out_1 = 1 # 1 means ON; 0 means OFF
            self.amplitude_0 = 600 # amlitude for CH0 in mV
            self.amplitude_1 = 533 # amlitude for CH0 in mV
            self.num_segments = 1 # number of segments for 'Multi mode'
            self.memsize = 64 # full card memory
            self.buffer_size = 2*self.memsize # two bytes per sample
            self.segment_memsize = 32 # segment card memory
            self.single_joined = 0
            # sequence mode
            self.sequence_mode = 0
            self.full_buffer = 0
            self.full_buffer_pointer = 0
            self.sequence_segments = 2
            self.sequence_loop = 1
            self.qwBufferSize = uint64 (2 *2)  # buffer size
            self.sequence_mode_count = 0
            # pulse settings
            self.reset_count = 0
            self.shift_count = 0
            self.increment_count = 0
            self.setting_change_count = 0
            self.pulse_array = []
            self.pulse_array_init = []
            self.pulse_name_array = []
            self.pulse_ch0_array = []
            self.pulse_ch1_array = []

            # state counter
            self.state = 0

        elif test_flag == 'test':
            # Collect all parameters for AWG settings
            self.sample_rate = 1250 
            self.clock_mode = 1
            self.reference_clock = 100
            self.card_mode = 32768
            self.trigger_ch = 2
            self.trigger_mode = 1
            self.loop = 0
            self.delay = 0
            self.channel = 3
            self.enable_out_0 = 1
            self.enable_out_1 = 1
            self.amplitude_0 = 600
            self.amplitude_1 = 533
            self.num_segments = 1
            self.memsize = 64 # full card memory
            self.buffer_size = 2*self.memsize # two bytes per sample
            self.segment_memsize = 32 # segment card memory
            self.single_joined = 0
            # sequence mode
            self.sequence_mode = 0
            self.full_buffer = 0
            self.full_buffer_pointer = 0
            self.sequence_segments = 2
            self.sequence_loop = 1
            self.qwBufferSize = uint64 (2 *2)  # buffer size
            self.sequence_mode_count = 0
            # pulse settings
            self.reset_count = 0
            self.shift_count = 0
            self.increment_count = 0
            self.setting_change_count = 0
            self.pulse_array = []
            self.pulse_array_init = []
            self.pulse_name_array = []
            self.pulse_ch0_array = []
            self.pulse_ch1_array = []

            # state counter
            self.state = 0

    # Module functions
    def awg_name(self):
        answer = 'Spectrum M4I.6631-X8'
        return answer

    def awg_setup(self):
        """
        Write settings to the AWG card. No argument; No output
        Everything except the buffer will be write to the AWG card

        This function should be called after all functions that change settings are called
        """
        if test_flag != 'test':
            
            if self.state == 0:
                # open card
                self.hCard = spcm_hOpen ( create_string_buffer (b'/dev/spcm0') )
                self.state = 1
                if self.hCard == None:
                    general.message("No card found...")
                    sys.exit()
            else:
                pass

            # general parameters of the card; internal/external clock
            if self.clock_mode == 1:
                spcm_dwSetParam_i64 (self.hCard, SPC_SAMPLERATE, MEGA(self.sample_rate))
            elif self.clock_mode == 32:
                spcm_dwSetParam_i32 (self.hCard, SPC_CLOCKMODE, self.clock_mode)
                spcm_dwSetParam_i64 (self.hCard, SPC_REFERENCECLOCK, MEGA(self.reference_clock))
                spcm_dwSetParam_i64 (self.hCard, SPC_SAMPLERATE, MEGA(self.sample_rate))

            # change card mode and memory
            if self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 0:
                #self.buf = self.define_buffer_single()[0]
                spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, self.card_mode)
                spcm_dwSetParam_i32(self.hCard, SPC_MEMSIZE, self.memsize)
            elif self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 1:
                #self.buf = self.define_buffer_single_joined()[0]
                spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, self.card_mode)
                spcm_dwSetParam_i32(self.hCard, SPC_MEMSIZE, self.memsize)
            elif self.card_mode == 512 and self.sequence_mode == 0:
                #self.buf = self.define_buffer_multi()[0]
                spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, self.card_mode)
                spcm_dwSetParam_i32(self.hCard, SPC_MEMSIZE, self.memsize)
                spcm_dwSetParam_i32(self.hCard, SPC_SEGMENTSIZE, self.segment_memsize)
            elif self.sequence_mode == 1:
                #self.full_buffer
                # 262144 is SPC_REP_STD_SEQUENCE 262144
                spcm_dwSetParam_i32( self.hCard, SPC_CARDMODE, SPC_REP_STD_SEQUENCE )
                spcm_dwSetParam_i64( self.hCard, SPC_SEQMODE_MAXSEGMENTS, self.sequence_segments )
                general.message( self.sequence_segments )
                spcm_dwSetParam_i32( self.hCard, SPC_SEQMODE_STARTSTEP, 0 ) # Step#0 is the first step after card start

            
            # trigger
            spcm_dwSetParam_i32(self.hCard, SPC_TRIG_TERM, 1) # 50 Ohm trigger load
            spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ORMASK, self.trigger_ch) # software / external
            if self.trigger_ch == 2:
                spcm_dwSetParam_i32(self.hCard, SPC_TRIG_EXT0_MODE, self.trigger_mode)
            
            # loop
            spcm_dwSetParam_i32(self.hCard, SPC_LOOPS, self.loop)

            # trigger delay
            spcm_dwSetParam_i32( self.hCard, SPC_TRIG_DELAY, int(self.delay) )

            # set the output channels
            spcm_dwSetParam_i32 (self.hCard, SPC_CHENABLE, self.channel)
            spcm_dwSetParam_i32 (self.hCard, SPC_ENABLEOUT0, self.enable_out_0)
            spcm_dwSetParam_i32 (self.hCard, SPC_ENABLEOUT1, self.enable_out_1)
            spcm_dwSetParam_i32 (self.hCard, SPC_AMP0, int32 (self.amplitude_0))
            spcm_dwSetParam_i32 (self.hCard, SPC_AMP1, int32 (self.amplitude_1))

            # define the memory size / max amplitude
            #llMemSamples = int64 (self.memsize)
            #lBytesPerSample = int32(0)
            #spcm_dwGetParam_i32 (hCard, SPC_MIINST_BYTESPERSAMPLE,  byref(lBytesPerSample))
            #lSetChannels = int32 (0)
            #spcm_dwGetParam_i32 (hCard, SPC_CHCOUNT, byref (lSetChannels))

            # MaxDACValue corresponds to the amplitude of the output signal; MaxDACValue - Amplitude and so on
            lMaxDACValue = int32 (0)
            spcm_dwGetParam_i32 (self.hCard, SPC_MIINST_MAXADCVALUE, byref(lMaxDACValue))
            lMaxDACValue.value = lMaxDACValue.value - 1

            if lMaxDACValue.value == maxCAD:
                pass
            else:
                general.message('maxCAD value does not equal to lMaxDACValue.value')
                sys.exit()

            if self.sequence_mode == 0:
                spcm_dwSetParam_i64 (self.hCard, SPC_M2CMD, M2CMD_CARD_WRITESETUP)
            else:
                # due to some reason in Sequence mode M2CMD_CARD_WRITESETUP does not work correctly:
                # Call: (SPC_M2CMD, M2CMD_CARD_WRITESETUP) -> the setup isn't valid"
                pass

        elif test_flag == 'test':
            # to run several important checks
            if self.reset_count == 0 or self.shift_count == 1 or self.increment_count == 1 or self.setting_change_count == 1 or self.sequence_mode_count == 1:
                if self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 0:
                    self.buf = self.define_buffer_single()[0]
                elif self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 1:
                    self.buf = self.define_buffer_single_joined()[0]
                elif self.card_mode == 512 and self.sequence_mode == 0:
                    self.buf = self.define_buffer_multi()[0]
            else:
                pass

    def awg_update(self):
        """
        Start AWG card. No argument; No output
        Default settings:
        Sample clock is 1250 MHz; Clock mode is 'Internal'; Reference clock is 100 MHz; Card mode is 'Single';
        Trigger channel is 'External'; Trigger mode is 'Positive'; Loop is infinity; Trigger delay is 0;
        Enabled channels is CH0 and CH1; Amplitude of CH0 is '600 mV'; Amplitude of CH1 is '533 mV';
        Number of segments is 1; Card memory size is 64 samples;
        """
        if test_flag != 'test':

            if self.reset_count == 0 or self.shift_count == 1 or self.increment_count == 1 or self.setting_change_count == 1 or self.sequence_mode_count == 1:

                # change card mode and memory
                if self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 0:
                    self.buf = self.define_buffer_single()[0]
                    spcm_dwSetParam_i32(self.hCard, SPC_MEMSIZE, self.memsize)
                elif self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 1:
                    #start_time = time.time()
                    self.buf = self.define_buffer_single_joined()[0]
                    #general.message('BUFFER TIME: ' + str( time.time() - start_time ))
                    spcm_dwSetParam_i32(self.hCard, SPC_MEMSIZE, self.memsize )
                elif self.card_mode == 512 and self.sequence_mode == 0:
                    self.buf = self.define_buffer_multi()[0]
                    spcm_dwSetParam_i32(self.hCard, SPC_MEMSIZE, self.memsize)
                    spcm_dwSetParam_i32(self.hCard, SPC_SEGMENTSIZE, self.segment_memsize)
                elif self.sequence_mode == 1:
                    pass

                if self.sequence_mode == 0:
                    spcm_dwSetParam_i64 (self.hCard, SPC_M2CMD, M2CMD_CARD_WRITESETUP)
                else:
                    # due to some reason in Sequence mode M2CMD_CARD_WRITESETUP does not work correctly:
                    # Call: (SPC_M2CMD, M2CMD_CARD_WRITESETUP) -> the setup isn't valid"
                    pass

                # define the buffer
                #pnBuffer = c_void_p()
                #qwBufferSize = uint64 (self.buffer_size)  # buffer size
                #pvBuffer = pvAllocMemPageAligned (qwBufferSize.value)
                #pnBuffer = cast (pvBuffer, ptr16)

                # fill the buffer
                if self.card_mode == 32768 and self.sequence_mode == 0:
                    # convert numpy array to integer
                    #for index, element in enumerate(buf):
                    #    pnBuffer[index] = int(lMaxDACValue.value * element)
                    #    #pnBuffer = (lMaxDACValue.value * self.define_buffer_single()).astype('int64')

                    # we define the buffer for transfer and start the DMA transfer
                    #sys.stdout.write("Starting the DMA transfer and waiting until data is in board memory\n")
                    # spcm_dwDefTransfer_i64 (device, buffer_type, direction, event (0=and of transfer), data, offset, buffer length)
                    spcm_dwDefTransfer_i64 (self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32 (0), self.buf, uint64 (0), self.qwBufferSize.value)
                    # transfer
                    spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
                    #general.message("AWG buffer has been transferred to board memory")

                elif self.card_mode == 512 and self.sequence_mode == 0:
                    #for index, element in enumerate(buf):
                    #    pnBuffer[index] = int(lMaxDACValue.value * element)
                    #    #pnBuffer = (lMaxDACValue.value * self.define_buffer_multi()).astype('int64')

                    # we define the buffer for transfer and start the DMA transfer
                    #sys.stdout.write("Starting the DMA transfer and waiting until data is in board memory\n")
                    # spcm_dwDefTransfer_i64 (device, buffer_type, direction, event (0=and of transfer), data, offset, buffer length)
                    spcm_dwDefTransfer_i64 (self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32 (0), self.buf, uint64 (0), self.qwBufferSize.value)
                    # transfer
                    spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
                    #general.message("AWG buffer has been transferred to board memory")

                elif self.sequence_mode == 1:
                    if self.channel == 1 or self.channel == 2:
                        seg_memory = int(self.qwBufferSize.value / 2)
                    elif self.channel == 3:
                        seg_memory = int(self.qwBufferSize.value / 4)

                    for index, element in enumerate(self.full_buffer_pointer):
                        # Setting up the data memory and transfer data
                        spcm_dwSetParam_i32 (self.hCard, SPC_SEQMODE_WRITESEGMENT, index) # set current configuration switch to segment
                        ###
                        ###
                        #seg_memory should be rounded to 32, which means repetition rate should be divisible by 32
                        ###
                        ###
                        spcm_dwSetParam_i64 (self.hCard, SPC_SEQMODE_SEGMENTSIZE, seg_memory ) # define size of current segment

                        # data transfer #self.full_buffer_pointer[index]
                        spcm_dwDefTransfer_i64 (self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32 (0), element, uint64 (0), self.qwBufferSize.value)
                        spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)

                        #general.message("AWG buffer has been transferred to board memory")

                        # self.vWriteStepEntry (hCard, dwStepIndex, dwStepNextIndex, dwSegmentIndex, dwLoops, dwFlags)
                        if index <= self.sequence_segments - 2:
                            self.write_seg_memory (self.hCard,  index,  index + 1, index, self.sequence_loop,  0) #0
                        elif index == self.sequence_segments - 1:
                            self.write_seg_memory (self.hCard,  index, 0, index, self.sequence_loop,  2147483648)

                        # SPCSEQ_ENDLOOPONTRIG = 1073741824
                        # Feature flag that marks the step to conditionally change to the next step on a trigger condition. The occurrence
                        # of a trigger event is repeatedly checked each time the defined loops for the current segment have been
                        # replayed. A temporary valid trigger condition will be stored until evaluation at the end of the step.
                        # SPCSEQ_END = 2147483648
                        # Feature flag that marks the current step to be the last in the sequence. The card is stopped at the end of this seg-
                        # ment after the loop counter has reached his end.

                # We'll start and wait until the card has finished or until a timeout occurs
                dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITTRIGGER)
                # test or error message
                #general.message(dwError)
                               
                # clean up
                #spcm_vClose (hCard)

                self.reset_count = 1
                self.shift_count = 0
                self.increment_count = 0
                self.setting_change_count = 0
                self.sequence_mode_count == 0

            else:
                pass

        elif test_flag == 'test':
            # to run several important checks
            if self.reset_count == 0 or self.shift_count == 1 or self.increment_count == 1 or self.setting_change_count == 1 or self.sequence_mode_count == 1:
                if self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 0:
                    self.buf = self.define_buffer_single()[0]
                elif self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 1:
                    self.buf = self.define_buffer_single_joined()[0]
                elif self.card_mode == 512 and self.sequence_mode == 0:
                    self.buf = self.define_buffer_multi()[0]

                self.reset_count = 1
                self.shift_count = 0
                self.increment_count = 0
                self.setting_change_count = 0
                self.sequence_mode_count == 0

            else:
                pass

    def awg_close(self):
        """
        Close AWG card. No argument; No output
        """
        if test_flag != 'test':
            # clean up
            spcm_vClose (self.hCard)

        elif test_flag == 'test':
            pass

    def awg_stop(self):
        """
        Stop AWG card. No argument; No output
        """
        if test_flag != 'test':

            # open card
            #hCard = spcm_hOpen( create_string_buffer (b'/dev/spcm0') )
            #if hCard == None:
            #    general.message("No card found...")
            #    sys.exit()
            spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
            #general.message('AWG card stopped')
            # for correct work awg_update() is called without length or start arguments changed
            self.reset_count = 0

        elif test_flag == 'test':
            pass

    def awg_pulse(self, name = 'P0', channel = 'CH0', func = 'SINE', frequency = '200 MHz', phase = 0,\
     delta_phase = 0, length = '16 ns', sigma = '16 ns', increment = '0 ns', start = '0 ns', delta_start = '0 ns'):
        """
        A function for awg pulse creation;
        The possible arguments:
        NAME, CHANNEL (CHO, CH1), FUNC (SINE, GAUSS, SINC, BLANK), FREQUENCY (1 - 280 MHz),
        PHASE (in rad), DELTA_PHASE (in rad), 
        LENGTH (in ns, us; should be longer than sigma; minimun is 10 ns; maximum is 1900 ns), 
        SIGMA (sigma for gauss; sinc (length = 32 ns, sigma = 16 ns means +-2pi); sine for uniformity )
        INCREMENT (in ns, us, ms; for incrementing both sigma and length)
        START (in ns, us, ms; for joined pulses in 'Single mode')
        DELTA_START (in ns, us, ms; for joined pulses in 'Single mode')

        Buffer according to arguments will be filled after
        """
        if test_flag != 'test':
            pulse = {'name': name, 'channel': channel, 'function': func, 'frequency': frequency, 'phase' : phase,\
             'delta_phase': delta_phase, 'length': length, 'sigma': sigma, 'increment': increment, 'start': start,\
              'delta_start': delta_start }

            self.pulse_array.append( pulse )
            # for saving the initial pulse_array without increments
            # deepcopy helps to create a TRULY NEW array and not a link to the object
            self.pulse_array_init = deepcopy( self.pulse_array )
            # pulse_name array
            self.pulse_name_array.append( pulse['name'] )

            # For Single/Multi mode checking
            # Only one pulse per channel for Single
            # Number of segments equals to number of pulses for each channel
            if channel == 'CH0':
                self.pulse_ch0_array.append( pulse['name'] )
            elif channel == 'CH1':
                self.pulse_ch1_array.append( pulse['name'] )
            
        elif test_flag == 'test':
            pulse = {'name': name, 'channel': channel, 'function': func, 'frequency': frequency, 'phase' : phase,\
             'delta_phase' : delta_phase, 'length': length, 'sigma': sigma, 'increment': increment, 'start': start,\
              'delta_start': delta_start }

            # Checks
            # two equal names
            temp_name = str(name)
            set_from_list = set(self.pulse_name_array)
            if temp_name in set_from_list:
                assert (1 == 2), 'Two pulses have the same name. Please, rename'

            self.pulse_name_array.append( pulse['name'] )

            # channels
            temp_ch = str(channel)
            assert (temp_ch in channel_dict), 'Incorrect channel. Only CH0 or CH1 are available'

            # for Single/Multi mode checking
            if channel == 'CH0':
                self.pulse_ch0_array.append( pulse['name'] )
            elif channel == 'CH1':
                self.pulse_ch1_array.append( pulse['name'] )

            # Function type
            temp_func = str(func)
            assert (temp_func in function_dict), 'Incorrect pulse type. Only SINE, GAUSS, SINC, and BLANK pulses are available'

            # Frequency
            temp_freq = frequency.split(" ")
            coef = temp_freq[1]
            p_freq = float(temp_freq[0])
            assert (coef == 'MHz'), 'Incorrect frequency dimension. Only MHz is possible'
            assert(p_freq >= min_freq), 'Frequency is lower than minimum available (' + str(min_freq) +' MHz)'
            assert(p_freq < max_freq), 'Frequency is longer than minimum available (' + str(max_freq) +' MHz)'

            # length
            temp_length = length.split(" ")
            if temp_length[1] in timebase_dict:
                coef = timebase_dict[temp_length[1]]
                p_length = coef*float(temp_length[0])
                assert(p_length >= min_pulse_length), 'Pulse is shorter than minimum available length (' + str(min_pulse_length) +' ns)'
                assert(p_length < max_pulse_length), 'Pulse is longer than maximum available length (' + str(max_pulse_length) +' ns)'
            else:
                assert( 1 == 2 ), 'Incorrect time dimension (ms, us, ns)'

            # sigma
            temp_sigma = sigma.split(" ")
            if temp_sigma[1] in timebase_dict:
                coef = timebase_dict[temp_sigma[1]]
                p_sigma = coef*float(temp_sigma[0])
                assert(p_sigma >= min_pulse_length), 'Sigma is shorter than minimum available length (' + str(min_pulse_length) +' ns)'
                assert(p_sigma < max_pulse_length), 'Sigma is longer than maximum available length (' + str(max_pulse_length) +' ns)'
            else:
                assert( 1 == 2 ), 'Incorrect time dimension (ms, us, ns)'

            # length should be longer than sigma
            assert( p_length >= p_sigma ), 'Pulse length should be longer or equal to sigma'

            # increment
            temp_increment = increment.split(" ")
            if temp_increment[1] in timebase_dict:
                coef = timebase_dict[temp_increment[1]]
                p_increment = coef*float(temp_increment[0])
                assert (p_increment >= 0 and p_increment < max_pulse_length), \
                'Length and sigma increment is longer than maximum available length or negative'
            else:
                assert( 1 == 2 ), 'Incorrect time dimension (ms, us, ns)'

            # start
            temp_start = start.split(" ")
            if temp_start[1] in timebase_dict:
                coef = timebase_dict[temp_start[1]]
                p_start = coef*float(temp_start[0])
                assert(p_start >= 0), 'Pulse start should be a positive number'
                assert(p_start % 2 == 0), 'Pulse start should be divisible by 2'                
            else:
                assert( 1 == 2 ), 'Incorrect time dimension (ms, us, ns)'

            # delta_start
            temp_delta_start = delta_start.split(" ")
            if temp_delta_start[1] in timebase_dict:
                coef = timebase_dict[temp_delta_start[1]]
                p_delta_start = coef*float(temp_delta_start[0])
                assert(p_delta_start >= 0), 'Pulse delta start should be a positive number'
                assert(p_delta_start % 2 == 0), 'Pulse delta start should be divisible by 2'
            else:
                assert( 1 == 2 ), 'Incorrect time dimension (ms, us, ns)'

            self.pulse_array.append( pulse )
            self.pulse_array_init = deepcopy(self.pulse_array)

    def awg_redefine_delta_start(self, *, name, delta_start):
        """
        A function for redefining delta_start of the specified pulse for Single Joined mode.
        awg_redefine_delta_start(name = 'P0', delta_start = '10 ns') changes delta_start of the 'P0' pulse to 10 ns.
        The main purpose of the function is non-uniform sampling.

        def func(*, name1, name2): defines a function without default values of key arguments
        """

        if test_flag != 'test':
            i = 0

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:
                    self.pulse_array[i]['delta_start'] = str(delta_start)
                    self.shift_count = 1
                else:
                    pass

                i += 1

        elif test_flag == 'test':
            i = 0
            assert( name in self.pulse_name_array ), 'Pulse with the specified name is not defined'

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:

                    # checks
                    temp_delta_start = delta_start.split(" ")
                    if temp_delta_start[1] in timebase_dict:
                        coef = timebase_dict[temp_delta_start[1]]
                        p_delta_start = coef*float(temp_delta_start[0])
                        assert(p_delta_start % 2 == 0), 'Pulse delta start should be divisible by 2'
                        assert(p_delta_start >= 0), 'Pulse delta start is a negative number'
                    else:
                        assert( 1 == 2 ), 'Incorrect time dimension (s, ms, us, ns)'

                    self.pulse_array[i]['delta_start'] = str(delta_start)
                    self.shift_count = 1
                else:
                    pass

                i += 1

    def awg_redefine_phase(self, *, name, phase):
        """
        A function for redefining phase of the specified pulse.
        awg_redefine_phase(name = 'P0', phase = pi/2) changes phase of the 'P0' pulse to pi/2.
        The main purpose of the function phase cycling.

        def func(*, name1, name2): defines a function without default values of key arguments
        """

        if test_flag != 'test':
            i = 0

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:
                    self.pulse_array[i]['phase'] = float(phase)
                    self.shift_count = 1
                else:
                    pass

                i += 1

        elif test_flag == 'test':
            i = 0
            assert( name in self.pulse_name_array ), 'Pulse with the specified name is not defined'

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:
                    self.pulse_array[i]['phase'] = float(phase)
                    self.shift_count = 1
                else:
                    pass

                i += 1

    def awg_redefine_delta_phase(self, *, name, delta_phase):
        """
        A function for redefining delta_phase of the specified pulse.
        awg_redefine_delta_phase(name = 'P0', delta_phase = pi/2) changes delta_phase of the 'P0' pulse to pi/2.
        The main purpose of the function is non-uniform sampling.

        def func(*, name1, name2): defines a function without default values of key arguments
        """

        if test_flag != 'test':
            i = 0

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:
                    self.pulse_array[i]['delta_phase'] = float(delta_phase)
                    self.shift_count = 1
                else:
                    pass

                i += 1

        elif test_flag == 'test':
            i = 0
            assert( name in self.pulse_name_array ), 'Pulse with the specified name is not defined'

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:
                    self.pulse_array[i]['delta_phase'] = float(delta_phase)
                    self.shift_count = 1
                else:
                    pass

                i += 1

    def awg_add_phase(self, *, name, add_phase):
        """
        A function for adding a constant phase to the specified pulse.
        awg_add_phase(name = 'P0', add_phase = pi/2) changes the current phase of the 'P0' pulse 
        to the value ofcurrent_phase + pi/2.
        The main purpose of the function is phase cycling.

        def func(*, name1, name2): defines a function without default values of key arguments
        """

        if test_flag != 'test':
            i = 0

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:
                    self.pulse_array[i]['phase'] = self.pulse_array[i]['phase'] + float(add_phase)
                    self.shift_count = 1
                else:
                    pass

                i += 1

            # it is bad idea to update it here, since the phase of only one pulse
            # can be changed in one call of this function
            #self.awg_update_test()

        elif test_flag == 'test':
            i = 0
            assert( name in self.pulse_name_array ), 'Pulse with the specified name is not defined'

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:
                    self.pulse_array[i]['phase'] = self.pulse_array[i]['phase'] + float(add_phase)
                    self.shift_count = 1
                else:
                    pass

                i += 1

            #self.awg_update_test()

    def awg_redefine_increment(self, *, name, increment):
        """
        A function for redefining increment of the specified pulse.
        awg_redefine_increment(name = 'P0', increment = '10 ns') changes increment of the 'P0' pulse to '10 ns'.
        The main purpose of the function is non-uniform sampling.

        def func(*, name1, name2): defines a function without default values of key arguments
        """

        if test_flag != 'test':
            i = 0

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:
                    self.pulse_array[i]['increment'] = str(increment)
                    self.increment_count = 1
                else:
                    pass

                i += 1

        elif test_flag == 'test':
            i = 0
            assert( name in self.pulse_name_array ), 'Pulse with the specified name is not defined'

            while i < len( self.pulse_array ):
                if name == self.pulse_array[i]['name']:
                    # checks
                    temp_increment = increment.split(" ")
                    if temp_increment[1] in timebase_dict:
                        coef = timebase_dict[temp_increment[1]]
                        p_increment = coef*float(temp_increment[0])
                        assert (p_increment >= 0 and p_increment < max_pulse_length), \
                        'Length and sigma increment is longer than maximum available length or negative'
                    else:
                        assert( 1 == 2 ), 'Incorrect time dimension (ms, us, ns)'

                    self.pulse_array[i]['increment'] = str(increment)
                    self.increment_count = 1
                else:
                    pass

                i += 1

    def awg_shift(self, *pulses):
        """
        A function to shift the phase of the pulses for Single mode.
        Or the start of the pulses for Single Joined mode.
        The function directly affects the pulse_array.
        """
        if test_flag != 'test':
            self.shift_count = 1

            if len(pulses) == 0:
                i = 0
                if self.single_joined == 0:
                    while i < len( self.pulse_array ):
                        if float( self.pulse_array[i]['delta_phase'] ) == 0.:
                            pass
                        else:
                            temp = float( self.pulse_array[i]['phase'] )
                            temp2 = float( self.pulse_array[i]['delta_phase'] )

                            self.pulse_array[i]['phase'] = temp + temp2
     
                        i += 1

                # increment start if in Single Joined
                elif self.single_joined == 1:
                    while i < len( self.pulse_array ):
                        if int( self.pulse_array[i]['delta_start'][:-3] ) == 0:
                            pass
                        else:
                            # convertion to ns
                            temp = self.pulse_array[i]['delta_start'].split(' ')
                            if temp[1] in timebase_dict:
                                flag = timebase_dict[temp[1]]
                                d_start = int((temp[0]))*flag
                            else:
                                pass

                            temp2 = self.pulse_array[i]['start'].split(' ')
                            if temp2[1] in timebase_dict:
                                flag2 = timebase_dict[temp2[1]]
                                st = int((temp2[0]))*flag2
                            else:
                                pass
                                    
                            self.pulse_array[i]['start'] = str( st + d_start ) + ' ns'

                        i += 1

            else:
                if self.single_joined == 0:
                    set_from_list = set(pulses)
                    for element in set_from_list:
                        if element in self.pulse_name_array:
                            pulse_index = self.pulse_name_array.index(element)

                            if float( self.pulse_array[pulse_index]['delta_phase'] ) == 0.:
                                pass

                            else:
                                temp = float( self.pulse_array[pulse_index]['phase'] )
                                temp2 = float( self.pulse_array[pulse_index]['delta_phase'] )
                                        
                                self.pulse_array[pulse_index]['phase'] = temp + temp2

                elif self.single_joined == 1:
                    set_from_list = set(pulses)
                    for element in set_from_list:
                        if element in self.pulse_name_array:
                            pulse_index = self.pulse_name_array.index(element)

                            if int( self.pulse_array[pulse_index]['delta_start'][:-3] ) == 0:
                                pass
                            else:
                                # convertion to ns
                                temp = self.pulse_array[pulse_index]['delta_start'].split(' ')
                                if temp[1] in timebase_dict:
                                    flag = timebase_dict[temp[1]]
                                    d_start = int((temp[0]))*flag
                                else:
                                    pass

                                temp2 = self.pulse_array[pulse_index]['start'].split(' ')
                                if temp2[1] in timebase_dict:
                                    flag2 = timebase_dict[temp2[1]]
                                    st = int((temp2[0]))*flag2
                                else:
                                    pass
                                        
                                self.pulse_array[pulse_index]['start'] = str( st + d_start ) + ' ns'

        elif test_flag == 'test':
            self.shift_count = 1

            if len(pulses) == 0:
                i = 0
                if self.single_joined == 0:
                    while i < len( self.pulse_array ):
                        if float( self.pulse_array[i]['delta_phase'] ) == 0.:
                            pass
                        else:
                            temp = float( self.pulse_array[i]['phase'] )
                            temp2 = float( self.pulse_array[i]['delta_phase'] )

                            self.pulse_array[i]['phase'] = temp + temp2
                            
                        i += 1

                # increment start if in Single Joined
                elif self.single_joined == 1:
                    while i < len( self.pulse_array ):
                        if int( self.pulse_array[i]['delta_start'][:-3] ) == 0:
                            pass
                        else:
                            # convertion to ns
                            temp = self.pulse_array[i]['delta_start'].split(' ')
                            if temp[1] in timebase_dict:
                                flag = timebase_dict[temp[1]]
                                d_start = int((temp[0]))*flag
                            else:
                                pass

                            temp2 = self.pulse_array[i]['start'].split(' ')
                            if temp2[1] in timebase_dict:
                                flag2 = timebase_dict[temp2[1]]
                                st = int((temp2[0]))*flag2
                            else:
                                pass
                                    
                            self.pulse_array[i]['start'] = str( st + d_start ) + ' ns'

                        i += 1

            else:
                if self.single_joined == 0:
                    set_from_list = set(pulses)
                    for element in set_from_list:
                        if element in self.pulse_name_array:
                            pulse_index = self.pulse_name_array.index(element)

                            if float( self.pulse_array[pulse_index]['delta_phase'] ) == 0.:
                                pass

                            else:
                                temp = float( self.pulse_array[pulse_index]['phase'] )
                                temp2 = float( self.pulse_array[pulse_index]['delta_phase'] )
                                        
                                self.pulse_array[pulse_index]['phase'] = temp + temp2

                        else:
                            assert(1 == 2), "There is no pulse with the specified name"

                elif self.single_joined == 1:

                    set_from_list = set(pulses)
                    for element in set_from_list:
                        if element in self.pulse_name_array:
                            pulse_index = self.pulse_name_array.index(element)

                            if int( self.pulse_array[pulse_index]['delta_start'][:-3] ) == 0:
                                pass
                            else:
                                # convertion to ns
                                temp = self.pulse_array[pulse_index]['delta_start'].split(' ')
                                if temp[1] in timebase_dict:
                                    flag = timebase_dict[temp[1]]
                                    d_start = int((temp[0]))*flag
                                else:
                                    pass

                                temp2 = self.pulse_array[pulse_index]['start'].split(' ')
                                if temp2[1] in timebase_dict:
                                    flag2 = timebase_dict[temp2[1]]
                                    st = int((temp2[0]))*flag2
                                else:
                                    pass
                                        
                                self.pulse_array[pulse_index]['start'] = str( st + d_start ) + ' ns'

                        else:
                            assert(1 == 2), "There is no pulse with the specified name"

    def awg_increment(self, *pulses):
        """
        A function to increment both the length and sigma of the pulses.
        The function directly affects the pulse_array.
        """
        if test_flag != 'test':
            if len(pulses) == 0:
                i = 0
                while i < len( self.pulse_array ):
                    if int( self.pulse_array[i]['increment'][:-3] ) == 0:
                        pass
                    else:
                        # convertion to ns
                        temp = self.pulse_array[i]['increment'].split(' ')
                        if temp[1] in timebase_dict:
                            flag = timebase_dict[temp[1]]
                            d_length = int(float(temp[0]))*flag
                        else:
                            pass

                        temp2 = self.pulse_array[i]['length'].split(' ')
                        if temp2[1] in timebase_dict:
                            flag2 = timebase_dict[temp2[1]]
                            leng = int(float(temp2[0]))*flag2
                        else:
                            pass

                        temp3 = self.pulse_array[i]['sigma'].split(' ')
                        if temp3[1] in timebase_dict:
                            flag3 = timebase_dict[temp3[1]]
                            sigm = int(float(temp3[0]))*flag3
                        else:
                            pass

                        if self.pulse_array[i]['function'] == 'SINE':
                            self.pulse_array[i]['length'] = str( leng + d_length ) + ' ns'
                            self.pulse_array[i]['sigma'] = str( sigm + d_length ) + ' ns'
                        elif self.pulse_array[i]['function'] == 'GAUSS' or self.pulse_array[i]['function'] == 'SINC':
                            ratio = leng/sigm
                            self.pulse_array[i]['length'] = str( leng + ratio*d_length ) + ' ns'
                            self.pulse_array[i]['sigma'] = str( sigm + d_length ) + ' ns'

                    i += 1

                self.increment_count = 1

            else:
                set_from_list = set(pulses)
                for element in set_from_list:
                    if element in self.pulse_name_array:
                        pulse_index = self.pulse_name_array.index(element)

                        if int( self.pulse_array[pulse_index]['increment'][:-3] ) == 0:
                            pass
                        else:
                            # convertion to ns
                            temp = self.pulse_array[pulse_index]['increment'].split(' ')
                            if temp[1] in timebase_dict:
                                flag = timebase_dict[temp[1]]
                                d_length = int(float(temp[0]))*flag
                            else:
                                pass

                            temp2 = self.pulse_array[pulse_index]['length'].split(' ')
                            if temp2[1] in timebase_dict:
                                flag2 = timebase_dict[temp2[1]]
                                leng = int(float(temp2[0]))*flag2
                            else:
                                pass

                            temp3 = self.pulse_array[pulse_index]['sigma'].split(' ')
                            if temp3[1] in timebase_dict:
                                flag3 = timebase_dict[temp3[1]]
                                sigm = int(float(temp3[0]))*flag3
                            else:
                                pass

                            if self.pulse_array[i]['function'] == 'SINE':
                                self.pulse_array[i]['length'] = str( leng + d_length ) + ' ns'
                                self.pulse_array[i]['sigma'] = str( sigm + d_length ) + ' ns'
                            elif self.pulse_array[i]['function'] == 'GAUSS' or self.pulse_array[i]['function'] == 'SINC':
                                ratio = leng/sigm
                                self.pulse_array[i]['length'] = str( leng + ratio*d_length ) + ' ns'
                                self.pulse_array[i]['sigma'] = str( sigm + d_length ) + ' ns'

                        self.increment_count = 1

        elif test_flag == 'test':

            if len(pulses) == 0:
                i = 0
                while i < len( self.pulse_array ):
                    if int( self.pulse_array[i]['increment'][:-3] ) == 0:
                        pass
                    else:
                        # convertion to ns
                        temp = self.pulse_array[i]['increment'].split(' ')
                        if temp[1] in timebase_dict:
                            flag = timebase_dict[temp[1]]
                            d_length = int(float(temp[0]))*flag
                        else:
                            assert(1 == 2), "Incorrect time dimension (ns, us, ms)"

                        temp2 = self.pulse_array[i]['length'].split(' ')
                        if temp2[1] in timebase_dict:
                            flag2 = timebase_dict[temp2[1]]
                            leng = int(float(temp2[0]))*flag2
                        else:
                            assert(1 == 2), "Incorrect time dimension (ns, us, ms)"

                        temp3 = self.pulse_array[i]['sigma'].split(' ')
                        if temp3[1] in timebase_dict:
                            flag3 = timebase_dict[temp3[1]]
                            sigm = int(float(temp3[0]))*flag3
                        else:
                            assert(1 == 2), "Incorrect time dimension (ns, us, ms)"
                        
                        ratio = leng/sigm
                        if ( leng + ratio*d_length ) <= max_pulse_length:
                            self.pulse_array[i]['length'] = str( leng + d_length ) + ' ns'
                        else:
                            assert(1 == 2), 'Exceeded maximum pulse length (1900 ns) when increment the pulse'

                    i += 1

                self.increment_count = 1

            else:
                set_from_list = set(pulses)
                for element in set_from_list:
                    if element in self.pulse_name_array:

                        pulse_index = self.pulse_name_array.index(element)
                        if int( self.pulse_array[pulse_index]['increment'][:-3] ) == 0:
                            pass
                        else:
                            # convertion to ns
                            temp = self.pulse_array[pulse_index]['increment'].split(' ')
                            if temp[1] in timebase_dict:
                                flag = timebase_dict[temp[1]]
                                d_length = int(float(temp[0]))*flag
                            else:
                                assert(1 == 2), "Incorrect time dimension (ns, us, ms, s)"

                            temp2 = self.pulse_array[pulse_index]['length'].split(' ')
                            if temp2[1] in timebase_dict:
                                flag2 = timebase_dict[temp2[1]]
                                leng = int(float(temp2[0]))*flag2
                            else:
                                assert(1 == 2), "Incorrect time dimension (ns, us, ms, s)"
                            
                            temp3 = self.pulse_array[pulse_index]['sigma'].split(' ')
                            if temp3[1] in timebase_dict:
                                flag3 = timebase_dict[temp3[1]]
                                sigm = int(float(temp3[0]))*flag3
                            else:
                                assert(1 == 2), "Incorrect time dimension (ns, us, ms)"

                            ratio = leng/sigm
                            if ( leng + ratio*d_length ) <= max_pulse_length:
                                self.pulse_array[pulse_index]['length'] = str( leng + d_length ) + ' ns'
                            else:
                                assert(1 == 2), 'Exceeded maximum pulse length (1900 ns) when increment the pulse'

                        self.increment_count = 1

                    else:
                        assert(1 == 2), "There is no pulse with the specified name"

    def awg_reset(self):
        """
        Reset all pulses to the initial state it was in at the start of the experiment.
        It includes the complete functionality of awg_pulse_reset(), but also immediately
        updates the AWG card as it is done by calling awg_update().
        """
        if test_flag != 'test':
            # reset the pulses; deepcopy helps to create a TRULY NEW array
            self.pulse_array = deepcopy( self.pulse_array_init )

            self.awg_update()

            self.reset_count = 1
            self.increment_count = 0
            self.shift_count = 0
            self.current_phase_index = 0

        elif test_flag == 'test':
            # reset the pulses; deepcopy helps to create a TRULY NEW array
            self.pulse_array = deepcopy( self.pulse_array_init )

            self.awg_update()

            self.reset_count = 1
            self.increment_count = 0
            self.shift_count = 0
            self.current_phase_index = 0

    def awg_pulse_reset(self, *pulses):
        """
        Reset all pulses to the initial state it was in at the start of the experiment.
        It does not update the AWG card, if you want to reset all pulses and and also update 
        the AWG card use the function awg_reset() instead.
        """
        if test_flag != 'test':
            if len(pulses) == 0:
                self.pulse_array = deepcopy(self.pulse_array_init)
                self.reset_count = 0
                self.increment_count = 0
                self.shift_count = 0
            else:
                set_from_list = set(pulses)
                for element in set_from_list:
                    if element in self.pulse_name_array:
                        pulse_index = self.pulse_name_array.index(element)

                        self.pulse_array[pulse_index]['phase'] = self.pulse_array_init[pulse_index]['phase']
                        self.pulse_array[pulse_index]['length'] = self.pulse_array_init[pulse_index]['length']
                        self.pulse_array[pulse_index]['sigma'] = self.pulse_array_init[pulse_index]['sigma']

                        self.reset_count = 0
                        self.increment_count = 0
                        self.shift_count = 0

        elif test_flag == 'test':
            if len(pulses) == 0:
                self.pulse_array = deepcopy(self.pulse_array_init)
                self.reset_count = 0
                self.increment_count = 0
                self.shift_count = 0
            else:
                set_from_list = set(pulses)
                for element in set_from_list:
                    if element in self.pulse_name_array:
                        pulse_index = self.pulse_name_array.index(element)

                        self.pulse_array[pulse_index]['phase'] = self.pulse_array_init[pulse_index]['phase']
                        self.pulse_array[pulse_index]['length'] = self.pulse_array_init[pulse_index]['length']
                        self.pulse_array[pulse_index]['sigma'] = self.pulse_array_init[pulse_index]['sigma']

                        self.reset_count = 0
                        self.increment_count = 0
                        self.shift_count = 0

    def awg_number_of_segments(self, *segmnets):
        """
        Set or query the number of segments for 'Multi" card mode;
        AWG should be in 'Multi' mode. Please, refer to awg_card_mode() function
        Input: awg_number_of_segments(2); Number of segment is from the range 0-200
        Default: 1;
        Output: '2'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(segmnets) == 1:
                seg = int(segmnets[0])
                if self.card_mode == 512:
                    self.num_segments = seg
                elif self.card_mode == 32768 and seg == 1:
                    self.num_segments = seg
                else:
                    general.message('AWG is not in Multi mode')
                    sys.exit()
            elif len(segmnets) == 0:
                return self.num_segments

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(segmnets) == 1:
                seg = int(segmnets[0])
                if seg != 1:
                    assert( self.card_mode == 512), 'Number of segments higher than one is available only in Multi mode. Please, change it using awg_card_mode()'
                assert (seg > 0 and seg <= 200), 'Incorrect number of segments; Should be 0 < segmenets <= 200'
                self.num_segments = seg

            elif len(segmnets) == 0:
                return test_num_segments
            else:
                assert( 1 == 2 ), 'Incorrect argumnet'

    def awg_channel(self, *channel):
        """
        Enable output from the specified channel or query enabled channels;
        Input: awg_channel('CH0', 'CH1'); Channel is 'CH0' or 'CH1'
        Default: both channels are enabled
        Output: 'CH0'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(channel) == 1:
                ch = str(channel[0])
                if ch == 'CH0':
                    self.channel = 1
                    self.enable_out_0 = 1
                    self.enable_out_1 = 0
                elif ch == 'CH1':
                    self.channel = 2
                    self.enable_out_0 = 0
                    self.enable_out_1 = 1
                else:
                    general.message('Incorrect channel')
                    sys.exit()
            elif len(channel) == 2:
                ch1 = str(channel[0])
                ch2 = str(channel[1])
                if (ch1 == 'CH0' and ch2 == 'CH1') or (ch1 == 'CH1' and ch2 == 'CH0'):
                    self.channel = 3
                    self.enable_out_0 = 1
                    self.enable_out_1 = 1
                else:
                    general.message('Incorrect channel; Channel should be CH0 or CH1')
                    sys.exit()
            elif len(channel) == 0:
                if self.channel == 1:
                    return 'CH0'
                elif self.channel == 2:
                    return 'CH1'
                elif self.channel == 3:
                    return 'CH0, CH1'

            else:
                general.message('Incorrect argument; Channel should be CH0 or CH1')
                sys.exit()

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(channel) == 1:
                ch = str(channel[0])
                assert( ch == 'CH0' or ch == 'CH1' ), 'Incorrect channel; Channel should be CH0 or CH1'
                if ch == 'CH0':
                    self.channel = 1
                    self.enable_out_0 = 1
                    self.enable_out_1 = 0
                elif ch == 'CH1':
                    self.channel = 2
                    self.enable_out_0 = 0
                    self.enable_out_1 = 1
            elif len(channel) == 2:
                ch1 = str(channel[0])
                ch2 = str(channel[1])
                assert( (ch1 == 'CH0' and ch2 == 'CH1') or (ch1 == 'CH1' and ch2 == 'CH0')), 'Incorrect channel; Channel should be CH0 or CH1'
                if (ch1 == 'CH0' and ch2 == 'CH1') or (ch1 == 'CH1' and ch2 == 'CH0'):
                    self.channel = 3
                    self.enable_out_0 = 1
                    self.enable_out_1 = 1
            elif len(channel) == 0:
                return test_channel
            else:
                assert( 1 == 2 ), 'Incorrect argument'

    def awg_sample_rate(self, *s_rate):
        """
        Set or query sample rate; Range: 50 - 1250
        Input: awg_sample_rate('1250'); Sample rate is in MHz
        Default: '1250';
        Output: '1000 MHz'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(s_rate) == 1:
                rate = int(s_rate[0])
                if rate <= sample_rate_max and rate >= sample_rate_min:
                    self.sample_rate = rate
                else:
                    general.message('Incorrect sample rate; Should be 1250 MHz <= Rate <= 50 MHz')
                    sys.exit()

            elif len(s_rate) == 0:
                return str(self.sample_rate) + ' MHz'

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(s_rate) == 1:
                rate = int(s_rate[0])
                assert(rate <= sample_rate_max and rate >= sample_rate_min), "Incorrect sample rate; Should be 1250 MHz <= Rate <= 50 MHz"
                self.sample_rate = rate

            elif len(s_rate) == 0:
                return test_sample_rate
            else:
                assert( 1 == 2 ), 'Incorrect argument'

    def awg_clock_mode(self, *mode):
        """
        Set or query clock mode; the driver needs to know the external fed in frequency
        Input: awg_clock_mode('Internal'); Clock mode is 'Internal' or 'External'
        Default: 'Internal';
        Output: 'Internal'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(mode) == 1:
                md = str(mode[0])
                if md == 'Internal':
                    self.clock_mode = 1
                elif md == 'External':
                    self.clock_mode = 32
                else:
                    general.message('Incorrect clock mode; Only Internal and External modes are available')
                    sys.exit()

            elif len(mode) == 0:
                if self.clock_mode == 1:
                    return 'Internal'
                elif self.clock_mode == 32:
                    return 'External'

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(mode) == 1:
                md = str(mode[0])
                assert(md == 'Internal' or md == 'External'), "Incorrect clock mode; Only Internal and External modes are available"
                if md == 'Internal':
                    self.clock_mode = 1
                elif md == 'External':
                    self.clock_mode = 32

            elif len(mode) == 0:
                return test_clock_mode
            else:
                assert( 1 == 2 ), 'Incorrect argument'

    def awg_reference_clock(self, *ref_clock):
        """
        Set or query reference clock; the driver needs to know the external fed in frequency
        Input: awg_reference_clock(100); Reference clock is in MHz; Range: 10 - 1000
        Default: '100';
        Output: '200 MHz'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(ref_clock) == 1:
                rate = int(ref_clock[0])
                if rate <= sample_ref_clock_max and rate >= sample_ref_clock_min:
                    self.reference_clock = rate
                else:
                    general.message('Incorrect reference clock; Should be 1000 MHz <= Clock <= 10 MHz')
                    sys.exit()

            elif len(ref_clock) == 0:
                return str(self.reference_clock) + ' MHz'

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(ref_clock) == 1:
                rate = int(ref_clock[0])
                assert(rate <= sample_ref_clock_max and rate >= sample_ref_clock_min), "Incorrect reference clock; Should be 1000 MHz <= Clock <= 10 MHz"
                self.reference_clock = rate

            elif len(ref_clock) == 0:
                return test_ref_clock
            else:
                assert( 1 == 2 ), 'Incorrect argument'

    def awg_card_mode(self, *mode):
        """
        Set or query card mode;

        'Single' is "Data generation from on-board memory replaying the complete programmed memory on every detected trigger
        event. The number of replays can be programmed by loops."
        "Single Joined" is 'Single' with joined pulses in one combined buffer.
        'Multi' is "With every detected trigger event one data block (segment) is replayed."
        Segmented memory is available only in External trigger mode.
        'Sequence' is 'The sequence replay mode is a special firmware mode that allows to program an output sequence by defining one or more sequences each
        associated with a certain memory pattern.'

        Input: awg_card_mode('Single'); Card mode is 'Single'; 'Multi'; 'Sequence'
        Default: 'Single';
        Output: 'Single'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(mode) == 1:
                md = str(mode[0])
                if md == 'Single':
                    self.single_joined = 0
                    self.card_mode = 32768
                elif md == 'Single Joined':
                    self.single_joined = 1
                    self.card_mode = 32768                    
                elif md == 'Multi':
                    self.card_mode = 512
                elif md == 'Sequence':
                    self.sequence_mode = 1
                else:
                    general.message('Incorrect card mode; Only Single, Single Joined, Multi, and Sequence modes are available')
                    sys.exit()

            elif len(mode) == 0:
                if self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 0:
                    return 'Single'
                elif self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 1:
                    return 'Single Joined'
                elif self.card_mode == 512 and self.sequence_mode == 0:
                    return 'Multi'
                elif self.sequence_mode == 1:
                    return 'Sequence'

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(mode) == 1:
                md = str(mode[0])
                assert(md == 'Single' or md == 'Single Joined' or md == 'Multi' or md == 'Sequence'), "Incorrect card mode; Only Single,\
                                                                               Single Joined, Multi, and Sequence modes are available"
                if md == 'Single':
                    self.single_joined = 0
                    self.card_mode = 32768
                elif md == 'Single Joined':
                    self.single_joined = 1
                    self.card_mode = 32768
                elif md == 'Multi':
                    self.card_mode = 512
                elif md == 'Sequence':
                    self.sequence_mode = 1                    

            elif len(mode) == 0:
                return test_card_mode        
            else:
                assert( 1 == 2 ), 'Incorrect argument'

    def awg_trigger_channel(self, *ch):
        """
        Set or query trigger channel;
        Input: awg_trigger_channel('Software'); Trigger channel is 'Software'; 'External'
        Default: 'External';
        Output: 'Software'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(ch) == 1:
                md = str(ch[0])
                if md == 'Software':
                    self.trigger_ch = 1
                elif md == 'External':
                    self.trigger_ch = 2
                else:
                    general.message('Incorrect trigger channel; Only Software and External modes are available')
                    sys.exit()

            elif len(ch) == 0:
                if self.trigger_ch == 1:
                    return 'Software'
                elif self.trigger_ch == 2:
                    return 'External'

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(ch) == 1:
                md = str(ch[0])
                assert(md == 'Software' or md == 'External'), "Incorrect trigger channel; Only Software and External modes are available"
                if md == 'Software':
                    self.trigger_ch = 1
                elif md == 'External':
                    self.trigger_ch = 2

            elif len(ch) == 0:
                return test_trigger_ch
            else:
                assert( 1 == 2 ), 'Incorrect argument'

    def awg_trigger_mode(self, *mode):
        """
        Set or query trigger mode;
        Input: awg_trigger_mode('Positive'); Trigger mode is 'Positive'; 'Negative'; 'High'; 'Low'
        Default: 'Positive';
        Output: 'Positive'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(mode) == 1:
                md = str(mode[0])
                if md == 'Positive':
                    self.trigger_mode = 1
                elif md == 'Negative':
                    self.trigger_mode = 2
                elif md == 'High':
                    self.trigger_mode = 8
                elif md == 'Low':
                    self.trigger_mode = 10
                else:
                    general.message("Incorrect trigger mode; Only Positive, Negative, High, and Low are available")
                    sys.exit()

            elif len(mode) == 0:
                if self.trigger_mode == 1:
                    return 'Positive'
                elif self.trigger_mode == 2:
                    return 'Negative'
                elif self.trigger_mode == 8:
                    return 'High'
                elif self.trigger_mode == 10:
                    return 'Low'

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(mode) == 1:
                md = str(mode[0])
                assert(md == 'Positive' or md == 'Negative' or md == 'High' or md == 'Low'), "Incorrect trigger mode; \
                    Only Positive, Negative, High, and Low are available"
                if md == 'Positive':
                    self.trigger_mode = 1
                elif md == 'Negative':
                    self.trigger_mode = 2
                elif md == 'High':
                    self.trigger_mode = 8
                elif md == 'Low':
                    self.trigger_mode = 10

            elif len(mode) == 0:
                return test_trigger_mode        
            else:
                assert( 1 == 2 ), 'Incorrect argument'

    def awg_loop(self, *loop):
        """
        Set or query number of loop;
        Input: awg_loop(0); Number of loop from 1 to 100000; 0 is infinite loop
        Default: 0;
        Output: '100'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(loop) == 1:
                lp = int(loop[0])
                self.loop = lp

            elif len(loop) == 0:
                return self.loop

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(loop) == 1:
                lp = int(loop[0])
                assert( lp >= 0 and lp <= loop_max ), "Incorrect number of loops; Should be 0 <= Loop <= 100000"
                self.loop = lp

            elif len(loop) == 0:
                return test_loop      
            else:
                assert( 1 == 2 ), 'Incorrect argument'

    def awg_trigger_delay(self, *delay):
        """
        Set or query trigger delay;
        Input: awg_trigger_delay('100 ns'); delay in [ms, us, ns]
        Step is 32 sample clock; will be rounded if input is not divisible by 32 sample clock
        Default: 0 ns;
        Output: '100 ns'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(delay) == 1:
                temp = delay[0].split(' ')
                delay_num = int(temp[0])
                dimen = str(temp[1])

                if dimen in timebase_dict:
                    flag = timebase_dict[dimen]
                    # trigger delay in samples; maximum is 8589934560, step is 32
                    del_in_sample = int( delay_num*flag*self.sample_rate / 1000 )
                    if del_in_sample % 32 != 0:
                        #self.delay = int( 32*(del_in_sample // 32) )
                        self.delay = self.round_to_closest( del_in_sample, 32 )
                        general.message('Delay should be divisible by 32 samples (25.6 ns at 1.250 GHz); The closest avalaibale number ' + str(self.delay * 1000 / self.sample_rate) + ' ns is used')

                    else:
                        self.delay = del_in_sample

                else:
                    general.message('Incorrect delay dimension; Should be ns, us or ms')
                    sys.exit()

            elif len(delay) == 0:
                return str(self.delay / self.sample_rate * 1000) + ' ns'

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(delay) == 1:
                temp = delay[0].split(' ')
                delay_num = int(temp[0])
                dimen = str(temp[1])

                assert( dimen in timebase_dict), 'Incorrect delay dimension; Should be ns, us or ms'
                flag = timebase_dict[dimen]
                # trigger delay in samples; maximum is 8589934560, step is 32
                del_in_sample = int( delay_num*flag*self.sample_rate / 1000 )
                if del_in_sample % 32 != 0:
                    #self.delay = int( 32*(del_in_sample // 32) )
                    self.delay = self.round_to_closest( del_in_sample, 32 )
                else:
                    self.delay = del_in_sample

                assert(self.delay >= delay_min and self.delay <= delay_max), 'Incorrect delay; Should be 0 <= Delay <= 8589934560 samples'


            elif len(delay) == 0:
                return test_delay
            else:
                assert( 1 == 2 ), 'Incorrect argument'

    def awg_amplitude(self, *amplitude):
        """
        Set or query amplitude of the channel;
        Input: awg_amplitude('CH0', '600'); amplitude is in mV
        awg_amplitude('CH0', '600', 'CH1', '600')
        Default: CH0 - 600 mV; CH1 - 533 mV;
        Output: '600 mV'
        """
        if test_flag != 'test':
            self.setting_change_count = 1

            if len(amplitude) == 2:
                ch = str(amplitude[0])
                ampl = int(amplitude[1])
                if ch == 'CH0':
                    self.amplitude_0 = ampl
                elif ch == 'CH1':
                    self.amplitude_1 = ampl
            
            elif len(amplitude) == 4:
                ch1 = str(amplitude[0])
                ampl1 = int(amplitude[1])
                ch2 = str(amplitude[2])
                ampl2 = int(amplitude[3])
                if ch1 == 'CH0':
                    self.amplitude_0 = ampl1
                elif ch1 == 'CH1':
                    self.amplitude_1 = ampl1
                if ch2 == 'CH0':
                    self.amplitude_0 = ampl2
                elif ch2 == 'CH1':
                    self.amplitude_1 = ampl2

            elif len(amplitude) == 1:
                ch = str(amplitude[0])
                if ch == 'CH0':
                    return str(self.amplitude_0) + ' mV'
                elif ch == 'CH1':
                    return str(self.amplitude_1) + ' mV'

        elif test_flag == 'test':
            self.setting_change_count = 1

            if len(amplitude) == 2:
                ch = str(amplitude[0])
                ampl = int(amplitude[1])
                assert(ch == 'CH0' or ch == 'CH1'), "Incorrect channel; Should be CH0 or CH1"
                assert( ampl >= amplitude_min and ampl <= amplitude_max ), "Incorrect amplitude; Should be 80 <= amplitude <= 2500"
                if ch == 'CH0':
                    self.amplitude_0 = ampl
                elif ch == 'CH1':
                    self.amplitude_1 = ampl
            
            elif len(amplitude) == 4:
                ch1 = str(amplitude[0])
                ampl1 = int(amplitude[1])
                ch2 = str(amplitude[2])
                ampl2 = int(amplitude[3])
                assert(ch1 == 'CH0' or ch1 == 'CH1'), "Incorrect channel 1; Should be CH0 or CH1"
                assert( ampl1 >= amplitude_min and ampl1 <= amplitude_max ), "Incorrect amplitude 1; Should be 80 <= amplitude <= 2500"
                assert(ch2 == 'CH0' or ch2 == 'CH1'), "Incorrect channel 2; Should be CH0 or CH1"
                assert( ampl2 >= amplitude_min and ampl2 <= amplitude_max ), "Incorrect amplitude 2; Should be 80 <= amplitude <= 2500"
                if ch1 == 'CH0':
                    self.amplitude_0 = ampl1
                elif ch1 == 'CH1':
                    self.amplitude_1 = ampl1
                if ch2 == 'CH0':
                    self.amplitude_0 = ampl2
                elif ch2 == 'CH1':
                    self.amplitude_1 = ampl2

            elif len(amplitude) == 1:
                assert(ch1 == 'CH0' or ch1 == 'CH1'), "Incorrect channel; Should be CH0 or CH1"
                return test_amplitude

            else:
                assert( 1 == 2 ), 'Incorrect arguments'

    #CHECK
    def awg_pulse_list(self):
        """
        Function for saving a pulse list from 
        the script into the header of the experimental data
        """
        pulse_list_mod = ''
        if self.card_mode == 32768 and self.single_joined == 0:
            pulse_list_mod = pulse_list_mod + 'AWG card mode: ' + str( "Single" ) + '\n'
        elif self.card_mode == 32768 and self.single_joined == 1:
            pulse_list_mod = pulse_list_mod + 'AWG card mode: ' + str( "Single Joined" ) + '\n'
        elif self.card_mode == 512:
            pulse_list_mod = pulse_list_mod + 'AWG card mode: ' + str( "Multi" ) + '\n'
        elif self.card_mode == 262144:
            pulse_list_mod = pulse_list_mod + 'AWG card mode: ' + str( "Sequence" ) + '\n'

        if  self.sequence_mode == 0:
            for element in self.pulse_array:
                pulse_list_mod = pulse_list_mod + str(element) + '\n'

            return pulse_list_mod

        elif  self.sequence_mode == 1:
            pulse_list_mod = pulse_list_mod + str(arguments_array) + '\n'

            return pulse_list_mod

    def awg_visualize(self):
        """
        A function for buffer visualization
        """
        if test_flag != 'test':

            if self.sequence_mode == 0:
                if self.card_mode == 32768 and (self.channel == 1 or self.channel == 2) and self.single_joined == 0:
                    buf = self.define_buffer_single()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    # A way to convert c_types poinet back to np.array
                    # np.ctypeslib.as_array(buf, shape = (int(self.memsize), ))
                    # also try
                    # pbyData = cast  (addressof (pvBuffer) + lPCPos.value, ptr8) # cast to pointer to 8bit integer
                    general.plot_1d('Buffer_single', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize), )), \
                                    label = 'ch0 or ch1')

                # Single Joined
                elif self.card_mode == 32768 and (self.channel == 1 or self.channel == 2) and self.single_joined == 1:
                    buf = self.define_buffer_single_joined()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    # A way to convert c_types poinet back to np.array
                    # np.ctypeslib.as_array(buf, shape = (int(self.memsize), ))
                    general.plot_1d('Buffer_single_joined', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize), )), \
                                    label = 'ch0 or ch1')

                elif self.card_mode == 32768 and self.channel == 3 and self.single_joined == 0:
                    buf = self.define_buffer_single()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    general.plot_1d('Buffer_single', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[0::2], \
                                    label = 'ch0')
                    general.plot_1d('Buffer_single', xs, 2 * maxCAD + np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[1::2], \
                                    label = 'ch1')

                # Single Joined
                elif self.card_mode == 32768 and self.channel == 3 and self.single_joined == 1:
                    buf = self.define_buffer_single_joined()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    general.plot_1d('Buffer_single_joined', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[0::2], \
                                    label = 'ch0')
                    general.plot_1d('Buffer_single_joined', xs, 2 * maxCAD + np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[1::2], \
                                    label = 'ch1')

                elif self.card_mode == 512 and (self.channel == 1 or self.channel == 2):
                    buf = self.define_buffer_multi()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    general.plot_1d('Buffer_multi', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize), )), \
                                    label = 'ch0 or ch1')
               
                elif self.card_mode == 512 and self.channel == 3:
                    buf = self.define_buffer_multi()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    general.plot_1d('Buffer_multi', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[0::2], \
                                    label = 'ch0')
                    general.plot_1d('Buffer_multi', xs, 2 * maxCAD + np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[1::2], \
                                    label = 'ch1')

            elif self.sequence_mode == 1:
                if self.channel != 3:
                    xs = np.arange( 0, int(self.qwBufferSize.value/2) )*0.8

                    general.plot_1d('Buffer_sequence', xs, np.ctypeslib.as_array(self.full_buffer[0], shape = (int(self.qwBufferSize.value/2), )),\
                                        label = '0ch0')
                    general.plot_1d('Buffer_sequence', xs, 2*maxCAD + np.ctypeslib.as_array(self.full_buffer[1], shape = (int(self.qwBufferSize.value/2), )), \
                                        label = '1ch0')
                    general.plot_1d('Buffer_sequence', xs, 4*maxCAD + np.ctypeslib.as_array(self.full_buffer[2], shape = (int(self.qwBufferSize.value/2), )), \
                                        label = '2ch0')

                elif self.channel == 3:
                    xs = np.arange( 0, int(self.qwBufferSize.value/4) )*0.8
                
                    general.plot_1d('Buffer_sequence', xs, np.ctypeslib.as_array(self.full_buffer[0], \
                                    shape = (int(self.qwBufferSize.value/2), ))[0::2], label = '0ch0')
                    general.plot_1d('Buffer_sequence', xs, 2*maxCAD + np.ctypeslib.as_array(self.full_buffer[0], \
                                    shape = (int(self.qwBufferSize.value/2), ))[1::2], label = '0ch1')
                    general.plot_1d('Buffer_sequence', xs, 4*maxCAD + np.ctypeslib.as_array(self.full_buffer[1], \
                                    shape = (int(self.qwBufferSize.value/2), ))[0::2], label = '1ch0')
                    general.plot_1d('Buffer_sequence', xs, 6*maxCAD + np.ctypeslib.as_array(self.full_buffer[1], \
                                    shape = (int(self.qwBufferSize.value/2), ))[1::2], label = '1ch1')

                    #pure_sinus = np.zeros(len(xs))
                    #for i in range(len(xs)):
                    #    pure_sinus[i] = sin(2*pi*(( i ))*70 / 1250)
                    #general.plot_1d('buffer', xs, 6 + pure_sinus, label = 'sine')

        elif test_flag == 'test':
            if self.sequence_mode == 0:
                if self.card_mode == 32768 and (self.channel == 1 or self.channel == 2) and self.single_joined == 0:
                    buf = self.define_buffer_single()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    general.plot_1d('Buffer_single', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize), )), \
                                    label = 'ch0 or ch1')

                # Single Joined
                elif self.card_mode == 32768 and (self.channel == 1 or self.channel == 2) and self.single_joined == 1:
                    buf = self.define_buffer_single_joined()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    # A way to convert c_types poinet back to np.array
                    #np.ctypeslib.as_array(buf, shape = (int(self.memsize), ))
                    general.plot_1d('Buffer_single_joined', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize), )), \
                                    label = 'ch0 or ch1')

                elif self.card_mode == 32768 and self.channel == 3 and self.single_joined == 0:
                    buf = self.define_buffer_single()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    general.plot_1d('Buffer_single', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[0::2], \
                                    label = 'ch0')
                    general.plot_1d('Buffer_single', xs, 2 * maxCAD + np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[1::2], \
                                    label = 'ch1')
                
                # Single Joined
                elif self.card_mode == 32768 and self.channel == 3 and self.single_joined == 1:
                    buf = self.define_buffer_single_joined()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    general.plot_1d('Buffer_single_joined', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[0::2], \
                                    label = 'ch0')
                    general.plot_1d('Buffer_single_joined', xs, 2 * maxCAD + np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[1::2], \
                                    label = 'ch1')

                elif self.card_mode == 512 and (self.channel == 1 or self.channel == 2):
                    buf = self.define_buffer_multi()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    general.plot_1d('Buffer_multi', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize), )), \
                                    label = 'ch0 or ch1')
               
                elif self.card_mode == 512 and self.channel == 3:
                    buf = self.define_buffer_multi()[1]
                    xs = 0.8*np.arange( int(self.memsize) )
                    general.plot_1d('Buffer_multi', xs, np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[0::2], \
                                    label = 'ch0')
                    general.plot_1d('Buffer_multi', xs, 2 * maxCAD + np.ctypeslib.as_array(buf, shape = (int(self.memsize * 2), ))[1::2], \
                                    label = 'ch1')

            elif self.sequence_mode == 1:
                if self.channel != 3:
                    pass
                    # no buffer is filled in test


                elif self.channel == 3:
                    pass
                    # no buffer is filled in test

            elif self.full_buffer != 0:
                assert( 1 == 2 ), 'No pulse sequence is defined'

    def awg_pulse_sequence(self, *, pulse_type, pulse_start, pulse_delta_start,\
                            pulse_length, pulse_phase, pulse_sigma, pulse_frequency, number_of_points, loop, rep_rate):
        """
        A function for Sequence mode of the AWG card.
        The sequence replay mode is a special firmware mode that allows to program an output sequence by defining one or more sequences each
        associated with a certain memory pattern.
        #######
        Possible trigger is needed only once. Should be also zeros that define repetition rate
        #######

        Pulse_start must be sorted
        number of channel should be defined before awg.awg_pulse_sequence()

        The function refilling the sequence buffer after each call
        After going through all the buffer the AWG card will be stopped
        Inside an experimental script this function can be called only with all arguments specified
        """
        if test_flag != 'test':

            # collect arguments in special array for further handling
            arguments_array = []
            arguments_array.append(pulse_type)          #   0   type: SINE, GAUSS, SINC
            arguments_array.append(pulse_start)         #   1   start
            arguments_array.append(pulse_delta_start)   #   2   delta start
            arguments_array.append(pulse_length)        #   3   length
            arguments_array.append(pulse_phase)         #   4   phase
            arguments_array.append(pulse_sigma)         #   5   sigma
            arguments_array.append(pulse_frequency)     #   6   frequency
            arguments_array.append(number_of_points)    #   7   number of points in sequence
            arguments_array.append(loop)                #   8   number of loops
            arguments_array.append(rep_rate)            #   9   repetition rate

            # turn on the sequence mode
            self.sequence_mode = 1
            # number of segments should be power of two
            # segments can be empty
            self.sequence_segments = self.closest_power_of_two( arguments_array[7] )
            #self.sequence_segments = self.round_to_closest( arguments_array[7], 32 )
            self.sequence_loop = arguments_array[8]
            # repetition rate for pulse sequence in samples; Hz -> ns -> samples
            seq_rep_rate = int( ( 10**9 / (arguments_array[9])) * self.sample_rate/1000 )
            if seq_rep_rate % 32 != 0:
                #seq_rep_rate = int( 32*(seq_rep_rate // 32) )
                seq_rep_rate = self.round_to_closest( seq_rep_rate, 32 )
                general.message('Repetition rate should be divisible by 25.6 ns; The closest avalaibale number is used: ' + str( 1/(seq_rep_rate*10**-6 / self.sample_rate )) + ' Hz' )

            # convert phase list to radians
            pulse_phase_converted = []
            for el in arguments_array[4]:
                if el == '+x':
                    pulse_phase_converted.append(0)
                elif el == '-x':
                    pulse_phase_converted.append(pi)
                elif el == '+y':
                    pulse_phase_converted.append(pi/2)
                elif el == '-y':
                    pulse_phase_converted.append(3*pi/2)
                elif el == 'rand':
                    # 1000 will never be reached
                    # use it for distinguish DEER random phase pulse
                    pulse_phase_converted.append(1000)
            
            # convert everything in samples 
            pulse_phase_np = np.asarray(pulse_phase_converted)
            pulse_start_smp = (np.asarray(arguments_array[1])*self.sample_rate/1000).astype('int64')
            pulse_delta_start_smp = (np.asarray(arguments_array[2])*self.sample_rate/1000).astype('int64')
            pulse_length_smp = (np.asarray(arguments_array[3])*self.sample_rate/1000).astype('int64')
            pulse_sigma_smp = np.asarray(arguments_array[5])*self.sample_rate/1000

            # we do not expect that pulses will have drastically different
            # pulse length. It is better to define the same buffer length 
            # for each segment, since in this case we should not worry about
            # saving the information about memory size for each segments
            max_pulse_length = max( pulse_length_smp )
            max_start = max( pulse_start_smp )
            max_delta_start = max( pulse_delta_start_smp )
            # buffer length is defined from the largest delay
            #segment_length = self.closest_power_of_two( max_start + max_pulse_length + max_delta_start*arguments_array[7] )
            # buffer length is defined from the repetiton rate
            segment_length = seq_rep_rate

            # define buffer differently for only one or two channels enabled
            # for ch1 phase is automatically shifted by phase_shift_ch1_seq_mode
            if self.channel == 1 or self.channel == 2:
                #self.full_buffer = np.zeros( ( arguments_array[7], segment_length ), dtype = np.float64 )
                # save buffer directly in c_types format
                self.full_buffer = []
                self.full_buffer_pointer = []

                # define the buffer
                pnBuffer = c_void_p()
                self.qwBufferSize = uint64 (2 * segment_length)  # buffer size
                
                # run over all defined pulses for each point
                for point in range(arguments_array[7]):
                    # clear the buffer
                    pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
                    pnBuffer = cast (pvBuffer, ptr16)

                    # run over defined pulses inside a sequence point
                    for index, element in enumerate(pulse_type):
                        # for DEER pulse with random phase
                        rnd_phase = 2*pi*random.random()

                        if element == 'SINE':
                            for i in range(pulse_start_smp[index] + pulse_delta_start_smp[index]*point, segment_length):

                                    if i < (pulse_start_smp[index] + pulse_delta_start_smp[index]*point):
                                        pass
                                    elif ( i >= ( pulse_start_smp[index] + pulse_delta_start_smp[index]*point) and \
                                           i <= (pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point) ):
                                        
                                        if pulse_phase_np[index] != 1000:
                                            #self.full_buffer[point, i] 
                                            pnBuffer[i] = int( maxCAD * sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )

                                        else:
                                            #self.full_buffer[point, i] 
                                            pnBuffer[i] = int( maxCAD * sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + rnd_phase) )
                                    else:
                                        break

                        elif element == 'GAUSS':
                            # mid_point for GAUSS and SINC
                            mid_point = int( (pulse_start_smp[index] + pulse_delta_start_smp[index]*point + \
                                        pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point)/2 )

                            for i in range(pulse_start_smp[index] + pulse_delta_start_smp[index]*point, segment_length):
                                    if i < (pulse_start_smp[index] + pulse_delta_start_smp[index]*point):
                                        pass
                                    elif ( i >= ( pulse_start_smp[index] + pulse_delta_start_smp[index]*point) and \
                                           i <= (pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point) ):
                                        
                                        if pulse_phase_np[index] != 1000:
                                            #self.full_buffer[point, i] 
                                            pnBuffer[i] = int( maxCAD * exp(-((( i ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                                                    sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )

                                        else:
                                            #self.full_buffer[point, i] 
                                            pnBuffer[i] = int( maxCAD * exp(-((( i ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                                                    sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                                                    rnd_phase) )
                                    else:
                                        break

                        elif element == 'SINC':
                            # mid_point for GAUSS and SINC
                            mid_point = int( (pulse_start_smp[index] + pulse_delta_start_smp[index]*point + \
                                        pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point)/2 )

                            for i in range(pulse_start_smp[index] + pulse_delta_start_smp[index]*point, segment_length):
                                    if i < (pulse_start_smp[index] + pulse_delta_start_smp[index]*point):
                                        pass
                                    elif ( i >= ( pulse_start_smp[index] + pulse_delta_start_smp[index]*point) and \
                                           i <= (pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point) ):
                                        
                                        if pulse_phase_np[index] != 1000:
                                            #self.full_buffer[point, i] 
                                            pnBuffer[i] = int( maxCAD * np.sinc(2*(( i ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                                                    sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )

                                        else:
                                            #self.full_buffer[point, i]
                                            pnBuffer[i] = int( maxCAD * np.sinc(2*(( i ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                                                    sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                                                    rnd_phase) )
                                    else:
                                        break

                        elif element == 'BLANK':
                            break

                    self.full_buffer.append(pnBuffer)
                    self.full_buffer_pointer.append(pvBuffer)
                
                # for sequence_segments as a power of two
                for point in range(arguments_array[7], self.sequence_segments):
                    # clear the buffer
                    pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
                    pnBuffer = cast (pvBuffer, ptr16)

                    self.full_buffer.append(pnBuffer)
                    self.full_buffer_pointer.append(pvBuffer)

            elif self.channel == 3:
                #self.full_buffer = np.zeros( ( arguments_array[7], 2*segment_length ), dtype = np.float64 )
                # save buffer directly in c_types format
                self.full_buffer = []
                self.full_buffer_pointer = []

                # define the buffer
                pnBuffer = c_void_p()
                self.qwBufferSize = uint64 (2 * segment_length * 2)  # buffer size for two channels

                # run over all defined pulses for each point
                for point in range(arguments_array[7]):
                    # clear the buffer
                    pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
                    pnBuffer = cast (pvBuffer, ptr16)

                    # run over defined pulses inside a sequence point
                    for index, element in enumerate(pulse_type):
                        # DEER pulse
                        rnd_phase = 2*pi*random.random()

                        if element == 'SINE':

                            for i in range(2 * (pulse_start_smp[index] + pulse_delta_start_smp[index]*point), 2*segment_length, 2):
                                    if i < 2 * (pulse_start_smp[index] + pulse_delta_start_smp[index]*point):
                                        pass
                                    elif ( i >= 2 * ( pulse_start_smp[index] + pulse_delta_start_smp[index]*point) and \
                                           i <= 2 * (pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point) ):
                                        
                                        if pulse_phase_np[index] != 1000:
                                            #self.full_buffer[point, i]
                                            #ch0
                                            pnBuffer[i] = int( maxCAD * sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )
                                            #ch1
                                            pnBuffer[i + 1] = int( maxCAD * sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                                                     phase_shift_ch1_seq_mode) )

                                        else:
                                            #self.full_buffer[point, i]
                                            #ch0
                                            pnBuffer[i] = int( maxCAD * sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + rnd_phase) )
                                            #ch1
                                            pnBuffer[i + 1] = int( maxCAD * sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + rnd_phase + \
                                                                     phase_shift_ch1_seq_mode) )                                            
                                    else:
                                        break

                        elif element == 'GAUSS':
                            # mid_point for GAUSS and SINC
                            mid_point = int( (pulse_start_smp[index] + pulse_delta_start_smp[index]*point + \
                                pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point)/2 )

                            for i in range(2 * (pulse_start_smp[index] + pulse_delta_start_smp[index]*point), 2*segment_length, 2):

                                    if i < 2 * (pulse_start_smp[index] + pulse_delta_start_smp[index]*point):
                                        pass
                                    elif ( i >= 2 * ( pulse_start_smp[index] + pulse_delta_start_smp[index]*point) and \
                                           i <= 2 * (pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point) ):
                                        
                                        if pulse_phase_np[index] != 1000:
                                            #self.full_buffer[point, i]
                                            #ch0
                                            pnBuffer[i] = int( maxCAD * exp(-((( i/2 ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                                                    sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )
                                            #ch1
                                            pnBuffer[i + 1] = int( maxCAD * exp(-((( ( i )/2 ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                                                    sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                                                    phase_shift_ch1_seq_mode) )

                                        else:
                                            #self.full_buffer[point, i]
                                            #ch0
                                            pnBuffer[i] = int( maxCAD * exp(-((( i/2 ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                                                    sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + rnd_phase ) )
                                            #ch1
                                            pnBuffer[i + 1] = int( maxCAD * exp(-((( ( i )/2 ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                                                    sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + rnd_phase +\
                                                                    phase_shift_ch1_seq_mode) )

                                    else:
                                        break
                            
                        elif element == 'SINC':
                            # mid_point for GAUSS and SINC
                            mid_point = int( (pulse_start_smp[index] + pulse_delta_start_smp[index]*point + \
                                pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point)/2 )

                            for i in range(2 * (pulse_start_smp[index] + pulse_delta_start_smp[index]*point), 2*segment_length, 2):

                                    if i < 2 * (pulse_start_smp[index] + pulse_delta_start_smp[index]*point):
                                        pass
                                    elif ( i >= 2 * ( pulse_start_smp[index] + pulse_delta_start_smp[index]*point) and \
                                           i <= 2 * (pulse_start_smp[index] + pulse_length_smp[index] + pulse_delta_start_smp[index]*point) ):
                                        
                                        if pulse_phase_np[index] != 1000:
                                            #self.full_buffer[point, i]
                                            #ch0
                                            pnBuffer[i] = int( maxCAD * np.sinc(2*(( i/2 ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                                                     sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )
                                            #ch1
                                            pnBuffer[i + 1] = int( maxCAD * np.sinc(2*(( ( i )/2 ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                                                     sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                                                     phase_shift_ch1_seq_mode) )                                            

                                        else:
                                            #self.full_buffer[point, i]
                                            #ch0
                                            pnBuffer[i] = int( maxCAD * np.sinc(2*(( i/2 ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                                                     sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] + rnd_phase) )
                                            #ch1
                                            pnBuffer[i + 1] = int( maxCAD * np.sinc(2*(( ( i )/2 ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                                                     sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                                                     phase_shift_ch1_seq_mode + rnd_phase) )                                            

                                    else:
                                        break
                        
                        elif element == 'BLANK':
                            break

                    self.full_buffer.append(pnBuffer)
                    self.full_buffer_pointer.append(pvBuffer)

                # for sequence_segments as a power of two
                for point in range(arguments_array[7], self.sequence_segments):
                    # clear the buffer
                    pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
                    pnBuffer = cast (pvBuffer, ptr16)

                    self.full_buffer.append(pnBuffer)
                    self.full_buffer_pointer.append(pvBuffer)

            #general.message( (self.full_buffer_pointer[0]) )
            # A way to convert c_types poinet back to np.array
            #general.message( np.ctypeslib.as_array(self.full_buffer[0], shape = (int(self.qwBufferSize.value/2), )) )
            self.sequence_mode_count == 1

        elif test_flag == 'test':
            self.sequence_mode = 1
            # check that length of pulse_start, pulse_delta_start and other are the same
            # collect arguments in special array for check
            arguments_array = []
            length_array = []

            arguments_array.append(pulse_type)          #   0
            length_array.append( len( pulse_type ) )
            arguments_array.append(pulse_start)         #   1
            length_array.append( len( pulse_start ) )
            arguments_array.append(pulse_delta_start)   #   2
            length_array.append( len( pulse_delta_start ) )
            arguments_array.append(pulse_length)        #   3
            length_array.append( len( pulse_length ) )
            arguments_array.append(pulse_phase)         #   4
            length_array.append( len( pulse_phase ) )
            arguments_array.append(pulse_sigma)         #   5
            length_array.append( len( pulse_sigma ) )
            arguments_array.append(pulse_frequency)     #   6
            length_array.append( len( pulse_frequency ) )
            arguments_array.append(number_of_points)    #   7
            arguments_array.append(loop)                #   8

            assert( loop > 0 ), 'Number of loops should be higher than 0'
            assert( number_of_points > 0 ), 'Number of points in pulse sequence should be higher than 0'
            assert( rep_rate > 0 ), 'Repetition rate is given in Hz and should be a positive number'

            for el in arguments_array[0]:
                assert( el == 'SINE' or el == 'GAUSS' or el == 'SINC' or el == 'BLANK'), 'Only SINE, GAUSS, SINC, and BLANK pulses are available'

            for el in arguments_array[1]:
                assert( el >= 0 ), 'Pulse starts should be positive numbers'
                assert( el % 2 == 0 ), 'Pulse starts should be divisible by 2'

            for el in arguments_array[2]:
                assert( el >= 0 ), 'Pulse delta starts should be positive numbers'
                assert( el % 2 == 0 ), 'Pulse delta starts should be divisible by 2'

            for el in arguments_array[3]:
                assert( el > 0 ), 'Pulse lengths should be positive numbers'
                assert( el % 2 == 0 ), 'Pulse lengths should be divisible by 2'

            for el in arguments_array[4]:
                #assert( type(el) == int or type(el) == float ), 'Phase should be a number'
                assert( el == '+x' or el == '-x' or el == '-y' or el == '+y' ), 'Pulse phase should be one of the following: [+x, -x, +y, -y]'

            for el in arguments_array[5]:
                assert( el > 0 ), 'Pulse sigmas should be positive numbers'

            for el in arguments_array[5]:
                assert( el >= min_freq and el <= max_freq), 'Pulse frequency should be from 1 to 280 MHz'

            # check that the length is equal (compare all elements in length_array)
            gr = groupby(length_array)
            if (next(gr, True) and not next(gr, False)) == False:
                assert(1 == 2), 'Defined arrays in pulse sequence does not have equal length'

            # TO DO
            # overlapping check
            # It will be done by pulse_programmer RECT_AWG pulses

    # Auxilary functions
    def awg_update_test(self):
        """
        Function that can be used for tests instead of awg_update()
        """
        if test_flag != 'test':
            if self.reset_count == 0 or self.shift_count == 1 or self.increment_count == 1 or self.setting_change_count == 1:
                if self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 0:
                    buf = self.define_buffer_single()[0]
                elif self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 1:
                    buf = self.define_buffer_single_joined()[0]
                elif self.card_mode == 512 and self.sequence_mode == 0:
                    buf = self.define_buffer_multi()[0]

                self.reset_count = 1
                self.shift_count = 0
                self.increment_count = 0
                self.setting_change_count = 0

            else:
                pass

        elif test_flag == 'test':

            if self.reset_count == 0 or self.shift_count == 1 or self.increment_count == 1 or self.setting_change_count == 1:
                if self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 0:
                    buf = self.define_buffer_single()[0]
                elif self.card_mode == 32768 and self.sequence_mode == 0 and self.single_joined == 1:
                    buf = self.define_buffer_single_joined()[0]
                elif self.card_mode == 512 and self.sequence_mode == 0:
                    buf = self.define_buffer_multi()[0]

                self.reset_count = 1
                self.shift_count = 0
                self.increment_count = 0
                self.setting_change_count = 0

            else:
                pass

    def convertion_to_numpy(self, p_array):
        """
        Convertion of the pulse_array into numpy array in the form of
        [channel_number, function, frequency, phase, length, sigma, start, delta_start, mode]
        channel_number is from channel_dict
        function is from function_dict
        frequency is a pulse frequency in MHz
        phase is a pulse phase in rad
        length is a pulse length in sample rate
        sigma is a pulse sigma in sample rate
        start is a pulse start in sample rate forjoined pulses in 'Single'
        delta_start is a pulse delta start in sample rate forjoined pulses in 'Single'

        The numpy array is sorted according to channel number
        """
        if test_flag != 'test':
            i = 0
            pulse_temp_array = []
            num_pulses = len( p_array )
            while i < num_pulses:
                # get channel number
                ch = p_array[i]['channel']
                if ch in channel_dict:
                    ch_num = channel_dict[ch]

                # get function
                fun = p_array[i]['function']
                if fun in function_dict:
                    func = function_dict[fun]
                
                # get length
                leng = p_array[i]['length']

                if leng[-2:] == 'ns':
                    leng_time = int(float(leng[:-3])*self.sample_rate/1000)
                elif leng[-2:] == 'us':
                    leng_time = int(float(leng[:-3])*1000*self.sample_rate/1000)
                elif leng[-2:] == 'ms':
                    leng_time = int(float(leng[:-3])*1000000*self.sample_rate/1000)
                elif leng[-2:] == 's':
                    leng_time = int(float(leng[:-3])*1000000000*self.sample_rate/1000)

                # get frequency
                freq = p_array[i]['frequency']
                freq_mhz = int(float(freq[:-3]))

                # get phase
                phase = float(p_array[i]['phase'])

                # get sigma
                sig = p_array[i]['sigma']

                if sig[-2:] == 'ns':
                    sig_time = int(float(sig[:-3])*self.sample_rate/1000)
                elif sig[-2:] == 'us':
                    sig_time = int(float(sig[:-3])*1000*self.sample_rate/1000)
                elif sig[-2:] == 'ms':
                    sig_time = int(float(sig[:-3])*1000000*self.sample_rate/1000)
                elif sig[-2:] == 's':
                    sig_time = int(float(sig[:-3])*1000000000*self.sample_rate/1000)

                # get start
                st = p_array[i]['start']

                if st[-2:] == 'ns':
                    st_time = int(float(st[:-3])*self.sample_rate/1000)
                elif st[-2:] == 'us':
                    st_time = int(float(st[:-3])*1000*self.sample_rate/1000)
                elif st[-2:] == 'ms':
                    st_time = int(float(st[:-3])*1000000*self.sample_rate/1000)
                elif st[-2:] == 's':
                    st_time = int(float(st[:-3])*1000000000*self.sample_rate/1000)

                # get delta_start
                del_st = p_array[i]['delta_start']

                if del_st[-2:] == 'ns':
                    del_st_time = int(float(del_st[:-3])*self.sample_rate/1000)
                elif del_st[-2:] == 'us':
                    del_st_time = int(float(del_st[:-3])*1000*self.sample_rate/1000)
                elif del_st[-2:] == 'ms':
                    del_st_time = int(float(del_st[:-3])*1000000*self.sample_rate/1000)
                elif del_st[-2:] == 's':
                    del_st_time = int(float(del_st[:-3])*1000000000*self.sample_rate/1000)

                # creating converted array
                pulse_temp_array.append( (ch_num, func, freq_mhz, phase, leng_time, sig_time, st_time, del_st_time) )

                i += 1

            # should be sorted according to channel number for corecct splitting into subarrays
            return np.asarray(sorted(pulse_temp_array, key = lambda x: int(x[0])) ) #, dtype = np.int64 

        elif test_flag == 'test':
            i = 0
            pulse_temp_array = []
            num_pulses = len( p_array )
            while i < num_pulses:
                # get channel number
                ch = p_array[i]['channel']
                if ch in channel_dict:
                    ch_num = channel_dict[ch]

                # get function
                fun = p_array[i]['function']
                if fun in function_dict:
                    func = function_dict[fun]
                
                # get length
                leng = p_array[i]['length']

                if leng[-2:] == 'ns':
                    leng_time = int(float(leng[:-3])*self.sample_rate/1000)
                elif leng[-2:] == 'us':
                    leng_time = int(float(leng[:-3])*1000*self.sample_rate/1000)
                elif leng[-2:] == 'ms':
                    leng_time = int(float(leng[:-3])*1000000*self.sample_rate/1000)
                elif leng[-2:] == 's':
                    leng_time = int(float(leng[:-3])*1000000000*self.sample_rate/1000)

                # get frequency
                freq = p_array[i]['frequency']
                freq_mhz = int(float(freq[:-3]))

                # get phase
                phase = float(p_array[i]['phase'])

                # get sigma
                sig = p_array[i]['sigma']

                if sig[-2:] == 'ns':
                    sig_time = int(float(sig[:-3])*self.sample_rate/1000)
                elif sig[-2:] == 'us':
                    sig_time = int(float(sig[:-3])*1000*self.sample_rate/1000)
                elif sig[-2:] == 'ms':
                    sig_time = int(float(sig[:-3])*1000000*self.sample_rate/1000)
                elif sig[-2:] == 's':
                    sig_time = int(float(sig[:-3])*1000000000*self.sample_rate/1000)

                # get start
                st = p_array[i]['start']

                if st[-2:] == 'ns':
                    st_time = int(float(st[:-3])*self.sample_rate/1000)
                elif st[-2:] == 'us':
                    st_time = int(float(st[:-3])*1000*self.sample_rate/1000)
                elif st[-2:] == 'ms':
                    st_time = int(float(st[:-3])*1000000*self.sample_rate/1000)
                elif st[-2:] == 's':
                    st_time = int(float(st[:-3])*1000000000*self.sample_rate/1000)

                # get delta_start
                del_st = p_array[i]['delta_start']

                if del_st[-2:] == 'ns':
                    del_st_time = int(float(del_st[:-3])*self.sample_rate/1000)
                elif del_st[-2:] == 'us':
                    del_st_time = int(float(del_st[:-3])*1000*self.sample_rate/1000)
                elif del_st[-2:] == 'ms':
                    del_st_time = int(float(del_st[:-3])*1000000*self.sample_rate/1000)
                elif del_st[-2:] == 's':
                    del_st_time = int(float(del_st[:-3])*1000000000*self.sample_rate/1000)

                # creating converted array
                pulse_temp_array.append( (ch_num, func, freq_mhz, phase, leng_time, sig_time, st_time, del_st_time) )

                i += 1

            # should be sorted according to channel number for corecct splitting into subarrays
            return np.asarray(sorted(pulse_temp_array, key = lambda x: int(x[0])), dtype = np.int64) 

    def splitting_acc_to_channel(self, np_array):
        """
        A function that splits pulse array into
        several array that have the same channel
        I.E. [[0, 0, 10, 70, 10, 16], [0, 0, 20, 70, 10, 16]]
        -> [array([0, 0, 10, 70, 10, 16], [0, 0, 20, 70, 10, 16])]
        Input array should be sorted
        """
        if test_flag != 'test':
            # according to 0 element (channel number)
            answer = np.split(np_array, np.where(np.diff(np_array[:,0]))[0] + 1)
            return answer

        elif test_flag == 'test':
            # according to 0 element (channel number)
            try:
                answer = np.split(np_array, np.where(np.diff(np_array[:,0]))[0] + 1)
            except IndexError:
                assert( 1 == 2 ), 'No AWG pulses are defined'

            return answer

    def preparing_buffer_multi(self):
        """
        A function to prepare everything for buffer filling in the 'Multi' mode
        Check of 'Multi' card mode and 'External' trigger mode.
        Check of number of segments.
        Find a pulse with the maximum length. Determine the memory size.
        Return memory_size, number_of_segments, pulses
        """
        if test_flag != 'test':
            
            #checks
            #if self.card_mode == 32768:
            #    general.message('You are not in Multi mode')
            #    sys.exit()
            #if self.trigger_ch == 'Software':
            #    general.message('Segmented memory is available only in External trigger mode')
            #    sys.exit()

            # maximum length and number of segments array
            num_segments_array = []
            max_length_array = []

            pulses = self.splitting_acc_to_channel( self.convertion_to_numpy(self.pulse_array) )

            # Check that number of segments and number of pulses are equal
            for index, element in enumerate(pulses):
                num_segments_array.append( len(element) )
                max_length_array.append( max( element[:,4] ))
            num_segments_array.append( self.num_segments )

            #gr = groupby( num_segments_array )
            #if (next(gr, True) and not next(gr, False)) == False:
            #    assert(1 == 2), 'Number of segments are not equal to the number of AWG pulses'

            # finding the maximum pulse length to create a buffer
            max_pulse_length = max( max_length_array )
            #buffer_per_max_pulse = self.closest_power_of_two( max_pulse_length )
            buffer_per_max_pulse = self.round_to_closest( max_pulse_length , 32 )
            if buffer_per_max_pulse < 32:
                buffer_per_max_pulse = 32
                general.message('Buffer size was rounded to the minimal available value (32 samples)')

            # determine the memory size (buffer size will be doubled if two channels are enabled)
            memsize = buffer_per_max_pulse*num_segments_array[0]

            return memsize, num_segments_array[0], pulses

        elif test_flag == 'test':            
            
            #checks
            assert( self.card_mode == 512 ), 'You are not in Multi mode'
            assert( self.trigger_ch == 2 ), 'Segmented memory is available only in External trigger mode'

            num_segments_array = []
            max_length_array = []

            pulses = self.splitting_acc_to_channel( self.convertion_to_numpy(self.pulse_array) )

            # Check that number of segments and number of pulses
            # are equal
            for index, element in enumerate(pulses):
                num_segments_array.append( len(element) )
                max_length_array.append( max( element[:,4] ))
            num_segments_array.append( self.num_segments )

            gr = groupby( num_segments_array )
            if (next(gr, True) and not next(gr, False)) == False:
                assert(1 == 2), 'Number of segments are not equal to the number of AWG pulses'

            # finding the maximum pulse length to create a buffer
            max_pulse_length = max( max_length_array )
            #buffer_per_max_pulse = self.closest_power_of_two( max_pulse_length )
            buffer_per_max_pulse = self.round_to_closest( max_pulse_length , 32 )
            if buffer_per_max_pulse < 32:
                buffer_per_max_pulse = 32
                general.message('Buffer size was rounded to the minimal available value (32 samples)')

            # check number of channels and determine the memory size 
            # (buffer size will be doubled if two channels are enabled)
            num_ch = len(pulses)
            assert( (num_ch == 1 and ( self.channel == 1 or self.channel == 2)) or \
                num_ch == 2 and ( self.channel == 3 ) ), 'Number of enabled channels are not equal to the number of channels in AWG pulses'

            memsize = buffer_per_max_pulse*num_segments_array[0]

            return memsize, num_segments_array[0], pulses

    def define_buffer_multi(self):
        """
        Define and fill the buffer in 'Multi' mode;
        Full card memory is splitted into number of segments.
        
        If a number of enabled channels is 1:
        In every segment each index is a new data sample
        
        If a number of enabled channels are 2:
        In every segment every even index is a new data sample for CH0,
        every odd index is a new data sample for CH1.
        """

        # all the data
        self.memsize, segments, pulses = self.preparing_buffer_multi()
        self.segment_memsize = int( self.memsize / segments )

        # define buffer differently for only one or two channels enabled
        if self.channel == 1 or self.channel == 2:
            # two bytes per sample; multiply by number of enabled channels
            self.buffer_size = len( np.zeros(2 * self.memsize * 1) ) 
            #buf = np.zeros(self.memsize * 1, dtype = np.float64 )

            # define the buffer
            pnBuffer = c_void_p()
            self.qwBufferSize = uint64 (self.buffer_size)  # buffer size
            pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
            pnBuffer = cast (pvBuffer, ptr16)

            # pulses for different channel
            for element in pulses:
                # individual pulses at each channel
                for index2, element2 in enumerate( element ):
                    # segmented memory; mid_point for GAUSS and SINC
                    mid_point = int( element2[4]/2 )

                    # take a segment: self.segment_memsize*index2, self.segment_memsize*(index2 + 1)
                    for i in range ( self.segment_memsize*index2, self.segment_memsize*(index2 + 1), 1):
                        if element2[1] == 0: # SINE; [channel, function, frequency (MHz), phase, length (samples), sigma (samples)]
                            if i <= ( element2[4] + self.segment_memsize*index2 ):
                                #buf[i]
                                pnBuffer[i] = int(maxCAD * sin(2*pi*(( i - self.segment_memsize*index2))*element2[2] / self.sample_rate + element2[3]) )
                            else:
                                # keeping 0
                                break
                        elif element2[1] == 1: # GAUSS
                            if i <= ( element2[4] + self.segment_memsize*index2 ):
                                pnBuffer[i] = int(maxCAD * exp(-((( i - self.segment_memsize*index2) - mid_point)**2)*(1/(2*element2[5]**2)))*\
                                    sin(2*pi*(( i - self.segment_memsize*index2))*element2[2] / self.sample_rate + element2[3]) )
                            else:
                                break
                        elif element2[1] == 2: # SINC
                            if i <= ( element2[4] + self.segment_memsize*index2 ):
                                pnBuffer[i] = int(maxCAD * np.sinc(2*(( i - self.segment_memsize*index2) - mid_point) / (element2[5]) )*\
                                    sin(2*pi*( i - self.segment_memsize*index2)*element2[2] / self.sample_rate + element2[3]) )
                            else:
                                break
                        elif element2[1] == 3: # BLANK
                            break

            return pvBuffer, pnBuffer

        elif self.channel == 3:
            # two bytes per sample; multiply by number of enabled channels
            self.buffer_size = len( np.zeros(2 * self.memsize * 2) )
            #buf = np.zeros(self.memsize * 2)
            
            # define the buffer
            pnBuffer = c_void_p()
            self.qwBufferSize = uint64 (self.buffer_size)  # buffer size
            pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
            pnBuffer = cast (pvBuffer, ptr16)

            # pulses for different channel
            for element in pulses:
                # individual pulses at each channel
                for index2, element2 in enumerate( element ):
                    # segments memory; mid_point for GAUSS and SINC
                    mid_point = int( element2[4]/2 )
                    
                    if element2[0] == 0:
                        # take a segment: 2*self.segment_memsize*index2, 2*self.segment_memsize*(index2 + 1)
                        # fill even indexes for CH0; step in the cycle is 2
                        for i in range (2 * self.segment_memsize*index2, 2 * self.segment_memsize*(index2 + 1), 2):
                            if element2[1] == 0: # SINE; [channel, function, frequency (MHz), phase, length (samples), sigma (samples)]
                                if i <= (2 * element2[4] + 2 * self.segment_memsize*index2 ):
                                    # Sine Signal CH_0; i/2 in order to keep the frequency
                                    #buf[i] 
                                    pnBuffer[i] = int(maxCAD * sin(2*pi*(( i/2 - self.segment_memsize*index2))*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 1: # GAUSS
                                if i <= ( 2 * element2[4] + 2 * self.segment_memsize*index2 ):
                                    pnBuffer[i] = int(maxCAD * exp(-((( i/2 - self.segment_memsize*index2) - mid_point)**2)*(1/(2*element2[5]**2)))*\
                                        sin(2*pi*(( i/2 - self.segment_memsize*index2))*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 2: # SINC
                                if i <= ( 2 * element2[4] + 2 * self.segment_memsize*index2 ):
                                    pnBuffer[i] = int(maxCAD * np.sinc(2*(( i/2 - self.segment_memsize*index2) - mid_point) / (element2[5]) )*\
                                        sin(2*pi*( i/2 - self.segment_memsize*index2)*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 3: # BLANK
                                break

                    elif element2[0] == 1:
                        # fill odd indexes for CH1; step in the cycle is 2
                        for i in range (1 + 2 * self.segment_memsize*index2, 1 + 2 * self.segment_memsize*(index2 + 1), 2):
                            if element2[1] == 0: # SINE; [channel, function, frequency (MHz), phase, length (samples), sigma (samples)]
                                if (i - 1) <= ( 2 * element2[4] + 2 * self.segment_memsize*index2 ):
                                    # Sine Signal CH_1; i/2 in order to keep the frequency
                                    #buf[i]
                                    pnBuffer[i] = int(maxCAD * sin(2*pi*(( (i - 1)/2 - self.segment_memsize*index2 ))*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 1: # GAUSS
                                if (i - 1) <= ( 2 * element2[4] + 2 * self.segment_memsize*index2 ):
                                    pnBuffer[i] = int(maxCAD * exp(-((( (i - 1)/2 - self.segment_memsize*index2) - mid_point)**2)*(1/(2*element2[5]**2)))*\
                                        sin(2*pi*(( (i - 1)/2 - self.segment_memsize*index2))*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 2: # SINC
                                if (i - 1) <= ( 2 * element2[4] + 2 * self.segment_memsize*index2 ):
                                    pnBuffer[i] = int(maxCAD * np.sinc(2*(( (i - 1)/2 - self.segment_memsize*index2) - mid_point) / (element2[5]) )*\
                                        sin(2*pi*( (i - 1)/2 - self.segment_memsize*index2)*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 3: # BLANK
                                break

            return pvBuffer, pnBuffer

    def preparing_buffer_single(self):
        """
        A function to prepare everything for buffer filling in the 'Single' mode
        Check of 'Single' card mode.
        Check of number of segments (should be  1).
        Find a pulse with the maximum length. Determine the memory size.
        Return memory_size, pulses
        """
        if test_flag != 'test':
            
            max_length_array = []
            pulses = self.splitting_acc_to_channel( self.convertion_to_numpy(self.pulse_array) )

            # Max pulse length
            for index, element in enumerate(pulses):
                max_length_array.append( max( element[:,4] ))

            # finding the maximum pulse length to create a buffer
            max_pulse_length = max( max_length_array )
            #buffer_per_max_pulse = self.closest_power_of_two( max_pulse_length )
            buffer_per_max_pulse = self.round_to_closest( max_pulse_length , 32 )
            if buffer_per_max_pulse < 32:
                buffer_per_max_pulse = 32
                general.message('Buffer size was rounded to the minimal available value (32 samples)')

            # determine the memory size (buffer size will be doubled if two channels are enabled)
            memsize = buffer_per_max_pulse

            return memsize, pulses

        elif test_flag == 'test':

            # checks
            assert( self.card_mode == 32768 ), 'You are not in Single mode'

            max_length_array = []
            pulses = self.splitting_acc_to_channel( self.convertion_to_numpy(self.pulse_array) )

            if self.single_joined == 0:
                if self.channel == 1:
                    assert( len(self.pulse_ch0_array) == 1 and len(self.pulse_ch1_array) == 0 ), "Only one CH0 pulse can be defined in Single mode with only CH0 enabled"
                elif self.channel == 2:
                    assert( len(self.pulse_ch1_array) == 1 and len(self.pulse_ch0_array) == 0 ), "Only one CH1 pulse can be defined in Single mode with only CH1 enabled"
                elif self.channel == 3:
                    assert( len(self.pulse_ch0_array) == 1 and len(self.pulse_ch1_array) == 1 ), "Only one CH1 and CH0 pulses can be defined in Single mode with CH0 and CH1 enabled"
            elif self.single_joined == 1:
                if self.channel == 1:
                    assert( len(self.pulse_ch0_array) > 1 and len(self.pulse_ch1_array) == 0 ), "At least one CH0 pulse should be defined in Single Joined mode with only CH0 enabled"
                elif self.channel == 2:
                    assert( len(self.pulse_ch1_array) > 1 and len(self.pulse_ch0_array) == 0 ), "At least one CH1 pulse should be defined in Single Joined mode with only CH1 enabled"
                elif self.channel == 3:
                    assert( len(self.pulse_ch0_array) > 0 and len(self.pulse_ch1_array) == 0 ), "Only CH0 pulses should be defined in Single Joined mode with CH0 and CH1 enabled. CH1 pulses are filled automatically"                

            assert( self.num_segments == 1 ), 'More than one segment is declared in Single mode. Please, use Multi mode'

            # finding the maximum pulse length to create a buffer
            for index, element in enumerate(pulses):
                max_length_array.append( max( element[:,4] ))

            max_pulse_length = max( max_length_array )
            #buffer_per_max_pulse = self.closest_power_of_two( max_pulse_length )
            buffer_per_max_pulse = self.round_to_closest( max_pulse_length , 32 )
            if buffer_per_max_pulse < 32:
                buffer_per_max_pulse = 32
                general.message('Buffer size was rounded to the minimal available value (32 samples)')

            # check number of channels and determine the memory size 
            # (buffer size will be doubled if two channels are enabled)
            #num_ch = len(pulses)
            #assert( (num_ch == 1 and ( self.channel == 1 or self.channel == 2)) or \
            #    num_ch == 2 and ( self.channel == 3 ) ), 'Number of enabled channels are not equal to the number of channels in AWG pulses'

            memsize = buffer_per_max_pulse

            return memsize, pulses

    def define_buffer_single(self):
        """
        Define and fill the buffer in 'Single' mode;
        
        If a number of enabled channels is 1:
        Each index is a new data sample
        
        If a number of enabled channels are 2:
        Every even index is a new data sample for CH0,
        Every odd index is a new data sample for CH1.
        """

        self.memsize, pulses = self.preparing_buffer_single()

        # define buffer differently for only one or two channels enabled
        if self.channel == 1 or self.channel == 2:
            # two bytes per sample; multiply by number of enabled channels
            self.buffer_size = len( np.zeros(2 * self.memsize * 1) )
            #buf = np.zeros(self.memsize * 1, dtype = np.float64 )

            # define the buffer
            pnBuffer = c_void_p()
            self.qwBufferSize = uint64 (self.buffer_size)  # buffer size
            pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
            pnBuffer = cast (pvBuffer, ptr16)

            # pulses for different channel
            for element in pulses:
                # individual pulses at each channel
                for index2, element2 in enumerate( element ):
                    # mid_point for GAUSS and SINC
                    mid_point = int( element2[4]/2 )

                    for i in range (0, self.memsize, 1):
                        if element2[1] == 0: # SINE; [channel, function, frequency (MHz), phase, length (samples), sigma (samples)]
                            if i <= ( element2[4] ):
                                #buf[i]
                                pnBuffer[i] = int( maxCAD * sin(2*pi*(( i ))*element2[2] / self.sample_rate + element2[3]) )
                            else:
                                break
                        elif element2[1] == 1: # GAUSS
                            if i <= ( element2[4] ):
                                pnBuffer[i] = int( maxCAD * exp(-((( i ) - mid_point)**2)*(1/(2*element2[5]**2)))*\
                                    sin(2*pi*(( i ))*element2[2] / self.sample_rate + element2[3]) )
                            else:
                                break
                        elif element2[1] == 2: # SINC
                            if i <= ( element2[4] ):
                                pnBuffer[i] = int( maxCAD * np.sinc(2*(( i ) - mid_point) / (element2[5]) )*\
                                    sin(2*pi*( i )*element2[2] / self.sample_rate + element2[3]) )
                            else:
                                break
                        elif element2[1] == 3: # BLANK
                            break

            return pvBuffer, pnBuffer

        elif self.channel == 3:
            # two bytes per sample; multiply by number of enabled channels
            self.buffer_size = len( np.zeros(2 * self.memsize * 2) )
            #buf = np.zeros(self.memsize * 2, dtype = np.float64 )

            # define the buffer
            pnBuffer = c_void_p()
            self.qwBufferSize = uint64 (self.buffer_size)  # buffer size
            pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
            pnBuffer = cast (pvBuffer, ptr16)

            # pulses for different channel
            for element in pulses:
                # individual pulses at each channel
                for index2, element2 in enumerate( element ):
                    # mid_point for GAUSS and SINC
                    mid_point = int( element2[4]/2 )
                    
                    if element2[0] == 0:
                        # fill even indexes for CH0; step in the cycle is 2
                        for i in range (0, 2 * self.memsize, 2):
                            if element2[1] == 0: # SINE; [channel, function, frequency (MHz), phase, length (samples), sigma (samples)]
                                if i <= ( 2 * element2[4] ):
                                    # Sine Signal CH_0; i/2 in order to keep the frequency
                                    #buf[i]
                                    pnBuffer[i] = int( maxCAD * sin(2*pi*(( i/2 ))*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 1: # GAUSS
                                if i <= ( 2 * element2[4] ):
                                    pnBuffer[i] = int( maxCAD * exp(-((( i/2 ) - mid_point)**2)*(1/(2*element2[5]**2)))*\
                                        sin(2*pi*(( i/2 ))*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 2: # SINC
                                if i <= ( 2 * element2[4] ):
                                    pnBuffer[i] = int( maxCAD * np.sinc(2*(( i/2 ) - mid_point) / (element2[5]) )*\
                                        sin(2*pi*( i/2 )*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 3: # BLANK
                                break

                    elif element2[0] == 1:
                        # fill odd indexes for CH1; step in the cycle is 2
                        for i in range (1 , 1 + 2 * self.memsize, 2):
                            if element2[1] == 0: # SINE; [channel, function, frequency (MHz), phase, length (samples), sigma (samples)]
                                if (i - 1) <= ( 2 * element2[4] ):
                                    # Sine Signal CH_1; i/2 in order to keep the frequency
                                    #buf[i]
                                    pnBuffer[i] = int( maxCAD * sin(2*pi*(( (i - 1)/2 ))*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 1: # GAUSS
                                if (i - 1) <= ( 2 * element2[4] ):
                                    pnBuffer[i] = int( maxCAD * exp(-((( (i - 1)/2 ) - mid_point)**2)*(1/(2*element2[5]**2)))*\
                                        sin(2*pi*(( (i - 1)/2 ))*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 2: # SINC
                                if (i - 1) <= ( 2 * element2[4] ):
                                    pnBuffer[i] = int( maxCAD * np.sinc(2*(( (i - 1)/2 ) - mid_point) / (element2[5]) )*\
                                        sin(2*pi*( (i - 1)/2 )*element2[2] / self.sample_rate + element2[3]) )
                                else:
                                    break
                            elif element2[1] == 3: # BLANK
                                break

            return pvBuffer, pnBuffer

    def closest_power_of_two(self, x):
        """
        A function to round card memory or sequence segments
        """
        return int( 2**int(log2(x - 1) + 1 ) )

    def write_seg_memory(self, hCard, dwStepIndex, dwStepNextIndex, dwSegmentIndex, dwLoops, dwFlags):
        """
        Function for setting up the sequence memory
        Page 104 of the programming guide of the AWG card
        """
        if test_flag != 'test':
            qwSequenceEntry = uint64 (0)

            # setup register value
            qwSequenceEntry = (dwFlags & ~SPCSEQ_LOOPMASK) | (dwLoops & SPCSEQ_LOOPMASK)
            qwSequenceEntry <<= 32
            qwSequenceEntry |= ((dwStepNextIndex << 16)& SPCSEQ_NEXTSTEPMASK) | (int(dwSegmentIndex) & SPCSEQ_SEGMENTMASK)

            dwError = spcm_dwSetParam_i64 (hCard, SPC_SEQMODE_STEPMEM0 + dwStepIndex, int64(qwSequenceEntry))

        elif test_flag == 'test':
            pass

    def define_buffer_single_joined(self):
        """
        Define and fill the buffer in 'Single Joined' mode;
        
        Every even index is a new data sample for CH0,
        Every odd index is a new data sample for CH1.

        Second channel will be filled automatically
        """
        # pulses are in a form [channel_number, function, frequency, phase, length, sigma, start, delta_start, mode] 
        #                      [0,               1,          2,      3,      4,      5,      6,      7,        8   ]
        self.memsize, pulses = self.preparing_buffer_single() # 0.2-0.3 ms
        # only ch0 pulses are analyzed
        # ch1 will be generated automatically with shifted phase
        # approach is similar to awg_pulse_sequence()
        arguments_array = [[], [], [], [], [], [], []]
        for element in pulses[0]:
            # collect arguments in special array for further handling
            arguments_array[0].append(int(element[1]))     #   0    type; 0 is SINE; 1 is GAUSS; 2 is SINC
            arguments_array[1].append(element[6])          #   1    start
            arguments_array[2].append(element[7])          #   2    delta_start
            arguments_array[3].append(element[4])          #   3    length
            arguments_array[4].append(element[3])          #   4    phase
            arguments_array[5].append(element[5])          #   5    sigma
            arguments_array[6].append(element[2])          #   6    frequency

        # TO DO
        # overlapping check
        # It will be done by pulse_programmer RECT_AWG pulses
        #general.message( np.diff(arguments_array[1]) )
        #if min(np.diff(arguments_array[1])) < 0:
        #    general.message('Overlapped pulses')
        
        #general.message(arguments_array)

        # convert everything in samples 
        pulse_phase_np = np.asarray(arguments_array[4])
        pulse_start_smp = (np.asarray(arguments_array[1])).astype('int64')
        pulse_delta_start_smp = (np.asarray(arguments_array[2])).astype('int64')
        pulse_length_smp = (np.asarray(arguments_array[3])).astype('int64')
        pulse_sigma_smp = np.asarray(arguments_array[5])
        pulse_frequency = np.asarray(arguments_array[6])
        
        # we do not expect that pulses will have drastically different
        # pulse length. It is better to define the same buffer length 
        # for each segment, since in this case we should not worry about
        # saving the information about memory size for each segments
        last_pulse_length = pulse_length_smp[-1]
        last_start = pulse_start_smp[-1]

        #general.message(last_start)
        # buffer length defines for the largest delay
        #self.memsize = self.closest_power_of_two( last_start + last_pulse_length )
        self.memsize = self.round_to_closest( (last_start + last_pulse_length) , 32)

        # define buffer differently for only one or two channels enabled
        # for ch1 phase is automatically shifted by phase_shift_ch1_seq_mode
        if (self.channel == 1 or self.channel == 2) and self.single_joined == 1:

            # define the buffer
            pnBuffer = c_void_p()
            self.qwBufferSize = uint64 (2 * self.memsize)  # buffer size
            pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
            pnBuffer = cast (pvBuffer, ptr16)

            # run over defined pulses inside a sequence point
            for index, element in enumerate(arguments_array[0]):
                # for DEER pulse with random phase
                #rnd_phase = 2*pi*random.random()

                if element == 0: # 'SINE'
                    for i in range(pulse_start_smp[index], self.memsize):

                            if i < (pulse_start_smp[index] ):
                                pass
                            elif ( i >= ( pulse_start_smp[index] ) and \
                                   i <= (pulse_start_smp[index] + pulse_length_smp[index] ) ):
                                
                                #if pulse_phase_np[index] != 1000:
                                pnBuffer[i] = int( maxCAD * sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )

                                #else:
                                    #self.full_buffer[point, i] 
                                #    pnBuffer[i] = int( maxCAD * sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + rnd_phase) )
                            else:
                                break

                elif element == 1: # GAUSS
                    # mid_point for GAUSS and SINC
                    mid_point = int( pulse_start_smp[index] + ( pulse_length_smp[index] )/2 )

                    for i in range(pulse_start_smp[index], self.memsize):
                            if i < (pulse_start_smp[index] ):
                                pass
                            elif ( i >= ( pulse_start_smp[index] ) and \
                                   i <= (pulse_start_smp[index] + pulse_length_smp[index] ) ):
                                
                                #if pulse_phase_np[index] != 1000:
                                    #self.full_buffer[point, i] 
                                pnBuffer[i] = int( maxCAD * exp(-((( i ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                                            sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )

                                #else:
                                    #self.full_buffer[point, i] 
                                #    pnBuffer[i] = int( maxCAD * exp(-((( i ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                #                            sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                #                            rnd_phase) )
                            else:
                                break

                elif element == 2: # 'SINC'
                    # mid_point for GAUSS and SINC
                    mid_point = int( pulse_start_smp[index] + ( pulse_length_smp[index] )/2 )

                    for i in range(pulse_start_smp[index], self.memsize):
                            if i < (pulse_start_smp[index]):
                                pass
                            elif ( i >= ( pulse_start_smp[index] ) and \
                                   i <= (pulse_start_smp[index] + pulse_length_smp[index] ) ):
                                
                                #if pulse_phase_np[index] != 1000:
                                    #self.full_buffer[point, i] 
                                pnBuffer[i] = int( maxCAD * np.sinc(2*(( i ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                                            sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )

                                #else:
                                    #self.full_buffer[point, i]
                                #    pnBuffer[i] = int( maxCAD * np.sinc(2*(( i ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                #                            sin(2*pi*(( i ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                #                            rnd_phase) )
                            else:
                                break

                elif element == 3: # BLANK
                    pass

            return pvBuffer, pnBuffer


        elif self.channel == 3 and self.single_joined == 1:
            # 1-2 ms per pulse
            # define the buffer
            pnBuffer = c_void_p()
            self.qwBufferSize = uint64 (2 * self.memsize * 2)  # buffer size for two channels
            pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
            pnBuffer = cast (pvBuffer, ptr16)

            # run over defined pulses inside a sequence point
            for index, element in enumerate(arguments_array[0]):
                # DEER pulse
                #rnd_phase = 2*pi*random.random()

                if element == 0: #'SINE'
                    for i in range(2 * (pulse_start_smp[index] ), 2*self.memsize, 2):
                            if i < 2 * (pulse_start_smp[index] ):
                                pass
                            elif ( i >= 2 * ( pulse_start_smp[index] ) and \
                                   i <= 2 * (pulse_start_smp[index] + pulse_length_smp[index] ) ):
                                
                                #if pulse_phase_np[index] != 1000:
                                    #self.full_buffer[point, i] 
                                # ch0
                                pnBuffer[i] = int( maxCAD * sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )
                                #ch1
                                pnBuffer[i + 1] = int( maxCAD * sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                                             phase_shift_ch1_seq_mode) )

                                #else:
                                    #self.full_buffer[point, i]
                                    # ch0
                                    #pnBuffer[i] = int( maxCAD * sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + rnd_phase) )
                                    #ch1
                                    #pnBuffer[i + 1] = int( maxCAD * sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + rnd_phase + \
                                    #                         phase_shift_ch1_seq_mode) )
                            else:
                                break

                elif element == 1: #'GAUSS'
                    # mid_point for GAUSS and SINC
                    mid_point = int( pulse_start_smp[index] + (pulse_length_smp[index] )/2 )

                    for i in range(2 * (pulse_start_smp[index] ), 2*self.memsize, 2):

                            if i < 2 * (pulse_start_smp[index] ):
                                pass
                            elif ( i >= 2 * ( pulse_start_smp[index] ) and \
                                   i <= 2 * (pulse_start_smp[index] + pulse_length_smp[index] ) ):
                                
                                #if pulse_phase_np[index] != 1000:
                                    #self.full_buffer[point, i]
                                # ch0
                                pnBuffer[i] = int( maxCAD * exp(-((( i/2 ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                                            sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )
                                #ch1
                                pnBuffer[i + 1] = int( maxCAD * exp(-((( ( i )/2 ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                                            sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                                            phase_shift_ch1_seq_mode) )

                                #else:
                                    #self.full_buffer[point, i]
                                    # ch0
                                #    pnBuffer[i] = int( maxCAD * exp(-((( i/2 ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                #                            sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + rnd_phase ) )
                                    #ch1
                                #    pnBuffer[i + 1] = int( maxCAD * exp(-((( ( i )/2 ) - mid_point)**2)*(1/(2*pulse_sigma_smp[index]**2)))*\
                                #                            sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + rnd_phase +\
                                #                            phase_shift_ch1_seq_mode) )                                    

                            else:
                                break
                    
                elif element == 2: #'SINC'
                    # mid_point for GAUSS and SINC
                    mid_point = int( pulse_start_smp[index] + (pulse_length_smp[index])/2 )

                    for i in range(2 * (pulse_start_smp[index] ), 2*self.memsize, 2):

                            if i < 2 * (pulse_start_smp[index] ):
                                pass
                            elif ( i >= 2 * ( pulse_start_smp[index] ) and \
                                   i <= 2 * (pulse_start_smp[index] + pulse_length_smp[index] ) ):
                                
                                #if pulse_phase_np[index] != 1000:
                                    #self.full_buffer[point, i]
                                # ch0
                                pnBuffer[i] = int( maxCAD * np.sinc(2*(( i/2 ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                                             sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] ) )
                                #ch1
                                pnBuffer[i + 1] = int( maxCAD * np.sinc(2*(( ( i  )/2 ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                                             sin(2*pi*(( ( i  )/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                                             phase_shift_ch1_seq_mode) )
                                #else:
                                    #self.full_buffer[point, i]
                                    # ch0
                                #    pnBuffer[i] = int( maxCAD * np.sinc(2*(( i/2 ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                #                             sin(2*pi*(( i/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] + rnd_phase) )
                                #    pnBuffer[i + 1] = int( maxCAD * np.sinc(2*(( ( i )/2 ) - mid_point) / (pulse_sigma_smp[index]) )*\
                                #                             sin(2*pi*(( ( i )/2 ))*pulse_frequency[index] / self.sample_rate + pulse_phase_np[index] +\
                                #                             phase_shift_ch1_seq_mode + rnd_phase) )
                            else:
                                break
                
                elif element == 3: # BLANK
                    pass

            return pvBuffer, pnBuffer
    
    def round_to_closest(self, x, y):
        """
        A function to round x to divisible by y
        """
        return int( y * ( ( x // y) + (x % y > 0) ) )

def main():
    pass

if __name__ == "__main__":
    main()
