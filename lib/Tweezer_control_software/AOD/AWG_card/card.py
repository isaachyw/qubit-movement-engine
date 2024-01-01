# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:21:22 2021
Revised on Sun Jul 17 11:39:00 2022

@author: Weijun Yuan, Chun-Wei Liu

Notes
-----
This is the AWG card (M4i.6622-x8)
"""
# Std imports
import sys
import time
import ctypes

# Spectrum AWG card lib
from lib.Tweezer_control_software.AOD.AWG_card.pyspcm import *
from lib.Tweezer_control_software.AOD.AWG_card.spcm_tools import *

# Numerical imports
import math
import numpy as np

# Misc imports
import h5py
from tqdm import tqdm
import msvcrt
from queue import Queue


class AtomRearrange_REP_STD_SEQUENCE_card(object):
    """
    The 2D atom rearranging controller with sequence replay mode. 

    Notes
    -----
    Process for rearranging:
        1. Initialize the card before use. 
        2. Load the precalculated data computed corresponding to the lattice graph
        3. Wait for software trigger (from GUI or after taken Andor initial image) to begin Rearrange
        4. Rearranger will setup card registers based on Sequence Replay Mode, state, mem_seg, loop_num etc.
        5. Start the card and wait for all sequence are played.
        4. After the rearranging is done, send a software trigger back to Andor camer for final image shoot.
    
    """
    def __init__(self, address = b'/dev/spcm0', channelNum = 4, sampleRate=625):
        ## This part defines the parameters of the card
        """

        Parameters
        ----------
        address :The address of the card. The default is b'/dev/spcm0'.
        channelNum : The numnber of channels. The default is 4.
        sampleRate : sample rate in Mbyte/s. The default is 625.
        Returns
        -------
        None.

        Note
        ----
        spcm_dwSetParam(device, register, value) : Reserve software registers for setting hardware commands.
        
        spcm_dwGetParam(device, register, value) : Reserve software registers for reading hardware commands.
        """

        ## open the card
        self.hCard = spcm_hOpen (create_string_buffer(address))
        self.lCardType = int32(0)
        spcm_dwGetParam_i32 (self.hCard, SPC_PCITYP, byref(self.lCardType))
        self.lSerialNumber = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_PCISERIALNO, byref(self.lSerialNumber))
        self.lFncType = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_FNCTYPE, byref(self.lFncType))

        ## Check if the card itself is valid
        Valid = self.checkCard()
        if Valid == False:
            exit()
        ## set the sample rate to 625MS/s
        ## MEGA here means 1e6
        self.SampleRate = MEGA(sampleRate)

        ## set samplerate to 1 MHz (M2i) or 50 MHz, no clock output
        if ((self.lCardType.value & TYP_SERIESMASK) == TYP_M4IEXPSERIES) or ((self.lCardType.value & TYP_SERIESMASK) == TYP_M4XEXPSERIES):
            spcm_dwSetParam_i64 (self.hCard, SPC_SAMPLERATE, self.SampleRate)
        else:
            spcm_dwSetParam_i64 (self.hCard, SPC_SAMPLERATE, MEGA(1))
        spcm_dwSetParam_i32 (self.hCard, SPC_CLOCKOUT,   0)

        ## setup the mode
        ## In this case, sequence replay mode
        self.llChEnable = int64 (CHANNEL0|CHANNEL1) # select channel 0 and channel 1
        
        # TODO: begin a more careful check on the relation between number of segments and data
        ## Refer to UCB/card.pt line: 424
        lMaxSegments = int32 (2**7) # pure_static: **10 
        spcm_dwSetParam_i32 (self.hCard, SPC_CARDMODE,            SPC_REP_STD_SEQUENCE)
        spcm_dwSetParam_i64 (self.hCard, SPC_CHENABLE,            self.llChEnable)
        spcm_dwSetParam_i32 (self.hCard, SPC_SEQMODE_MAXSEGMENTS, lMaxSegments)
        spcm_dwSetParam_i32 (self.hCard, SPC_SEQMODE_STARTSTEP, 0)

        ## setup trigger
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_ORMASK,      SPC_TMASK_SOFTWARE) # software trigger
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_ANDMASK,     0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ORMASK0,  0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ORMASK1,  0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ANDMASK0, 0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ANDMASK1, 0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIGGEROUT,       0)


        ## setup the channels
        self.lNumChannels = int32(0)
        spcm_dwGetParam_i32 (self.hCard, SPC_CHCOUNT, byref(self.lNumChannels))
        for lChannel in range (0, self.lNumChannels.value, 1):
            ## Enable output with 1, disabl output with 0
            spcm_dwSetParam_i32(self.hCard, SPC_ENABLEOUT0    + lChannel * (SPC_ENABLEOUT1    - SPC_ENABLEOUT0),    1)
            ## Set up amplifier
            spcm_dwSetParam_i32(self.hCard, SPC_AMP0          + lChannel * (SPC_AMP1          - SPC_AMP0),          242)
            ## Pause after replay
            spcm_dwSetParam_i32(self.hCard, SPC_CH0_STOPLEVEL + lChannel * (SPC_CH1_STOPLEVEL - SPC_CH0_STOPLEVEL), SPCM_STOPLVL_HOLDLAST)

        ## For voltage output control, consault FIFO mode.
        ## Setup all channels
        ## Improve the way to select the channel later
        ## The actual ouput power measurement from AWG ouutput P_F ~ (alpha/N)*|V_awg|**2
        self.volt_lv = [int(150),int(150)]
        # setup all channels
        ## improve the way to select the channel later
        for i in range (0, self.lNumChannels.value):
            spcm_dwSetParam_i32 (self.hCard, SPC_AMP0 + i * (SPC_AMP1 - SPC_AMP0), int32(self.volt_lv[i]))
            spcm_dwSetParam_i32 (self.hCard, SPC_ENABLEOUT0 + i * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0),  int32(1))
        
        ## Some misc parameters
        ## To trigger the shutters in the front of AOD.
        self.USING_EXTERNAL_TRIGGER  = False 

    def checkCard(self):
        """
        Function that checks if the card used is indeed an M4i.6622-x8 or is compatible with Analog Output.
        copied from the Toronto's group

        """

        ## Check if Card is connected
        if self.hCard == None:
            print("no card found...\n")
            return False

        ## Getting the card Name to check if it's supported.
        try:

            self.sCardName = szTypeToName (self.lCardType.value)
            if self.lFncType.value == SPCM_TYPE_AO:
                print("Found card: {0} with serial number {1:05d}\n".format(self.sCardName,self.lSerialNumber.value))
                return True
            else:
                print("Code is for an M4i.6622 Card.\nCard: {0} with serial number {1:05d} is not supported.\n".format(self.sCardName,self.lSerialNumber.value))
                return False

        except:
            dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
            print(dwError)
            print("Problem occured, mb")

    def vWriteSegmentData (self, lNumActiveChannels, dwSegmentIndex, dwSegmentLenSample, pvSegData):
        """
        Write the specific data to specific segment memory (from PC to card)under the sequence replay mode scheme.
        
        Parameters
        ----------
        lNumActiveChannels : integer
            the number of the channels activated. For cross AOD, it is two
        dwSegmentIndex :  integer
            the segment index for a specific data segement
        dwSegmentLenSample : integer
            the length of the sample required for the data segement
        pvSegData : pv buffer
            the pv buffer allocated for data transfer
        
        Returns
        -------
        None.

        Note
        ----
        spcm_dwDefTransfer_i64 & spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA): 
            Are the actual command that take the data pointed by the pvBuffer pointer.

        And every time we want to know if the process is valid, we just call the dwError value.
        """
        self.lBytesPerSample = 2
        self.dwSegLenByte = uint32 (dwSegmentLenSample * self.lBytesPerSample * lNumActiveChannels.value)

        ## seg_memory should be rounded to 32, which means repetition rate should be divisible by 32
        ## The maximum value is bounded to SPC_SEQMODE_MAXSEGMENTS
        ## And the min of 2 channel should be 192
        if dwSegmentLenSample < 192:
            seg_memory = 192
        else:
            seg_memory = int(32 * math.ceil((float(dwSegmentLenSample) - 192) / 32) + 192) 

        ## setup
        ## SPC_SEQMODE_WRITESEGMENT : switch to the assigned segment
        dwError = spcm_dwSetParam_i32 (self.hCard, SPC_SEQMODE_WRITESEGMENT, int(dwSegmentIndex))
        
        if dwError != ERR_OK:
            raise ValueError(f'Error: {dwError}, Cannot write data into segment {int(dwSegmentIndex)} since SPC_SEQMODE_WRITESEGMENT is invalid.')
        
        ## Set up segment size (signal size)
        if dwError == ERR_OK:
            dwError = spcm_dwSetParam_i32 (self.hCard, SPC_SEQMODE_SEGMENTSIZE, seg_memory)
            
            ## Checker for seg_memory
            if dwError != ERR_OK:
                raise ValueError(f'Error: {dwError}, seg_memory: {seg_memory} for SPC_SEQMODE_SEGMENTSIZE should be rounded to 32, and the maximum value is bounded to SPC_SEQMODE_MAXSEGMENTS')
        
        ## write data to board (main) sample memory
        if dwError == ERR_OK:
            dwError = spcm_dwDefTransfer_i64 (self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, 0, pvSegData, 0, self.dwSegLenByte)

        if dwError == ERR_OK:
            ## Execute command
            dwError = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        
    def vDoDataCalculation (self, lCardType, lNumActiveChannels, lMaxDACValue, load_mode):
        """
        The signal data is precalculated and stored in HDF5 file at specific location. 
        Here the function will take data and allocate them to specific segements.

        Parameters
        ----------
        lCardType : long?
            Card type argument. Kept here due to histrical reason.
        lNumActiveChannels : long?
            The number of activated channels, the actual number can be extractd by lNumActiveChannels.value.
        lMaxDACValue : long?
            The maximum allowed voltage output value of the AWG card.
        mode : str
            The card mode that control the signal mode loaded into card, can either be static or rearranging
        Note
        ----
        a. In the new version, the precalculated data is a 2D data already. And from the dataset, we need to embeded segment index.
        b. Memory buffers will enhence the efficiency of data transfer about 10~15%.
        And in this example, we load the precalculated data into PC memory buffer through pointer array.
        Then write the data (loaded in the buffer) into board memory segments.
        """
        dwSegmentLenSample = uint32 (0)

        ## Allocate memory buffer
        ## In UCB example, int(2E5) = NUMPY_MAX. The allocated size can not exceed the pc memory
        # TODO: write a constant.py

        ## helper values: Full Scale, the maximum allowed DAC voltage output.
        dwFS = uint32(lMaxDACValue.value)
        dwFShalf = uint32 (dwFS.value // 2)

        for signal_mode in ["static", "grab", "drop", "move"]:
            sys.stdout.write (f"Loading {signal_mode} data from PC to card...\n")
            
            signals = list(self.precal_data[signal_mode]["signal"]) # signals is a 2xt np.array
            
            for signal in tqdm(signals):
                signal_dataset = self.precal_data[signal_mode]["signal"][signal][()]
                dwSegmentLenSample = signal_dataset.shape[1] # the size of t

                ## Allocate memory buffer
                pvBuffer = pvAllocMemPageAligned(2 * 2 * dwSegmentLenSample * lNumActiveChannels.value)
                ## Pointer to the memory buffer
                pnData = cast(addressof(pvBuffer), ptr16) # ptr referes to pointer
                
                if dwSegmentLenSample == 0:
                    raise ValueError(f'In valid signal data, this is a empty dataset.')

                ## The loading loop
                for t in range(0, dwSegmentLenSample):
                    for ch in range(0, lNumActiveChannels.value):
                        pnData[t * lNumActiveChannels.value + ch] = int16(int(dwFS.value * signal_dataset[ch][t]))

                ## Write data into memory
                self.vWriteSegmentData(lNumActiveChannels, self.precal_data[signal_mode]["index"][signal][()], dwSegmentLenSample, pvBuffer)        
            
            ## Breakpoint when we only want to load static signal to the card (save time)
            if load_mode == 'static':
                break
            elif load_mode == 'rearranging':
                pass
                       
    def static(self):
        """
        Setup static array in sequence replay mode.
        
        Note
        ----
        Working.
        """
        spcm_dwSetParam_i32 (self.hCard, SPC_SEQMODE_STARTSTEP, 0)
        self.__vWriteStepEntry(0,  0, self.precal_data["static"]["index"]["static"][()],  1,  SPCSEQ_END)
        
    def Rearrange_test_1D(self, path):
        """
        A test for a short segment of path (1D), meant for illustrate change sequence during card running.

        Note
        ----
        Working.
        """
        ## Tell the card to start from step 0
        spcm_dwSetParam_i32 (self.hCard, SPC_SEQMODE_STARTSTEP, 0)

        ## Writing sequence
        for move_index, moves in enumerate(path):
            memory_seg = int(self.precal_data["move"]["index"][f"({tuple(moves[0])}, {tuple(moves[1])})"][()])
            self.__vWriteStepEntry(0,  0, memory_seg,  1,  SPCSEQ_ENDLOOPALWAYS)
        
    def Rearrange_test_2D(self, path):
        """
        A test for a short segment of path, meant for illustrate change sequence during card running.

        Note
        ----
        In this test example, the sequence never ends, if one wish to

        """
        ## Tell the card to start from step 0
        spcm_dwSetParam_i32 (self.hCard, SPC_SEQMODE_STARTSTEP, 0)

        ## Writing sequence
        step = 0
        for move_index, moves in enumerate(path):
            for site_index, site in enumerate(moves):
                 ## Idle atom, do nothing
                if (len(moves)) == 2 and (moves[0] == moves[-1]):
                    break

                if site_index == len(moves) - 1:
                    break ## Break this move since it's the last moving and we are not playing grab/drop
                else:
                    memory_seg = int(self.precal_data["move"]["index"][f"({tuple(moves[site_index])}, {tuple(moves[site_index + 1])})"][()])
                    print(f'current trap: {moves[site_index]}')
                    print(f'destination trap: {moves[site_index + 1]}')
                    print(f'm_seg: {memory_seg}\n')

                    if (move_index == len(path) - 1) and (site_index == len(moves) - 2):
                        self.__vWriteStepEntry(step,  0, memory_seg,  1000,  SPCSEQ_ENDLOOPALWAYS)

                    else:
                        self.__vWriteStepEntry(step,  step + 1, memory_seg,  1000,  SPCSEQ_ENDLOOPALWAYS)
                        time.sleep(3)
                        step + 1
        
    def Rearrange(self, path):
        """
        The function that define the order of the moving sequence based on path data in frequency graph.

        Parameters
        ----------
        path : The non-collision path in freq basis.

        Returns
        -------
        None.

        Note
        ----
        self.__vWriteStepEntry will setup the sequence registers.

        # TODO: Rewrite the data loader from here.

        """
        ## It is safer to reset current state back to 0
        spcm_dwSetParam_i32 (self.hCard, SPC_SEQMODE_STARTSTEP, 0)

        step = 0
        replay_loop_time = 1000
        for move_index, moves in enumerate(path):
            for site_index, site in enumerate(moves):
                ## Idle atom, do nothing
                if (len(moves)) == 2 and (moves[0] == moves[-1]):
                    ## Breaking the move loop
                    break

                ## Playing grab signals
                if site_index == 0:
                    print(f"site:{site}")
                    print(f'data: {self.precal_data["grab"]["index"][f"{tuple(site)}"][()]}')
                    memory_seg = int(self.precal_data["grab"]["index"][f"{tuple(site)}"][()])
                    print(f'grab: {memory_seg}')
                    self.__vWriteStepEntry(step,  step + 1, memory_seg,  replay_loop_time,  SPCSEQ_ENDLOOPALWAYS)
                    step += 1
                
                ## Playing drop signals
                elif site_index == len(moves) - 1:
                    memory_seg = int(self.precal_data["drop"]["index"][f"{tuple(site)}"][()])
                    print(f'drop: {memory_seg}')
                    
                    ## If at the end of the path, end the replay sequence (card)
                    if (move_index == len(path) - 1):
                        self.__vWriteStepEntry(step, 0, memory_seg,  replay_loop_time,  SPCSEQ_END)
                    
                    ## Otherwise, enter the next state
                    else:
                        self.__vWriteStepEntry(step, step + 1, memory_seg,  replay_loop_time,  SPCSEQ_ENDLOOPALWAYS)
                        step += 1
                    
                    ## Since it's the end of a move, breaking the move loop
                    break
                
                ## Playing move signals
                memory_seg = int(self.precal_data["move"]["index"][f"({tuple(moves[site_index])}, {tuple(moves[site_index + 1])})"][()])
                print(f'Move: {memory_seg}')
                self.__vWriteStepEntry(step,  step + 1, memory_seg,  replay_loop_time,  SPCSEQ_ENDLOOPALWAYS)
                step += 1
        
        # print("Starting the card\n")
        ## Star the card
        ## Wait for trigger
        ## Wait for the rearranging process
        # spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY) # ERR_SETUP = 0x010B = 267

        ## Return a software trigger at the end of rearrannging
        # return True
        pass

    def __vLoadPrecalData(self, precal_data_path):
        self.precal_data = h5py.File(precal_data_path, 'r')

    def __vWriteStepEntry(self, dwStepIndex, dwStepNextIndex, dwSegmentIndex, dwLoops, dwFlags):
        """
        Equilivent to the matlab file spcMSetupSwquenceStep.m

        The function used to write the step of the sequences. It will be called by the vConfigureSequence_dyn and vConfigureSequence_stat
        Parameters
        ----------
        dwStepIndex : int
            the current step index
        dwStepNextIndex : int
            the next step inex
        dwSegmentIndex : int
            the index of the segment index we want to play for the current step
        dwLoops : int
            the number of the loops required
        dwFlags : TYPE
            a flag for specific requirement (check the manual)

        Returns
        -------
        None.

        """
        qwSequenceEntry = uint64 (0)

        ## setup register value
        qwSequenceEntry = (dwFlags & ~SPCSEQ_LOOPMASK) | (dwLoops & SPCSEQ_LOOPMASK)
        qwSequenceEntry <<= 32
        qwSequenceEntry |= ((dwStepNextIndex << 16)& SPCSEQ_NEXTSTEPMASK) | (int(dwSegmentIndex) & SPCSEQ_SEGMENTMASK)

        dwError = spcm_dwSetParam_i64 (self.hCard, SPC_SEQMODE_STEPMEM0 + dwStepIndex, int64(qwSequenceEntry))
        if dwError != ERR_OK:
            raise ValueError(f'Invalid memory segmant entry.')
        ##SPCSEQ_END

    def loadDataIntoCard(self, precal_data_path, load_mode):
        """
        Laoding the precomputed rf signals and load them into AWG memory.
        """
        ## Chekpoint for card loading mode
        if load_mode not in ['static', 'rearranging']:
            raise ValueError(f"Card loading mode can only accept 'static' or 'rearranging', but got {load_mode}.")
        
        ## Open the precal data
        self.precal_data = h5py.File(precal_data_path, 'r')
        
        ## generate the data and transfer it to the card
        self.lMaxADCValue = int32 (0)
        dwError = spcm_dwGetParam_i32(self.hCard, SPC_MIINST_MAXADCVALUE, byref(self.lMaxADCValue))

        ## Taking the mathematica signal and compile it to analouge signals
        self.vDoDataCalculation(self.lCardType, self.lNumChannels, int32(self.lMaxADCValue.value - 1), load_mode)
        sys.stdout.write("... data has been transferred to board memory\n")

    def startCard(self, trigger, wait):
        """
        start the process to generate the tweezer and wait for the order of rearrangment


        Arguments
        ---------
        trigger: bool
            The option for enabling trigger after the card started.
        wait: bool
            The option for the card to excute any further commands after all current sequence is played.


        Returns
        -------
        None.

        """

        ## We'll start and wait until all sequences are replayed.
        ## Disable timeout
        spcm_dwSetParam_i32 (self.hCard, SPC_TIMEOUT, 0)

        ## Start the card with current registers
        print(f'[{time.strftime("%Y%m%d-%H%M%S")}] AWG card started...', end='\r')

        ## Wait command only returns until card has complete all steps in the sequence
        if trigger and wait:
            dwErr = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
            
        elif wait:
            dwErr = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_WAITREADY)
        
        elif trigger:
            dwErr = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
        
        else:
            dwErr = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START)

        ## Check point to see if card is started properly
        if dwErr != ERR_OK:
            spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
            sys.stdout.write ("... Error: {0:d}\n".format(dwErr))
            exit (1)

        ## If everything looks fine, then return a software trigger to indicate rearranging is completed.
        else:
            return dwErr

    def reset(self):
        """
        Command to stop the Card. To use card again, need to reinitialize
        """
        ## Send the stop command

        spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)
        print("Card has been reset")

    def close(self):
        """
        Command to stop the Card. To use card again, need to reinitialize
        """
        ## Send the stop command
        try:
            ## Stop the card
            dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA)
            print("Card has been Stopped")
            ## Wipe out card memory
            spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)
            print("Card has been reset")
            ## Close the card to release allocated memory
            spcm_vClose (self.hCard)
            return 0

        except Exception as e:
            print("Exception",str(e), " has occured")
            return -1
  
class M4i_FIFO_simple_rep_card:
    def __init__(self, in_q, address = b'/dev/spcm0', channelNum = 4, sampleRate = 625):
        """
        The FIFO simple mode for rearranging.

        Parameters
        ----------
        address :The address of the card. The default is b'/dev/spcm0'.
        channelNum : The numnber of channels. The default is 4.
        sampleRate : sample rate in MS/s. The default is 625.
        
        Returns
        -------
        None.

        Notes
        -----
        Process for rearranging:
            1. Initialize the card before use. 
            2. Wait for software trigger (from GUI or after taken Andor initial image) to begin Rearrange
            3. Rearrange signals are calculated in realtime, a short delay is expected
            4. After the rearranging is done, send a software trigger back to Andor camer for final image shoot.
        
        #: TODO: Might need
            M2CMD_CARD_WAITREADY to wait for rearranging to complete
        """
        ## Open the card
        self.hCard = spcm_hOpen (create_string_buffer(address))
        self.lCardType = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_PCITYP, byref (self.lCardType))
        self.lSerialNumber = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_PCISERIALNO, byref (self.lSerialNumber))
        self.lFncType = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_FNCTYPE, byref (self.lFncType))

        ## Check if the card itself is valid
        Valid = self.checkCard()
        if Valid == False:
            exit()
        ## Set the sample rate to 625MS/s
        self.SampleRate = MEGA(sampleRate)

        ## Set samplerate to 1 MHz (M2i) or 50 MHz, no clock output
        if ((self.lCardType.value & TYP_SERIESMASK) == TYP_M4IEXPSERIES) or ((self.lCardType.value & TYP_SERIESMASK) == TYP_M4XEXPSERIES):
            spcm_dwSetParam_i64 (self.hCard, SPC_SAMPLERATE, self.SampleRate)
        else:
            spcm_dwSetParam_i64 (self.hCard, SPC_SAMPLERATE, MEGA(1))
        spcm_dwSetParam_i32 (self.hCard, SPC_CLOCKOUT,   0)

        ## Setup the mode
        self.qwChEnable = uint64 (CHANNEL0 | CHANNEL1)
        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE,    SPC_REP_FIFO_SINGLE)
        spcm_dwSetParam_i64(self.hCard, SPC_CHENABLE,    self.qwChEnable)
        spcm_dwSetParam_i64(self.hCard, SPC_SEGMENTSIZE, 4096) # used to limit amount of replayed data if SPC_LOOPS != 0
        spcm_dwSetParam_i64(self.hCard, SPC_LOOPS,       0) # continuous replay

        # spcm_dwSetParam_i32(self.hCard,SPC_CLOCKMODE,SPC_CM_EXTREFCLOCK)
        # spcm_dwSetParam_i32(self.hCard,SPC_REFERENCECLOCK,10000000)
        # spcm_dwSetParam_i64(self.hCard,SPC_SAMPLERATE,625000000)

        # spcm_dwSetParam_i32 (self.hCard, SPC_FILTER0, int32(1))
        # spcm_dwSetParam_i32 (self.hCard, SPC_FILTER1, int32(1))
        self.lMaxDACValue = int32 (0)
        spcm_dwGetParam_i32(self.hCard, SPC_MIINST_MAXADCVALUE, byref(self.lMaxDACValue))
        self.lSetChannels = int32 (0)
        spcm_dwGetParam_i32(self.hCard, SPC_CHCOUNT, byref (self.lSetChannels))
        self.lBytesPerSample = int32 (0)
        spcm_dwGetParam_i32(self.hCard, SPC_MIINST_BYTESPERSAMPLE,  byref (self.lBytesPerSample))

        ## Setup the trigger
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_ORMASK,      SPC_TMASK_SOFTWARE)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_ANDMASK,     0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ORMASK0,  0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ORMASK1,  0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ANDMASK0, 0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ANDMASK1, 0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIGGEROUT,       0)

        ## Setup all channels
        ## Improve the way to select the channel later
        ## The actual ouput power measurement from AWG ouutput P_F ~ (alpha/N)*|V_awg|**2
        self.volt_lv = [int(120),int(120)]
        # setup all channels
        ## improve the way to select the channel later
        for i in range (0, self.lSetChannels.value):
            spcm_dwSetParam_i32 (self.hCard, SPC_AMP0 + i * (SPC_AMP1 - SPC_AMP0), int32(self.volt_lv[i]))
            spcm_dwSetParam_i32 (self.hCard, SPC_ENABLEOUT0 + i * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0),  int32(1))

        self.FeedbackOn = False
        self.in_q = in_q

    def checkCard(self):
        """
        Function that checks if the card used is indeed an M4i.6622-x8 or is compatible with Analog Output.
        copied from the Toronto's group

        """

        ## Check if Card is connected
        if self.hCard == None:
            print("no card found...\n")
            return False

        ## Getting the card Name to check if it's supported.
        try:

            self.sCardName = szTypeToName (self.lCardType.value)
            if self.lFncType.value == SPCM_TYPE_AO:
                print("Found: {0} sn {1:05d}\n".format(self.sCardName,self.lSerialNumber.value))
                return True
            else:
                print("Code is for an M4i.6622 Card.\nCard: {0} sn {1:05d} is not supported.\n".format(self.sCardName,self.lSerialNumber.value))
                return False

        except:
            dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
            print(dwError)
            print("Problem occured, mb")

    def setSoftwareBuffer(self):
        """
        Function to set up the SoftwareBuffer, no arguments required.
        """
        ## Setup software buffer
        print(self.lBytesPerSample,self.lSetChannels.value)
        self.lNotifySize_bytes = int32(4096*1024)
        self.qwBufferSize = uint64 (4*4096*1024)

        ## We try to use continuous memory if available and big enough
        self.pvBuffer = c_void_p ()
        self.qwContBufLen = uint64 (0)
        spcm_dwGetContBuf_i64 (self.hCard, SPCM_BUF_DATA, byref(self.pvBuffer), byref(self.qwContBufLen))
        sys.stdout.write ("ContBuf length: {0:d}\n".format(self.qwContBufLen.value))
        if self.qwContBufLen.value >= self.qwBufferSize.value:
            sys.stdout.write("Using continuous buffer\n")
        else:
            self.pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
            sys.stdout.write("Using buffer allocated by user program\n")

    def setupCard(self):
        """
        Get the precalculated the data and load the initial data to the buffer, ready for starting the card.
        """
        ## Precalculate data and pump it into the pointer array
        self.precalSignal()
        
        ## Transfer the data refereced by the pointer array to board
        self.transferInitData()

    def precalSignal(self):
        """
        Prepare the precalculated signal
        """
        ## Define the max length of all channels
        self.lPreCalcLen = int(self.precal_data.size / 2)# max length of all channels. denominator = 2 only for two channels
        
        ## Define the pointer array
        self.pnPreCalculated = ptr16 (pvAllocMemPageAligned (self.lPreCalcLen * self.lSetChannels.value*2)) # buffer for pre-calculated and mixed data
        sys.stdout.write(f'[{time.strftime("%Y%m%d-%H%M%S")}] Precal-data Len: {self.lPreCalcLen} Allocated Buffer size: {self.pnPreCalculated}\n')
        
        ## Loading data to pointer array
        for i in range (0, self.lPreCalcLen*2, 1):
            self.pnPreCalculated[i] = int((self.lMaxDACValue.value-1) * self.precal_data[i])
     
    def loadData(self,data):
        """
        The command to load the data to this program

        Note
        ----
        Here we assume the 2D data is a flattened array  (x0, x1, x0, x1, ...)
        """
        self.precal_data = data

    def transferInitData(self):
        """
        The function to transfer the initial data to the buffer
        """
        ## Calculating data for all enabled channels, starting at sample position 0, and fill the complete DMA buffer
        self.qwSamplePos = 0
        self.lNumAvailSamples = (self.qwBufferSize.value // self.lSetChannels.value) // self.lBytesPerSample.value
        
        ## Calculate the actual analogue data interns of DAC voltage
        self.vCalcNewData(self.pvBuffer, self.lSetChannels.value, self.qwSamplePos, self.lNumAvailSamples)
        self.qwSamplePos += self.lNumAvailSamples

        ## Defined the buffer for transfer and start the DMA transfer
        sys.stdout.write(f'[{time.strftime("%Y%m%d-%H%M%S")}] Starting the DMA transfer and waiting until data is in board memory\n')
        spcm_dwDefTransfer_i64(self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, self.lNotifySize_bytes, self.pvBuffer, uint64 (0), self.qwBufferSize)
        spcm_dwSetParam_i32(self.hCard, SPC_DATA_AVAIL_CARD_LEN, self.qwBufferSize)
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA)

    def vCalcNewData(self, pnBuffer, lNumCh, llSamplePos, llNumSamples):
        """
        the function to calculate new data
        """
        lStartPosInBuffer_bytes = (llSamplePos % self.lPreCalcLen) * 2 * lNumCh
        lToCopy_bytes = llNumSamples * 2 * lNumCh
        lPreCalcLen_bytes = self.lPreCalcLen * 2 * lNumCh
        lAlreadyCopied_bytes = 0

        while lAlreadyCopied_bytes < lToCopy_bytes:
            ## Copy at most the pre-calculated data
            lCopy_bytes = lToCopy_bytes - lAlreadyCopied_bytes
            if lCopy_bytes > lPreCalcLen_bytes - lStartPosInBuffer_bytes:
                lCopy_bytes = lPreCalcLen_bytes - lStartPosInBuffer_bytes

            ## Copy data from pre-calculated buffer to DMA buffer
            ctypes.memmove (cast (pnBuffer, c_void_p).value + lAlreadyCopied_bytes, cast (self.pnPreCalculated, c_void_p).value + lStartPosInBuffer_bytes, lCopy_bytes)
            lAlreadyCopied_bytes += lCopy_bytes
            lStartPosInBuffer_bytes = 0

    def startCard(self, repeat, wait):
        """
        The commands to start the card

        Notes
        -----
        1. Signals to be played are sitting in pvBuffer. 
        2. Card will play data in pvBuffer in each iteration.
        3. If ther is a new signal waiting in the que, then it will replace 
            old data in pvbuffer with new.

        """
        self.lStatus = int32(0)
        self.lAvailUser_bytes = int32(0)
        self.lPCPos = int32(0)
        self.lFillsize = int32 (0)
        self.bStarted = False

        ## While loop, will keep pumping data into card with new data
        test_iter = 0
        while True:
            print(f'\nTest iteration: {test_iter}')
            dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)
            
            ## If the card didn't pass through the error check stop the card
            if dwError != ERR_OK:
                if dwError == ERR_TIMEOUT:
                    sys.stdout.write ("... Timeout\n")
                else:
                    sys.stdout.write ("... Error: {0:d}\n".format(dwError))
                    break
            
            ## Otherwise, the card is all set
            else:
                ## Start the card if the onboard buffer has been filled completely
                ## SPC_FILLSIZEPROMILLE reports the filled memory with total 1000 unit, with 63 promile steps
                spcm_dwGetParam_i32 (self.hCard, SPC_FILLSIZEPROMILLE, byref (self.lFillsize));
                if self.lFillsize.value == 1000 and self.bStarted == False:
                   
                    ## Output the current card process
                    sys.stdout.write(f'[{time.strftime("%Y%m%d-%H%M%S")}] Data has been transferred to board memory\n')
                    sys.stdout.write(f'[{time.strftime("%Y%m%d-%H%M%S")}] Starting the card...\n')
                    
                    ## The actual start command itself.
                    ## Wait Flags added here, comment out if unstable
                    if wait:
                        dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
                    else:
                        dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
                    print(f'wait: {wait}, error: {dwError}')

                    if dwError == ERR_TIMEOUT:
                        spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
                        sys.stdout.write ("... Timeout at start\n")
                        break;
                    
                    ## If all set, turn on feedback mode
                    self.bStarted = True
                    self.FeedbackOn = True
                else:
                    pass
                    # sys.stdout.write ("... Fillsize: {0:d}/1000\n".format(self.lFillsize.value))
                
                ## Obtaining some required data for further use
                spcm_dwGetParam_i32 (self.hCard, SPC_M2STATUS,            byref (self.lStatus))
                spcm_dwGetParam_i32 (self.hCard, SPC_DATA_AVAIL_USER_LEN, byref (self.lAvailUser_bytes))
                err0 = spcm_dwGetParam_i32 (self.hCard, SPC_DATA_AVAIL_USER_POS, byref (self.lPCPos))
                print(f"err0: {err0}")
                ## Calculate new data
                if self.lAvailUser_bytes.value >= self.lNotifySize_bytes.value:
                    self.pnData = (c_char * (self.qwBufferSize.value - self.lPCPos.value)).from_buffer (self.pvBuffer, self.lPCPos.value)
                    self.lNumAvailSamples = (self.lNotifySize_bytes.value // self.lSetChannels.value) // self.lBytesPerSample.value # to avoid problems with buffer wrap-arounds we fill only one notify size
                    
                    ## If the feedback mode is turned on. The original application for this is to generate high quality square lattices
                    ## High quality static lattice rely on updating signals based on the trap image feedback
                    if self.FeedbackOn:
                        ## If there is no new data commming in 
                        if self.in_q.empty():
                            print("Empty....")
                            ## If static mode
                            if repeat:
                                ## Then just take the old data can calculate it and feed it to the card
                                print(f'[{time.strftime("%Y%m%d-%H%M%S")}] No new updata in que.')
                                self.vCalcNewData (self.pnData, self.lSetChannels.value, self.qwSamplePos, self.lNumAvailSamples)
                            ## If not out put required
                            else:
                                # TODO: Should still give zero aray output!
                                pass
                            
                        ## If we have new data in the que, just take it and put it into the card
                        else:
                            print(f'[{time.strftime("%Y%m%d-%H%M%S")}] Update new data now...')
                            ## Get new data from the que, after this, que is switched to the next element in que
                            new_par = self.in_q.get()

                            ## TODO: We are now feeding new data as a dictionary, need to make adaptions
                            combined_data = new_par[0] + new_par[1]
                            self.newdata = np.column_stack(combined_data).flatten() #Here we have the new data for the new signal
                            self.loadData(self.newdata)  #load the data to precalculated_data
                            self.precalSignal()
                            self.vCalcNewData(self.pvBuffer, self.lSetChannels.value, self.qwSamplePos, self.lNumAvailSamples)
                    else:
                        self.vCalcNewData(self.pnData, self.lSetChannels.value, self.qwSamplePos, self.lNumAvailSamples)
                    
                    err_1 = spcm_dwSetParam_i32 (self.hCard, SPC_DATA_AVAIL_CARD_LEN, self.lNotifySize_bytes)
                    print(f'err_1: {err_1}')
                    self.qwSamplePos += self.lNumAvailSamples
            test_iter += 1

    def stop(self):
        """
        Stop all current card process.
        """
        dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA)
        print(dwError)
        print("Card has been Stopped")

    def close(self):
        """
        Command to stop the Card. To use card again, need to reinitialize
        """
        ## Send the stop command
        try:
            dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA)
            print(dwError)
            print("Card has been Stopped")
            spcm_vClose (self.hCard)
            return 0

        except Exception as e:
            print("Exception",str(e), " has occured")
            return -1

class AtomRearrange_test_card(object):
    """
    The test version atom rearranging controller, the user interface with sequence replay mode.

    Notes
    -----
    Briefly describe of the process:
        1. Initialize the card. 
        2. It seems like the algorithm need some precomputed signals
    """
    def __init__(self, address = b'/dev/spcm0', channelNum = 4, sampleRate=625):
        ### This part defines the parameters of the card
        """

        Parameters
        ----------
        address :The address of the card. The default is b'/dev/spcm0'.
        channelNum : The numnber of channels. The default is 4.
        sampleRate : sample rate in Mbyte/s. The default is 625.
        Returns
        -------
        None.

        Note
        ----
        spcm_dwSetParam(device, register, value) : Reserve software registers for setting hardware commands.
        
        spcm_dwGetParam(device, register, value) : Reserve software registers for reading hardware commands.
        """

        #open the card
        print(f'Card opened.')


    def checkCard(self):
        """
        Function that checks if the card used is indeed an M4i.6622-x8 or is compatible with Analog Output.
        copied from the Toronto's group

        """
        print(f'Checking card info.')


        #Check if Card is connected
    def static(self):
        """
        Setup static array in sequence replay mode.
        
        Note
        ----
        Working.
        """
        print(f'Configuring static array.')

    def Rearrange(self, path):
        """
        The function that define the order of the moving sequence based on path data in frequency graph.

        Parameters
        ----------
        path : The non-collision path in freq basis.

        Returns
        -------
        None.

        # TODO: Rewrite the data loader from here.

        """
        step = 0
        for move_index, moves in enumerate(path):
            for site_index, site in enumerate(moves):
                ## grab
                if site_index == 0:
                    memory_seg = int(self.precal_data["grab"]["index"][f"{tuple(site)}"][()])
                    print(f'grab: {memory_seg}')
                    step += 1
                
                ## drop
                elif site_index == len(moves) - 1:
                    memory_seg = int(self.precal_data["drop"]["index"][f"{tuple(site)}"][()])
                    print(f'drop: {memory_seg}')
                    # if at the end of the path, end the replay sequence
                    if (move_index == len(path) - 1):
                        print(f'Sequence end.')
                    else:
                        step += 1
                    break
                
                ## move
                memory_seg = int(self.precal_data["move"]["index"][f"({tuple(moves[site_index])}, {tuple(moves[site_index + 1])})"][()])
                print(f'Move: {memory_seg}')
                step += 1
        


    def __vWriteStepEntry(self, dwStepIndex, dwStepNextIndex, dwSegmentIndex, dwLoops, dwFlags):
        """
        Equilivent to the matlab file spcMSetupSwquenceStep.m

        The function used to write the step of the sequences. It will be called by the vConfigureSequence_dyn and vConfigureSequence_stat
        Parameters
        ----------
        dwStepIndex : int
            the current step index
        dwStepNextIndex : int
            the next step inex
        dwSegmentIndex : int
            the index of the segment index we want to play for the current step
        dwLoops : int
            the number of the loops required
        dwFlags : TYPE
            a flag for specific requirement (check the manual)

        Returns
        -------
        None.
        """

        pass

    def vDoDataCalculation (self):
        """
        The signal data is precalculated and stored in HDF5 file at specific location. 
        Here the function will take data and allocate them to specific segements.

        Parameters
        ----------
        lCardType : long?
            Card type argument. Kept here due to histrical reason.
        lNumActiveChannels : long?
            The number of activated channels, the actual number can be extractd by lNumActiveChannels.value.
        lMaxDACValue : long?
            The maximum allowed voltage output value of the AWG card.

        Note
        ----
        a. In the new version, the precalculated data is a 2D data already. And from the dataset, we need to embeded segment index.
        b. Memory buffers will enhence the efficiency of data transfer about 10~15%.
        And in this example, we load the precalculated data into PC memory buffer through pointer array.
        Then write the data (loaded in the buffer) into board memory segments.
        """
        lNumActiveChannelsvalue = 2

        for signal_mode in ["static", "grab", "drop", "move"]:
            sys.stdout.write (f"Loading {signal_mode} data from PC to card...\n")
            
            # TODO: The code here might be slow
            signals = list(self.precal_data[signal_mode]["signal"]) # signals is a 2xt np.array

            for signal in tqdm(signals):
                signal_dataset = self.precal_data[signal_mode]["signal"][signal][()]
                dwSegmentLenSample = signal_dataset.shape[1] # the size of t
                if dwSegmentLenSample == 0:
                    raise ValueError(f'In valid signal data, this is a empty dataset.')
                for t in range(0, dwSegmentLenSample):
                    for ch in range(0, lNumActiveChannelsvalue):
                        data = signal_dataset[ch][t]
                        #print(f'Data: {data}')
                
                ## Write data into memory
                print("Write segment index: ", self.precal_data[signal_mode]["index"][signal][()])
    
    def loadDataIntoCard(self, precal_data_path):
        """
        Laoding the precomputed rf signals and load them into AWG memory.
        """
        # Open the precal data
        self.precal_data = h5py.File(precal_data_path, 'r')
        self.vDoDataCalculation()
        sys.stdout.write("... data has been transferred to board memory\n")

    def reset(self):
        """
        Command to stop the Card. To use card again, need to reinitialize
        """
        ## Send the stop command

        print("Card has been reset")

    def startCard(self, trigger, wait):
        """
        start the process to generate the tweezer and wait for the order of rearrangment

        Returns
        -------
        None.

        """

        print(f'Start the card.')


    def close(self):
        print(f'Card closed.')
    
def main():
    """
    Fix before use.
    """
    Rearrange_card = AtomRearrange_REP_STD_SEQUENCE_card(address=b'/dev/spcm0', channelNum = 4, sampleRate=625)
    Rearrange_card.startCard()
    Rearrange_card.__vLoadPrecalData(precal_data_path = "Meow")
    Rearrange_card.Rearrange(path = [[0, 1], [1, 1]])
    Rearrange_card.stop()
