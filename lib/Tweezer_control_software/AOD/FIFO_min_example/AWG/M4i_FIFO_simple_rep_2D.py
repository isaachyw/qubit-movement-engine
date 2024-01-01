# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 19:22:16 2020
(Merry Christmas)
@author: Weijun Yuan
"""
"""
This program is modified from the example simple_rep_fifo.py provided by Spectrum.
It is used to control the abitrary waveform generator (AWG) card. The model number is M4i.6622-x8
Reference:
simple_rep_fifo.py from Spectrum
Single reply code from Vuthalab at University of Toronto
https://github.com/vuthalab/spectrum-awg
First verson released: 25/12/2020
"""

from AWG.pyspcm import *
from AWG.spcm_tools import *
import sys
import ctypes
import numpy as np


class M4i_FIFO_simple_rep_2D:
    def __init__(self,in_q,address=b'/dev/spcm0',channelNum = 4,sampleRate=625):
        """

        Parameters
        ----------
        address :The address of the card. The default is b'/dev/spcm0'.
        channelNum : The numnber of channels. The default is 4.
        sampleRate : sample rate in MS/s. The default is 625.
        Returns
        -------
        None.
        """
        #open the card
        self.hCard = spcm_hOpen (create_string_buffer(address))
        self.lCardType = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_PCITYP, byref (self.lCardType))
        self.lSerialNumber = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_PCISERIALNO, byref (self.lSerialNumber))
        self.lFncType = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_FNCTYPE, byref (self.lFncType))

        #Check if the card itself is valid
        Valid = self.checkCard()
        if Valid == False:
            exit()
        #set the sample rate to 625MS/s
        self.SampleRate = MEGA(sampleRate)

        # set samplerate to 1 MHz (M2i) or 50 MHz, no clock output
        if ((self.lCardType.value & TYP_SERIESMASK) == TYP_M4IEXPSERIES) or ((self.lCardType.value & TYP_SERIESMASK) == TYP_M4XEXPSERIES):
            spcm_dwSetParam_i64 (self.hCard, SPC_SAMPLERATE, self.SampleRate)
        else:
            spcm_dwSetParam_i64 (self.hCard, SPC_SAMPLERATE, MEGA(1))
        spcm_dwSetParam_i32 (self.hCard, SPC_CLOCKOUT,   0)

        # setup the mode
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

        #setup the trigger
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_ORMASK,      SPC_TMASK_SOFTWARE)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_ANDMASK,     0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ORMASK0,  0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ORMASK1,  0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ANDMASK0, 0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ANDMASK1, 0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIGGEROUT,       0)

        # setup all channels
        ## improve the way to select the channel later
        for i in range (0, self.lSetChannels.value):
            spcm_dwSetParam_i32 (self.hCard, SPC_AMP0 + i * (SPC_AMP1 - SPC_AMP0), int32 (120))
            spcm_dwSetParam_i32 (self.hCard, SPC_ENABLEOUT0 + i * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0),  int32(1))

        self.FeedbackOn = False
        self.spacing = 0.5
        self.tones = 5
        self.min_freq = 60
        self.in_q = in_q


    def checkCard(self):
        """
        Function that checks if the card used is indeed an M4i.6622-x8 or is compatible with Analog Output.
        copied from the Toronto's group

        """

        #Check if Card is connected
        if self.hCard == None:
            print("no card found...\n")
            return False

        #Getting the card Name to check if it's supported.
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
        # setup software buffer
        print(self.lBytesPerSample,self.lSetChannels.value)
        self.lNotifySize_bytes = int32(4096*1024)
        self.qwBufferSize = uint64 (4*4096*1024)
        # we try to use continuous memory if available and big enough
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
        get the precalculated the data and load the initial data to the buffer, ready for starting the card.
        """
        self.precalSignal()
        self.transferInitData()

    def precalSignal(self):
        """
        prepare the precalculated signal
        """
        self.lPreCalcLen =int(self.precal_data.size/2)# max length of all channels. denominator = 2 only for two channels
        self.pnPreCalculated = ptr16 (pvAllocMemPageAligned (self.lPreCalcLen * self.lSetChannels.value*2)) # buffer for pre-calculated and muxed data
        sys.stdout.write("Len: {0} Buf: {1}\n".format(self.lPreCalcLen,self.pnPreCalculated))
        for i in range (0, self.lPreCalcLen*2, 1):
            #pnBuffer[i] = i

            self.pnPreCalculated[i] = int((self.lMaxDACValue.value-1)*self.precal_data[i])
        return 0

        
    def loadData(self,data):
        """
        The command to load the data to this program

        """
        self.precal_data = data

    def transferInitData(self):
        """
        The function to transfer the initial data to the buffer
        """
        # we calculate data for all enabled channels, starting at sample position 0, and fill the complete DMA buffer
        self.qwSamplePos = 0
        self.lNumAvailSamples = (self.qwBufferSize.value // self.lSetChannels.value) // self.lBytesPerSample.value
        self.vCalcNewData(self.pvBuffer, self.lSetChannels.value, self.qwSamplePos, self.lNumAvailSamples)
        self.qwSamplePos += self.lNumAvailSamples
        # we define the buffer for transfer and start the DMA transfer
        sys.stdout.write("Starting the DMA transfer and waiting until data is in board memory\n")
        spcm_dwDefTransfer_i64(self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, self.lNotifySize_bytes, self.pvBuffer, uint64 (0), self.qwBufferSize)
        spcm_dwSetParam_i32(self.hCard, SPC_DATA_AVAIL_CARD_LEN, self.qwBufferSize)

        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA)


    def vCalcNewData(self,pnBuffer, lNumCh, llSamplePos, llNumSamples):
        """
        the function to calculate new data
        """
        lStartPosInBuffer_bytes = (llSamplePos % self.lPreCalcLen) * 2 * lNumCh
        lToCopy_bytes = llNumSamples * 2 * lNumCh
        lPreCalcLen_bytes = self.lPreCalcLen * 2 * lNumCh
        lAlreadyCopied_bytes = 0
        while lAlreadyCopied_bytes < lToCopy_bytes:
            # copy at most the pre-calculated data
            lCopy_bytes = lToCopy_bytes - lAlreadyCopied_bytes
            if lCopy_bytes > lPreCalcLen_bytes - lStartPosInBuffer_bytes:
                lCopy_bytes = lPreCalcLen_bytes - lStartPosInBuffer_bytes

            # copy data from pre-calculated buffer to DMA buffer
            ctypes.memmove (cast (pnBuffer, c_void_p).value + lAlreadyCopied_bytes, cast (self.pnPreCalculated, c_void_p).value + lStartPosInBuffer_bytes, lCopy_bytes)
            lAlreadyCopied_bytes += lCopy_bytes
            lStartPosInBuffer_bytes = 0

    def startCard(self):
        """
        the command to start the card
        """
        self.lStatus = int32(0)
        self.lAvailUser_bytes = int32(0)
        self.lPCPos = int32(0)
        self.lFillsize = int32 (0)
        self.bStarted = False

        #while loop
        while True:
            dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)
            if dwError != ERR_OK:
                if dwError == ERR_TIMEOUT:
                    sys.stdout.write ("... Timeout\n")
                else:
                    sys.stdout.write ("... Error: {0:d}\n".format(dwError))
                    break

            else:
                # start the card if the onboard buffer has been filled completely
                spcm_dwGetParam_i32 (self.hCard, SPC_FILLSIZEPROMILLE, byref (self.lFillsize));
                if self.lFillsize.value == 1000 and self.bStarted == False:
                    sys.stdout.write("... data has been transferred to board memory\n")
                    sys.stdout.write("\nStarting the card...\n")
                    dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
                    if dwError == ERR_TIMEOUT:
                        spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
                        sys.stdout.write ("... Timeout at start\n")
                        break;
                    self.bStarted = True
                    self.FeedbackOn = True
                else:
                    pass
                    # sys.stdout.write ("... Fillsize: {0:d}/1000\n".format(self.lFillsize.value))
                spcm_dwGetParam_i32 (self.hCard, SPC_M2STATUS,            byref (self.lStatus))
                spcm_dwGetParam_i32 (self.hCard, SPC_DATA_AVAIL_USER_LEN, byref (self.lAvailUser_bytes))
                spcm_dwGetParam_i32 (self.hCard, SPC_DATA_AVAIL_USER_POS, byref (self.lPCPos))

                # calculate new data
                if self.lAvailUser_bytes.value >= self.lNotifySize_bytes.value:
                    self.pnData = (c_char * (self.qwBufferSize.value - self.lPCPos.value)).from_buffer (self.pvBuffer, self.lPCPos.value)
                    self.lNumAvailSamples = (self.lNotifySize_bytes.value // self.lSetChannels.value) // self.lBytesPerSample.value # to avoid problems with buffer wrap-arounds we fill only one notify size
                    if self.FeedbackOn:
                        if self.in_q.empty():
                            # print("no new updata")
                            self.vCalcNewData (self.pnData, self.lSetChannels.value, self.qwSamplePos, self.lNumAvailSamples)
                        else:
                            print("update now")
                            new_par = self.in_q.get()
                            print(f'new_par: {new_par}')
                            x_data = new_par["x_data"]
                            y_data = new_par["y_data"]
                            combined_data = [x_data]+[y_data]
                            self.newdata = np.column_stack(combined_data).flatten() #Here we have the new data for the new signal
                            self.loadData(self.newdata)  #load the data to precalculated_data
                            self.precalSignal()
                            self.vCalcNewData(self.pvBuffer, self.lSetChannels.value, self.qwSamplePos, self.lNumAvailSamples)
                    else:
                        self.vCalcNewData(self.pnData, self.lSetChannels.value, self.qwSamplePos, self.lNumAvailSamples)
                    spcm_dwSetParam_i32 (self.hCard, SPC_DATA_AVAIL_CARD_LEN, self.lNotifySize_bytes)
                    self.qwSamplePos += self.lNumAvailSamples

    def stop(self):
        """
        Command to stop the Card. To use card again, need to reinitialize
        """
        #send the stop command
        try:
            dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA)
            print(dwError)
            print("Card has been Stopped")
            spcm_vClose (self.hCard)
            return 0
        except Exception as e:
            print("Exception",str(e), " has occured")
            return -1
