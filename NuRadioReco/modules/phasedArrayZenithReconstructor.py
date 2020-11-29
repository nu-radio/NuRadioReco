from __future__ import absolute_import, division, print_function, unicode_literals
from NuRadioReco.modules.base.module import register_run
from scipy import signal, fftpack
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import scipy.optimize as opt
from radiotools import helper as hp
from NuRadioReco.utilities.geometryUtilities import get_time_delay_from_direction
from scipy.signal import hilbert
import NuRadioReco.utilities.io_utilities
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium

class phasedArrayZenithReconstructor:
    
    """
    This module gives an estimate for the zenith arrival direction of the signal using the phased array. It phases the four channel using a plane wave approximation and find the zenith for which the amplitude of the combined trace is the largest. 
    """
    
    def __init__(self):
        self.begin()
        
    def begin(self, channel_ids = [6,7,8,9], icemodel = medium.greenland_simple()):
        self.__channel_ids = channel_ids
        self.__icemodel = icemodel
        
        
        
    def run(self, evt, station, det, debug = True):
        
        sampling_rate = station.get_channel(self.__channel_ids[0]).get_sampling_rate()
        
        def amplitude(params, minimizer = True):
            zenith, azimuth = params
            pos1 = det.get_relative_position(station.get_id(), self.__channel_ids[0])
            n_index = self.__icemodel.get_index_of_refraction(pos1)

            timedelay1 = geo_utl.get_time_delay_from_direction(zenith, azimuth, pos1, n=n_index)
            phased_trace = np.zeros(len(station.get_channel(self.__channel_ids[0]).get_trace()))
            for channel in station.iter_channels():
                if channel.get_id() in self.__channel_ids:
                    pos = det.get_relative_position(station.get_id(), channel.get_id())
                    timedelay = geo_utl.get_time_delay_from_direction(zenith, azimuth, pos, n=n_index)
                    print('timedelay', timedelay)
                    delta_t = timedelay1 - timedelay
                    trace = np.roll(np.copy(channel.get_trace()), int(delta_t *sampling_rate))
                    phased_trace += trace 
            amplitude = np.max(abs(phased_trace))
            if minimizer == False:
                return phased_trace
            else: 
                return - 1* amplitude 
        
        ZenLim = [0, np.deg2rad(180)]
        AzLim = [-1*np.deg2rad(180), np.deg2rad(180)]
        ll = opt.brute(amplitude, ranges=(slice(ZenLim[0], ZenLim[1], .1), slice(AzLim[0], AzLim[1], 1)))
        station[stnp.zenith] = ll[0]
        
        
        if debug: 
            fig = plt.figure()
            ax = fig.add_subplot(111)
            zeniths = np.arange(0, np.deg2rad(180), 0.005)
            y_plot = np.zeros(len(zeniths))
            x_plot = np.zeros(len(zeniths))
            i = 0
            for zen in zeniths:
                y_plot[i] = amplitude([zen, 0])
                x_plot[i] = zen
                i += 1
                
            ax.plot(np.rad2deg(x_plot), -1*y_plot, '-')
            ax.axvline(np.rad2deg(ll[0]), label = 'reconstructed zenith', color = 'orange', linewidth = 1)
            
            trace = amplitude([ll[0],ll[1]], minimizer = False)
            fig1 = plt.figure(figsize =(10,10))
            ax1 = fig1.add_subplot(1, 1, 1)

            for iCh, channelid in enumerate(self.__channel_ids):
                ax1.plot(station.get_channel(channelid).get_trace(), color = 'blue', label = 'channel {}'.format(channelid))
            
            ax1.plot(trace, color = 'black', label = 'phased trace')
            ax1.set_xlim((np.argmax(trace) - 200, np.argmax(trace) + 200))
            ax1.set_xlabel("time [ns]", fontsize = 'large')
            ax1.grid()

            ax1.legend(fontsize = 'large')
            fig1.tight_layout()
           
           
            if station.has_sim_station():
                for efield in station.get_sim_station().get_electric_fields():
                    if efield.get_channel_ids()[0] in self.__channel_ids:
                        ax.axvline(np.rad2deg(efield[efp.zenith]), color = 'blue', linewidth = 1)
            ax.set_xlabel("zenith [degrees]", fontsize = 'small')
            ax.set_ylabel("amplitude", fontsize = 'small')
            ax.legend()
            fig.tight_layout()
            
            
        
        
            
            
    