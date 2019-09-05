import datetime
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import random 
import math
import scipy.optimize
from local_GP_toolbox import *


import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.utilities.trace_utilities as trace_utilities
import NuRadioReco.detector.detector as detector
import NuRadioReco.framework.electric_field as electric_field
import NuRadioReco.utilities.fft as fft
import NuRadioReco.framework.base_trace

from NuRadioReco.framework.parameters import electricFieldParameters as efieldp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.utilities import units
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from NuRadioReco.utilities.geometryUtilities import get_time_delay_from_direction


import NuRadioReco.modules.io.eventReader as eventReader

class GaussianProcessAnalyser:

    """
    Module which filters out noise on the trace for a given station and events using Locally Coupled Gaussian Process Regression
    Code for the Gaussian Process Analysis (local_GP_toolbox.py and example "Local GP toolbox") by Luca Ambrogioni: https://github.com/LucaAmbrogioni/LocallyCoupledGP
    """
    
    def __init__(self):
        pass

    def begin(self):
        pass

    def run(self, station, trace_window = 160*units.nanosecond, window_spacing = 5.0*units.nanosecond, window_width = 7.0 *units.nanosecond, frequency_min = 30*units.MHz, frequency_max = 300*units.MHz, frequency_width = 10.0 *units.nanosecond, signal_amplitude = 0.08*units.V, noise_sd = 20* units.mV, signal_cut = False,  keep_trace_length = True, cut_off_noise = False ):
        """
        Parameters
        -----------
        station: station object corresponding to event(s) to be analysed
        trace_window: time window (with even number of points wrt sampling rate) big enough to include the signal pulse but exclude most of the noise for more efficient analysis (make sure there is not too much noise included in this window, the GP analysis is less likely to work if a large proportion of the training data is just noise), default: 160 ns
        window_spacing: variable for locally coupled Gaussian Process, should be about the time length of the period of the oscillation with the highest frequency, default: 5 ns
        window_width: standard deviation of the Gaussian window function, should be slightly larger than window_spacing, default: 7 ns
        frequency_min : minimum frequency expected to occur in the signal, default: 30 MHz
        frequency_max: maximum frequency expected to occur in the signal, default: 300 MHz
        frequency_width: this is the range the Gaussian process tries to extrapolate a frequency over, should be large enough to "see" the signal is oscillating, but not larger than a couple of periods of the highest occuring frequency in the signal, default: 10.0 ns
        signal_amplitude: approximate amplitude of the signal, default: 0.08 V
        noise_sd: estimated noise level of the signal, default: 20 mV
        signal_cut: whether or not the pulse has already been isolated from the remaining noisy signal (using the channelLengthAdjuster module), default: False, UNTESTED FEATURE
        keep_trace_length: if true, makes sure the output trace has the same length as the original trace, default: True  (False:LEADS TO PROBLEM WITH GETTING THE SPECTRUM RIGHT NOW for trace_max=270, time and training_data length are fine, ValueError: Dimension n should be a positive integer not larger than the shape of the array along the chosen axis, also finds division by zero in rfftfreq -> probably in 1/sampling_rate? )
        cut_off_noise: if true all values of the trace except those in the trace_window are set to zero, if false the original values of the trace within the trace window are replaced by the noise filtered values and all other values are left as they are, default: False, WARNING: gives an error message if set to True when used in FullReconstruction 


        SETTINGS THAT DON'T RESULT IN AN ERROR MESSAGE RIGHT NOW: signal_cut = False, keep_trace_length = True, cut_off_noise = False
        

        
        """
        for channel_number in range(4):
            channel = station.get_channel(channel_number)
            trace = channel.get_trace()
            time = channel.get_times()
            sampling_rate = channel.get_sampling_rate()
            


            if signal_cut:
                time_min = time[0]
                time_max = time[-1]
                training_data = trace[::-1]
                time_step = np.round(1/sampling_rate, 3)


                if np.arange(time_min, time_max, time_step).size < training_data.size:
                    time_max += time_step
                


            else:    
                #finding the pulse in the trace
                trace_max = np.argmax(np.ravel(trace))
                time_step = 1./sampling_rate
                
                # a lot of messy if statements to make sure there aren't any problems if the pulse is very close to the beginning/ end of the trace
                cut_left = False
                cut_right = False
                cut_left_right = False
                
                #window doesn't have enough data points to the left of the signal maximum
                if trace_max < trace_window*sampling_rate/2:
                    time_min = time[0]
                    time_max = time[trace_max] +trace_window/2 
                    training_data = trace[:trace_max+int(np.round(trace_window*sampling_rate/2))][::-1]
                    cut_left = True
                    print("Couldn't split the window equally on either side of the pulse maximum. Some points to the left of the pulse were cut off. ")
                        
                #window doesn't have enough data points to the right of the signal maximum
                elif trace_max + trace_window*sampling_rate/2 > trace.size:
                    time_min = time[trace_max]-trace_window/2 
                    time_max = time[-1] + 1/sampling_rate
                    training_data = trace[trace_max-int(np.round(trace_window*sampling_rate/2)):][::-1]
                    cut_right = True
                    print("Couldn't split the window equally on either side of the pulse maximum. Some points to the right of the pulse were cut off. ")
                    
                #window doesn't have enough data points to the left and right of the signal maximum
                elif trace_max < trace_window*sampling_rate/2 and trace_max + trace_window*sampling_rate/2 > trace.size:
                    time_min = time[0]
                    time_max = time[-1] + 1/sampling_rate
                    training_data = trace[::-1]
                    cut_left_right = True
                    print("The time window was larger than the trace, adjusted it accordingly.")

                else:
                    time_min = time[trace_max]-trace_window/2
                    time_max = time[trace_max]+trace_window/2
                    training_data = trace[trace_max-int(np.round(trace_window*sampling_rate/2)):trace_max+int(np.round(trace_window*sampling_rate/2))][::-1]
                    print('max', trace_max)
                    print('time', np.arange(time_min, time_max, 1/sampling_rate).size)
                    print('data', training_data.size)
                    print('Succesfully cut off the trace on either side of the pulse.')


                # another messy if statement to deal with the fact that the GPregessor will only take the max and min of the time window instead of data points
                if time_max - time_min != trace_window:
                    time_min = np.round(time_min)
                    time_max = np.round(time_max)
                    print('Fixed time_min and time_max')

                
                

            noise_sd = noise_sd

            local_GP = locally_coupled_GP_analysis()
            #have to make sure that this works out exactly for inputting in np.arange. If it doesn't there will be an error
            training_time = {"time_min": time_min, "time_max": time_max, "time_step": time_step}
            local_GP.input_data(training_data, training_time = training_time)
            spacing = window_spacing
            width = window_width
            time_windows = np.arange(time_min, time_max, spacing)
            frequency_min = frequency_min
            frequency_max = frequency_max
            frequency_step = (frequency_max - frequency_min)/len(time_windows)
            frequency_range  = np.arange(frequency_min, frequency_max, frequency_step)
            function_and_parameters = {"function": local_GP.archive_window_functions["generalized_gaussian"], "parameters": {"width": width, "power":2.}}
            local_GP.initialize_windows(function_and_parameters, spacing)
            nonstationary_parameter = {"type": "frequency", "range": frequency_range}
            #make sure to put a decimal point after the parameter values
            covariance_function_and_parameters = {"function": local_GP.archive_covariance_functions["oscillatory"], "parameters":{"amplitude": signal_amplitude, "width": frequency_width}, "nonstationary_parameter": nonstationary_parameter}
            expectation_function_and_parameters = {"function": local_GP.archive_expectation_functions["constant"], "parameters":{"expectation": 0}}
            local_GP.initialize_local_prior_GP(covariance_function_and_parameters, expectation_function_and_parameters)
            local_GP.initialize_prior_HMM(expectation = 0.03, autoregressive_coefficient = 1, innovation_variance = 1**2 , initial_expectation = 0.1, initial_variance = 10**2)
            noise_function_and_parameters = {"function": local_GP.archive_covariance_functions["white_noise"], "parameters": {"standard_deviation": noise_sd}}
            local_GP.initialize_likelihood(noise_function_and_parameters)
            local_GP.perform_hierarchical_analysis()
    
            y_pred = np.array(local_GP.posterior_GP.expectation)
            trace_pred = np.ravel(y_pred)[::-1]
            
            
            if keep_trace_length:
                if cut_off_noise:
                    if cut_left:
                        new_trace = np.concatenate((trace_pred, np.zeros(trace.size - trace_max + int(np.round(trace_window*sampling_rate/2)))), axis = None)
                        
                    elif cut_right:
                        new_trace = np.concatenate((np.zeros(trace_max - int(np.round(trace_window*sampling_rate/2))),trace_pred), axis = None)

                    elif cut_left_right:
                        new_trace = trace_pred

                    else:
                        new_trace = np.concatenate((np.zeros(trace_max - int(np.round(trace_window*sampling_rate/2))),trace_pred, np.zeros(trace.size - trace_max + int(np.round(trace_window*sampling_rate/2)))), axis = None)
                
                else:
                    if cut_left:
                        new_trace = np.concatenate((trace_pred, trace[trace_max+ int(np.round(trace_window*sampling_rate/2)):]), axis = None)
                        
                    elif cut_right:
                        new_trace = np.concatenate((trace[:trace_max-int(np.round(trace_window*sampling_rate/2))],trace_pred), axis = None)

                    elif cut_left_right:
                        new_trace = trace_pred

                    else:
                        new_trace = np.concatenate((trace[:trace_max-int(np.round(trace_window*sampling_rate/2))], trace_pred, trace[trace_max+ int(np.round(trace_window*sampling_rate/2)):]), axis = None)
                channel.set_trace(new_trace, sampling_rate)

            else:
                channel.set_trace(trace_pred, sampling_rate)
                channel.set_trace_start_time(time_min)

        
        
    def end(self):
        pass
