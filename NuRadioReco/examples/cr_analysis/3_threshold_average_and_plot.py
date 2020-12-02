import numpy as np
import matplotlib.pyplot as plt
import os
from radiotools import helper as hp
from NuRadioReco.utilities import units, io_utilities
import pickle
import argparse


parser = argparse.ArgumentParser(description='Noise Trigger Rate')
parser.add_argument('passband_low', type=int, nargs='?', default = 80, help = 'passband low to check')
parser.add_argument('passband_high', type=int, nargs='?', default = 180, help = 'passband high to check')
parser.add_argument('number_of_files', type=int, nargs='?', default = 1, help = 'number of n_files to loop over')

args = parser.parse_args()
passband_low = args.passband_low
passband_high = args.passband_high
number_of_files = args.number_of_files

parser.add_argument('input_filename', type=str, nargs='?', default = 'output_threshold_final/final_threshold_pb_{:.0f}_{:.0f}_i2000_{}.pickle'.format(
        passband_low, passband_high, number_of_files), help = 'input filename of check')
parser.add_argument('output_path', type=os.path.abspath, nargs='?', default = '', help = 'Path to save output')

args = parser.parse_args()
input_filename = args.input_filename
output_path = args.output_path
abs_output_path = os.path.abspath(args.output_path)

n_files = number_of_files

trigger_status = []
triggered_true = []
triggered_all = []
trigger_efficiency = []
trigger_rate = []

for i_file in range(number_of_files):
    data = []
    data = io_utilities.read_pickle(input_filename, encoding='latin1')
    print(data)
    trigger_efficiency_one = data['efficiency']
    trigger_efficiency.append(trigger_efficiency_one)
    trigger_rate_one = data['trigger_rate']
    trigger_rate.append(trigger_rate_one)
    triggered_trigger_one = data['triggered_true']
    triggered_true.append(triggered_trigger_one)
    trigger_tested_one = data['triggered_all']
    triggered_all.append(trigger_tested_one)
    trigger_status_one = data['trigger_status']
    trigger_status.append(trigger_status_one)

detector_file = data['detector_file']
default_station = data['default_station']
sampling_rate = data['sampling_rate'] * units.gigahertz
station_time = data['station_time']
station_time_random = data['station_time_random']
hardware_response = data['hardware_response']

Vrms_thermal_noise = data['Vrms_thermal_noise'] * units.volt
T_noise = data['T_noise'] * units.kelvin
T_noise_min_freq = data['T_noise_min_freq'] * units.megahertz
T_noise_max_freq = data['T_noise_max_freq '] * units.megahertz

galactic_noise_n_side = data['galactic_noise_n_side']
galactic_noise_interpolation_frequencies_step = data['galactic_noise_interpolation_frequencies_step']

passband_trigger = data['passband_trigger']
number_coincidences = data['number_coincidences']
coinc_window = data['coinc_window'] * units.ns
order_trigger = data['order_trigger']
trigger_thresholds = data['threshold']
n_iterations = data['iteration']


trigger_efficiency = np.array(trigger_efficiency) # dimension 0: number of files, dimension 1: different thresholds
trigger_rate = np.array(trigger_rate)

trigger_tested =  np.sum(triggered_all)
triggered_trigger =  np.sum(triggered_true, axis=0)
trigger_efficiency_all = np.sum(trigger_efficiency, axis=0) / n_files
trigger_rate_all = np.sum(trigger_rate, axis=0) / n_files
iterations = n_iterations * n_files

print('trigger tested all', trigger_tested)
print('triggered true all', triggered_trigger)
print('trigger efficiency all', trigger_efficiency_all)
print('trigger rate all [Hz]', trigger_rate_all/units.Hz)

dic = {}
dic['detector_file'] = detector_file
dic['default_station'] = default_station
dic['sampling_rate'] = sampling_rate * units.gigahertz
dic['T_noise'] = T_noise
dic['T_noise_min_freq'] = T_noise_min_freq * units.megahertz
dic['T_noise_max_freq '] = T_noise_max_freq * units.megahertz
dic['Vrms_thermal_noise'] = Vrms_thermal_noise
dic['galactic_noise_n_side'] = galactic_noise_n_side
dic['galactic_noise_interpolation_frequencies_step'] = galactic_noise_interpolation_frequencies_step
dic['station_time'] = station_time
dic['station_time_random'] = station_time_random
dic['passband_trigger'] = passband_trigger
dic['coinc_window'] = coinc_window
dic['order_trigger'] = order_trigger
dic['number_coincidences'] = number_coincidences
dic['iteration'] = iterations
dic['threshold'] = trigger_thresholds
dic['trigger_status'] = trigger_status
dic['triggered_true'] = np.sum(trigger_status)
dic['triggered_all'] = len(trigger_status)
dic['efficiency'] = trigger_efficiency
dic['trigger_rate'] = trigger_rate
dic['hardware_response'] = hardware_response

print(dic)

with open('results/dict_ntr_pb_{:.0f}_{:.0f}.pickle'.format(passband_trigger[0]/units.megahertz, passband_trigger[1]/units.megahertz),
          'wb') as pickle_out:
    pickle.dump(dic, pickle_out)


filename = 'results/dict_ntr_pb_{:.0f}_{:.0f}.pickle'.format(passband_low, passband_high)

data = io_utilities.read_pickle(filename, encoding='latin1')

efficiency= data['efficiency']
trigger_rate = data['trigger_rate']
trigger_thresholds = np.array(data['threshold'])
passband_trigger = data['passband_trigger']
n_iterations = data['iteration']
T_noise = data['T_noise']
coinc_window = data['coinc_window']
order_trigger = data['order_trigger']
iterations = data['iteration']


print('threshold', trigger_thresholds)
print('efficiency', efficiency)
print('trigger rate', trigger_rate/units.Hz)
print('passband', passband_trigger/units.megahertz)


plt.plot(trigger_thresholds/units.mV, trigger_rate[0]/units.Hz, marker='x', label= 'Noise trigger rate', linestyle='none')
#plt.title('passband = {} MHz, iterations = {:.1e}'.format( passband_trigger/units.megahertz, iterations))
plt.xlabel('Threshold [mV]', fontsize=18)
plt.hlines(0, trigger_thresholds[0]/units.mV, trigger_thresholds[-1]/units.mV, label='0 Hz')
plt.ylabel('Trigger rate [Hz]', fontsize=18)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('results/fig_ntr_passband_{:.0f}_{:.0f}.png'.format(passband_trigger[0]/units.megahertz, passband_trigger[1]/units.megahertz))
plt.close()
