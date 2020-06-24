from __future__ import absolute_import, division, print_function, unicode_literals
from NuRadioReco.modules.base.module import register_run
import os
import time
import numpy as np
from scipy import signal
from scipy.signal import correlate
from scipy import optimize as opt
import matplotlib.pyplot as plt


from NuRadioReco.detector import detector
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import units, fft, trace_utilities
from NuRadioReco.modules.voltageToEfieldConverter import get_array_of_channels

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import NuRadioReco.framework.electric_field
from NuRadioReco.utilities.geometryUtilities import get_time_delay_from_direction

from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalGen.parametrizations import get_time_trace
import NuRadioReco.modules.electricFieldBandPassFilter
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
import NuRadioReco.modules.electricFieldResampler
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()




class voltageToAnalyticEfieldConverterNeutrinos:
	"""
	this module reconstructs the electric-field by forward folding an analytic 
	pulse function through the antenna.
	
	This module is specified for neutrinos. The function used to describe the 
	electric field is the Alvarez2009 parametrization. 
	
	Before the reconstruction, the channels are upsampled.
	
	The module is optimized for an electric-field reconstruction using two 
	channels. First, the shape of the pulse is fitted to determine the viewing angle, and then the amplitude is used to determine the energy. This is done using the Vpol 
	antenna. Then the viewing angle is fixed in the Hpol antenna, and the energy
	is fitted. 
	
	The module can fit purely hadronic and purely electromagnetic shower. It is not optimized yet for cc electron neutrino interactions. 
	"""

	def __init__(self):
		self.begin()
		
	def begin(self):
		"""
		begin method. This function is executed before the event loop. 
		The antenna pattern provieder is initialized here. 
		"""
		self.antenna_provider = antennapattern.AntennaPatternProvider()
		pass
	
	
   
	def run(self, evt, station, det, icemodel, shower_type = 'HAD', debug= True, 
			channel_Vpol = 0, channel_Hpol = 1, channels_phasing_Vpol = [2,4,5],
			channels_phasing_Hpol = [3], phasing =False, 
			passband = [150* units.MHz, 700 * units.MHz], bandpass = True, 
			use_MC=True, attenuation = True,  attenuation_model = 'GL1', parametrization = 'Alvarez2009'):
	
		"""
		run method. This function is executed for each event
		
		Parameters
		----------
		evt
		station 
		det
		icemodel
		shower_type: 
            if "EM", the average Alvarez2009 model is used to fit
            if "HAD", an hadronic shower is used for the fit 
		debug: bool
			if True debug plotting is enables
		channel_Vpol: int
			the channel id for the VPol used for the electric field reconstruction
		channel_Hpol: int
			the channel id for the HPol used for the electric field reconstruction
		channels_phasing_Vpol: array of ints
			the channel ids for the antennas used to obtain a phased trace
		channels_phasing_Hpol: array of ints
			the channel ids for the antennas used to obtain a phased trace
		phasing: bool
			if True the channels phased 
		passband: [float, float]
			the lower and upper frequency for which the analytic pulse is 
			calculated and the fit is performed
		bandpass: bool
			if True, the voltage traces are filtered
		use_MC: bool
			if True use simulated direction instead of reconstructed direction
		attenuation: bool
			if True attenuation is included in the reconstruction
		attenuation_model:
			the attenuation model used
		parametrization:
			parametrization model used for the reconstruction. At the moment 
			only optimized for Alvarez2009
		"""
		
		station_id = station.get_id()
		use_channels_efield = [channel_Vpol, channel_Hpol]
		if phasing: 
			use_channels_efield += channels_phasing_Vpol + channels_phasing_Hpol
			phasing_dict = {channel_Vpol: channels_phasing_Vpol + [channel_Vpol], 
						channel_Hpol: channels_phasing_Hpol + [channel_Hpol]}
			
			channel_position_Vpol = det.get_relative_position(station_id, channel_Vpol)
			for channel_id in phasing_dict[channel_Vpol]:
				pos = det.get_relative_position(station_id, channel_id)
				# Check if channels are on the same string and close by
				if np.abs(pos[0] - channel_position_Vpol[0]) > 1.*units.m or np.abs(pos[1] - channel_position_Vpol[1]) > 1.*units.m:
					raise ValueError('All channels have to be on the same string')
				if np.abs(pos[2] - channel_position_Vpol[2]) > 5 * units.m :
				 	raise ValueError('Channel to be phased is to far from original channel')
			for channel_id in phasing_dict[channel_Hpol]:
				pos = det.get_relative_position(station_id, channel_id)
				# Check if channels are on the same string and close by
				if np.abs(pos[0] - channel_position_Vpol[0]) > 1.*units.m or np.abs(pos[1] - channel_position_Vpol[1]) > 1.*units.m:
					raise ValueError('All channels have to be on the same string')
				if np.abs(pos[2] - channel_position_Vpol[2]) > 5 * units.m :
				 	raise ValueError('Channel to be phased is to far from original channel')

		use_channels_efield = det.get_channel_ids(station_id)
		use_channels_efield.sort()
		
		
		#### For now, the true values for direction, vertex positions and refractive index are used due to the simulations. The maximum voltage trace is determined, because the fit is performed around the maximum of the trace. 
		if use_MC and (station.get_sim_station() is not None):
			sampling_rate1 = 5

			electricFieldResampler.run(evt, station.get_sim_station(), det, sampling_rate = sampling_rate1)
			#find ray type with maximum amplitude
			efield_max = 0 
			for i_efield, efield in enumerate(station.get_sim_station().get_electric_fields()):
				if efield.get_channel_ids()[0] in [channel_Vpol]:
					max_efield =max(np.sqrt( efield.get_trace()[1]**2))# Simualted values are determined by getting maximum efield. There is no antenan model included, so this does not always correspond to actual simulated value!
					
					Reflection_Correction = 1
					if max_efield > efield_max:
						raytype = efield[efp.ray_path_type]
						if raytype == 'reflected':
							Reflection_Correction = -1
						zenith = efield[efp.zenith]
						azimuth = efield[efp.azimuth]
						efield_max = max_efield
						viewing_angle = efield[efp.nu_viewing_angle]
				
			ray_solution = ['direct', 'refracted', 'reflected'].index(raytype) + 1
			vertex_position = station.get_sim_station()[stnp.nu_vertex] 
			n_index = icemodel.get_index_of_refraction(vertex_position)
			cherenkov_angle = np.rad2deg(np.arccos(1./n_index))
		
		noise_RMS = det.get_noise_RMS(station.get_id(), 0)
		
		sampling_rate = station.get_channel(channel_Vpol).get_sampling_rate()
		dt = 1./sampling_rate
		efield_antenna_factor, V, V_timedomain = get_array_of_channels(station,
																	   use_channels_efield,
																	   det, 
																	   zenith, 
																	   azimuth, 
																	   antennapattern.AntennaPatternProvider(), 
																	   time_domain=True) 

		N_V_time_domain = len(V_timedomain[0])
		ff = np.fft.rfftfreq(N_V_time_domain, dt)
		mask = ff > 0

        ## determine filter. The filter is now set to the filter we use for RNOG. Probably better to not hardcode the filter, but be able to set it in _begin 
		order = 8
		passband = [50* units.MHz, 1150 * units.MHz]
		b, a = signal.butter(order, passband, 'bandpass', analog=True)
		w, ha = signal.freqs(b, a, ff[mask])
		order = 10
		passband = [0* units.MHz, 700 * units.MHz]
		b, a = signal.butter(order, passband, 'bandpass', analog=True)
		w, hb = signal.freqs(b, a, ff[mask])
		ff = np.fft.rfftfreq(N_V_time_domain, dt)
		fa = np.zeros_like(ff, dtype=np.complex)
		fa[mask] = ha
		fb = np.zeros_like(ff, dtype = np.complex)
		fb[mask] = hb
		h = fb*fa
		
		positions = np.array([det.get_relative_position(station_id,channel_Vpol),
							  det.get_relative_position(station_id,channel_Hpol)])
		

		def minimizer_theta_component(params, R, n_index, toffset, hilbert = False, fit = 'both'):

			"""
			params: Array of floats
				parameters to be minimized 
			R: float
				trajectory length 
			n_index: float
				refractive index 
			toffset: float
				time shift for the analytic trace 
			hilbert: bool
				if True, the hilbert envelop is used for the fit 
			fit:
				if 'viewing angle', viewing angle is fitted with a normalized pulse
				if 'energy', energy is fitted using the reconstructed viewing angle
				if 'both', viewing angle and energy are fitted simultaneously. This is used for electromagnetic showers. 
			"""
			N = 50
			
			if fit == 'viewing angle':
				E_theta = 10**18
				theta =params[0]
			if fit == 'energy':
				E_theta = params[0]
				theta = reconstructed_viewing_angle
			if fit == 'both':
				theta = params[0]
				E_theta = params[1]
			if phasing: E_theta *= len(phasing_dict[channel_Vpol])
		
			tot = 0
		
			time_trace_thet = Reflection_Correction *get_time_trace( E_theta, theta, N_V_time_domain, 
										 dt, shower_type, n_index, R,  parametrization, average_shower = True)

			trace_freq_thet = fft.time2freq(time_trace_thet, 1/dt)
			trace_freq_phi = np.zeros(len(trace_freq_thet))

			if attenuation: 
				trace_freq_thet *= attn
			if bandpass: 
				trace_freq_thet *= h

			chi2 = 0
			for iCh, trace in enumerate(V_timedomain):
				if use_channels_efield[iCh] == channel_Vpol:

					if phasing: trace = get_phased_trace(use_channels_efield[iCh], 
														 phasing_dict[use_channels_efield[iCh]])
					k = int(np.argmax(abs(trace))) 
					maximum_data = max(abs(trace))

					

					analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([trace_freq_thet, trace_freq_phi]), axis = 0)
					analytic_trace = fft.freq2time(analytic_trace_fft, 1/dt)

					normalization_factor = maximum_data / max(abs(analytic_trace))
					if fit == 'viewing angle':
						analytic_trace *= normalization_factor


					if hilbert: 
						chi2 += np.sum(abs((abs(signal.hilbert(np.roll(analytic_trace, 
																	   int(toffset)))[k - N: k+3*N] - abs(signal.hilbert(trace))[k - N: k+3*N]))**2))
					else:
						chi2 += np.sum((np.roll(analytic_trace, 
												int(toffset))[k-N:k+3*N] - trace[k-N:k+3*N])**2)



			tot += -1*(len(analytic_trace[k-N:k+3*N])*(np.log(1/np.sqrt(2*np.pi)) + 0.5*np.log(1/(noise_RMS**2))) - chi2/(2*noise_RMS**2))
	
			return tot

			
		def minimizer(params, R, n_index, toffset_pos, toffset_neg,  
					  hilbert = False, minimizer = True):
			
			"""
			params: Array of floats
				parameters to be minimized 
			R: float
				trajectory length 
			n_index: float
				refractive index 
			toffset_pos: float
				time shift for the analytic trace for negative sign
			toffset_neg: float
				time shift for the analytic trace for positive sign
			hilbert: bool
				if True, the hilbert envelop is used for the fit 
			minimizer: bool
				if True, the output of minimization is returned. 
				if False, the sign of the Hpol component is returned. 
			"""
			N = 50
			if len(params) == 3:
				E_phi = reconstructed_energy_phi
				E_theta = reconstructed_energy_theta
				theta = reconstructed_viewing_angle
			else:
				E_phi =params[0]
				E_theta = reconstructed_energy_theta
				theta = reconstructed_viewing_angle
			if phasing: E_phi *= len(phasing_dict[channel_Hpol])
			if phasing: E_theta *= len(phasing_dict[channel_Vpol])
			chi2 = np.inf
			sign = 0 


			for sign_tmp, toffset in zip([ 1, -1], [toffset_pos, toffset_neg]):
				tmp = 0
				time_trace_thet = Reflection_Correction*get_time_trace(E_theta, theta, N_V_time_domain, 
												 dt, shower_type, n_index, R, parametrization, average_shower = True)
				trace_freq_thet = fft.time2freq(time_trace_thet, 1/dt)
				time_trace_phi = sign_tmp*get_time_trace(E_phi, theta, N_V_time_domain, 
														 dt, shower_type, n_index, R, parametrization, average_shower = True)
				trace_freq_phi = fft.time2freq(time_trace_phi, 1/dt)

				if attenuation: 
					trace_freq_phi *= attn
					trace_freq_thet *= attn
				if bandpass: 
					trace_freq_phi *= h
					trace_freq_thet *= h
				
				
				for iCh, trace in enumerate(V_timedomain):
					if use_channels_efield[iCh] == channel_Vpol:
						if phasing: trace = get_phased_trace(use_channels_efield[iCh],
															 phasing_dict[use_channels_efield[iCh]])
						k = np.argmax(abs(trace))
					if use_channels_efield[iCh] == channel_Hpol:
						if phasing: trace = get_phased_trace(use_channels_efield[iCh],
															 phasing_dict[use_channels_efield[iCh]])
						analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([trace_freq_thet,
																						   trace_freq_phi]), axis = 0)

						analytic_trace = fft.freq2time(analytic_trace_fft, 1/dt)
	
						if hilbert: 
							tmp += np.sum(abs((abs(signal.hilbert(np.roll(analytic_trace, 
																	   int(toffset))))[k - N: k+3*N] - abs(signal.hilbert(trace))[k - N: k+3*N]))**2)

						else:
							tmp += np.sum((np.roll(analytic_trace, 
											int(toffset))[k-N:k+3*N] - trace[k-N:k+3*N])**2)
							
				if tmp < chi2:
					chi2 = tmp
					sign = sign_tmp
			if not minimizer:
				return sign
			else:
				return -1*(len(analytic_trace[k-N:k+3*N])*(np.log(1/np.sqrt(2*np.pi)) + 0.5*np.log(1/(noise_RMS**2))) - chi2/(2*noise_RMS**2))

        
		def get_phased_trace( channel_id, channel_ids_phasing, band_pass = bandpass):
			"""
			channel_id: int
				channel id of channel used for reconstruction 
			channel_ids_phasing: list of ints
				channel ids of the channels used for phasing
			band_pass: bool
				if True, filter is applied 
			"""
			phased_trace= np.zeros(len(V_timedomain[0]))
			
			i = 0
		
			for iCh, trace in enumerate(V_timedomain): 
				if use_channels_efield[iCh] in channel_ids_phasing:
					positions = np.array([det.get_relative_position(station_id,channel_id),
										  det.get_relative_position(station_id,use_channels_efield[iCh])])
				
					n_index_antennaposition = icemodel.get_index_of_refraction(positions[0])
					time_delay1 = get_time_delay_from_direction(zenith, azimuth, 
																positions[0], n = n_index_antennaposition)
					n_index_antennaposition = icemodel.get_index_of_refraction(positions[1])

					time_delay2 = get_time_delay_from_direction(zenith, azimuth, 
																positions[1], n = n_index_antennaposition)
				
					roll = (time_delay1 - time_delay2)
				
					trace_phasing = np.roll(np.copy(trace), int(roll)) 
				

					phased_trace += trace_phasing
					i+= 1
			
			return phased_trace

		prop = propagation.get_propagation_module(name = 'analytic')
		antenna_position = det.get_relative_position(station_id, channel_Vpol)
		r = prop(vertex_position, antenna_position, icemodel, 
				 attenuation_model = attenuation_model)
		r.find_solutions()
		n = r.get_number_of_solutions()
		for iS in range(n):
			
			if r.get_solution_type(iS) == ray_solution:
				R = r.get_path_length(iS)
				attn = r.get_attenuation(iS, ff, 0.5*sampling_rate)
			
		
		time_trace = Reflection_Correction*get_time_trace(10**19,viewing_angle, N_V_time_domain, 
									dt, shower_type, n_index, R, parametrization) 
		trace_freq = fft.time2freq(time_trace, 1/dt)

		if attenuation:
				trace_freq *= attn
		if bandpass:
				trace_freq *= h
		
		analytic_trace_fft = np.sum(efield_antenna_factor[use_channels_efield.index(channel_Vpol)]*np.array([trace_freq,
																											 np.zeros(len(trace_freq))]), axis = 0)
		for iCh, trace in enumerate(V_timedomain):
			if use_channels_efield[iCh] == channel_Vpol: # use Vpol to determine offset
				if phasing: trace = get_phased_trace(use_channels_efield[iCh], phasing_dict[use_channels_efield[iCh]])
				
				k = np.argmax(abs(trace))
			
				
				analytic_trace = fft.freq2time(analytic_trace_fft, 1/dt)
				normalization_factor = max(abs(trace)) / max(abs(analytic_trace))
				analytic_trace *= normalization_factor
				trace = np.copy(trace)
				trace_range = 100 * sampling_rate
				trace[ abs(np.arange(0, len(trace)) - k) > trace_range] = 0  ## assume the pulse is at the maximum 
				corr = signal.hilbert(signal.correlate(trace, analytic_trace))
			
				time_shift_Vpol = np.argmax(corr) - (len(corr)/2) 
				toffset = time_shift_Vpol
				
	
		options = {'maxiter':500, 'disp':True}
		
		bnds = ((np.deg2rad(40), np.deg2rad(70)))

		if shower_type == "HAD":
			hadronic_viewing_angle = opt.minimize(minimizer_theta_component, x0 = (np.deg2rad(55)), args = (R, n_index, toffset, False, 'viewing angle'), method = 'Nelder-Mead', bounds = bnds, options = options)
			
			reconstructed_viewing_angle = hadronic_viewing_angle.x[0]
			bnds = ((10**16, 10**20))
			hadronic_energy = opt.minimize(minimizer_theta_component, x0 = (10**19), args = (R, n_index, toffset, False, 'energy'), method = 'Nelder-Mead', bounds = bnds, options = options)
			
			reconstructed_energy_theta = hadronic_energy.x[0]
			bnds = ((10**16, 10**20))
		
		
		
		if shower_type == "EM":
			electromagnetic = opt.minimize(minimizer_theta_component, x0 = (np.deg2rad(55),  10**19), args = (R, n_index, toffset,  False, 'both'), method = 'Nelder-Mead', bounds = bnds, options = options)

			reconstructed_energy_theta = electromagnetic.x[1]
			reconstructed_viewing_angle = electromagnetic.x[0]

		
		###determine offset for Hpol ###
		
		for sign in [1, -1]:
			E_phi = 10**18
			E_theta = reconstructed_energy_theta
			theta = reconstructed_viewing_angle
			time_trace_thet = get_time_trace(E_theta, theta, N_V_time_domain, 
												 dt, shower_type, n_index, R, parametrization)
			trace_freq_thet = fft.time2freq(time_trace_thet, 1/dt)
			time_trace_phi = sign*get_time_trace(E_phi, theta, N_V_time_domain, 
														 dt, shower_type, n_index, R, parametrization)
			trace_freq_phi = fft.time2freq(time_trace_phi, 1/dt)
			

			if attenuation:
					trace_freq_thet *= attn
					trace_freq_phi *= attn
			if bandpass:
					trace_freq_thet *= h
					trace_freq_phi *= h

			for iCh, trace in enumerate(V_timedomain):
				if use_channels_efield[iCh] == channel_Vpol:
					if phasing: trace = get_phased_trace(use_channels_efield[iCh],
														 phasing_dict[use_channels_efield[iCh]])
					k = np.argmax(abs(trace))
				if use_channels_efield[iCh] == channel_Hpol: # use Vpol to determine offset
					if phasing: trace = get_phased_trace(use_channels_efield[iCh], phasing_dict[use_channels_efield[iCh]])
						
					analytic_trace_fft = np.sum(efield_antenna_factor[iCh]*np.array([trace_freq_thet, trace_freq_phi]), axis = 0)
	
					#cut Hpol trace around maximum due to Vpol
					trace = np.copy(trace)
					trace_range = 100 * sampling_rate
					trace[ abs(np.arange(0, len(trace)) - k) > trace_range] = 0 
					analytic_trace = fft.freq2time(analytic_trace_fft, 1/dt)
					
					normalization_factor = max(abs(trace)) / max(abs(analytic_trace))
					analytic_trace *= normalization_factor

					corr = signal.hilbert(signal.correlate(trace, analytic_trace))

					if sign == 1:
						time_shift_Hpol = np.argmax(corr) - (len(corr)/2)
					
					else: 
						time_shift_Hpol_neg = np.argmax(corr) - (len(corr)/2)

	
		bnds = ((10**17, 10**19))
		Hpol_energy = opt.minimize(minimizer, x0=(reconstructed_energy_theta), method = 'Nelder-Mead', 
								args = (R, n_index, time_shift_Hpol, time_shift_Hpol_neg, False, True), 
								bounds = bnds, options = options) #Fit amplitude phi component
		
		reconstructed_energy_phi = Hpol_energy.x[0]
		
		
		#### store efield 
		time_trace_theta = get_time_trace(reconstructed_energy_theta, reconstructed_viewing_angle, N_V_time_domain, 
										  dt, shower_type, n_index, R, parametrization)
		time_trace_phi = get_time_trace(reconstructed_energy_phi, reconstructed_viewing_angle, N_V_time_domain, 
										dt, shower_type, n_index, R, parametrization)

		trace_freq_thet = fft.time2freq(time_trace_theta, 1/dt)
		trace_freq_phi = fft.time2freq(time_trace_phi, 1/dt)

		if attenuation:
				trace_freq_thet *= attn
				trace_freq_phi *= attn
		if bandpass:
				trace_freq_thet *= h
				trace_freq_phi *= h

		time_trace_theta = fft.freq2time(trace_freq_thet, 1/dt)
		time_trace_phi = fft.freq2time(trace_freq_phi, 1/dt)

		time_trace_theta = np.roll(time_trace_theta, int(toffset))
		time_trace_phi = np.roll(time_trace_phi, int(toffset))
		station_trace = np.array([np.zeros_like(time_trace_theta), time_trace_theta, time_trace_phi])
		electric_field = NuRadioReco.framework.electric_field.ElectricField([channel_Vpol])
		electric_field.set_trace(station_trace, sampling_rate)
		electric_field.set_parameter(efp.polarization_angle, np.arctan2(reconstructed_energy_phi, reconstructed_energy_theta))
		electric_field.set_parameter(efp.nu_viewing_angle, reconstructed_viewing_angle)
		electric_field.set_parameter(efp.signal_energy_fluence, reconstructed_energy_theta)

		# Use Vpol to figure out the timing of the electric field
		
		voltages_from_efield = trace_utilities.get_channel_voltage_from_efield(station, 
																			   electric_field, 
																			   [channel_Vpol], 
																			   det, 
																			   zenith, 
																			   azimuth, 
																			   self.antenna_provider, False)
		correlation = np.zeros(voltages_from_efield.shape[1] + station.get_channel(channel_Vpol).get_trace().shape[0] - 1)
		channel_trace_start_times = []
		for channel_id in [channel_Vpol]:
			channel_trace_start_times.append(station.get_channel(channel_id).get_trace_start_time())
		average_trace_start_time = np.average(channel_trace_start_times)
		for i_trace, v_trace in enumerate(voltages_from_efield):
			if attenuation: 
					v_trace = fft.freq2time(fft.time2freq(v_trace, 1/dt) * attn, 1/dt)
			if bandpass: 
					v_trace = fft.freq2time(fft.time2freq(v_trace, 1/dt) * h, 1/dt)

			n_index_antennaposition = icemodel.get_index_of_refraction(positions[0])

			channel = station.get_channel([channel_Vpol][i_trace])
			time_shift = get_time_delay_from_direction(zenith, 
													   azimuth, 
													   det.get_relative_position(station.get_id(), 
																						  [channel_Vpol][i_trace])) - (channel.get_trace_start_time() - average_trace_start_time)
			voltage_trace = np.roll(np.copy(v_trace), int(time_shift * electric_field.get_sampling_rate()))
			correlation += signal.correlate(voltage_trace, channel.get_trace())

			t_offset = (np.arange(0, correlation.shape[0]) - channel.get_trace().shape[0]) / electric_field.get_sampling_rate()
			electric_field.set_trace_start_time(-t_offset[np.argmax(correlation)] + average_trace_start_time)
			station.add_electric_field(electric_field)
			
		if debug:
			E_theta = reconstructed_energy_theta
			E_phi =  reconstructed_energy_phi
			view_angle = reconstructed_viewing_angle
			sign_phi_component = minimizer([E_theta, view_angle, E_phi], 
							   			R, n_index, time_shift_Hpol, time_shift_Hpol_neg, minimizer = False)
			if phasing: E_theta *= len(phasing_dict[channel_Vpol])
			if phasing: E_phi *= len(phasing_dict[channel_Hpol])
			time_trace_thet = Reflection_Correction *get_time_trace( E_theta, view_angle, N_V_time_domain, 
								 			dt, shower_type, n_index, R, parametrization, average_shower = True)
			
			
			
			time_trace_phi = sign_phi_component* get_time_trace(E_phi, view_angle, 
													 			N_V_time_domain, dt, 
													 			shower_type, n_index, 
													 			R, parametrization, average_shower = True)
			
			trace_freq_thet = fft.time2freq(time_trace_thet, 1/dt)
			trace_freq_phi = fft.time2freq(time_trace_phi, 1/dt)

			if attenuation: 
				trace_freq_thet *= attn
				trace_freq_phi *= attn

			if bandpass: 
				trace_freq_thet *= h
				trace_freq_phi *= h
			

			fig, ax = plt.subplots(2, 2, sharex=False, figsize=(10, 5))
			N = 50
			i = 0
			for iCh, trace in enumerate(V_timedomain):
				if ((use_channels_efield[iCh] == channel_Vpol) or (use_channels_efield[iCh] == channel_Hpol)):

						if phasing: trace = get_phased_trace(use_channels_efield[iCh], 
															 phasing_dict[use_channels_efield[iCh]])
					
						if iCh == use_channels_efield.index(channel_Vpol):
							k = np.argmax(abs(trace))
						if iCh == use_channels_efield.index(channel_Hpol):
							toffset = time_shift_Hpol
							if sign_phi_component == -1:
								toffset = time_shift_Hpol_neg
					

						
						analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([trace_freq_thet, 
																						   trace_freq_phi]), axis = 0)
						
						
						
						time_trace = fft.freq2time(analytic_trace_fft, 1/dt)
						ax[i][0].plot(trace, label = 'voltage trace')
						ax[i][1].plot(ff/units.MHz, abs(fft.time2freq(trace, 1/dt)), label = 'voltage trace')
						ax[i][1].plot(ff/units.MHz, abs(analytic_trace_fft), label = 'fit')
						ax[i][0].plot(np.roll(time_trace, int(toffset)), label = 'fit')
						ax[i][0].set_xlim(([k - N - 50, k + 3*N + 50]))
						ax[i][0].axvline([k+3*N], color = 'purple', label = 'fitting area')
						ax[i][0].axvline([k-N], color = 'purple')
						if i == 0:#
							ax[i][0].legend()
							ax[i][1].legend()
						ax[i][0].set_xlabel("samples", fontsize = 'large')
						ax[i][1].set_xlabel("frequency [MHz]", fontsize = 'large')
						i += 1
						
			
				fig.tight_layout()

	
			fig2, ax2 = plt.subplots(1, 2, sharex=False, figsize=(10, 5))
			E = np.logspace(17, 20, num = 100)
			theta = np.arange(np.deg2rad(20), np.deg2rad(90), np.deg2rad(.5))
			xplot = np.zeros(len(E)*len(theta))
			yplot = np.zeros(len(E)*len(theta))
			zplot = np.zeros(len(E)*len(theta))
			i = 0
			for energy in E:
				for view_angle in theta:
					zplot[i] = minimizer_theta_component([view_angle, energy], [R], n_index, time_shift_Vpol, hilbert = False, fit = 'both')
					xplot[i] = energy
					yplot[i] = view_angle
					i += 1
				
					
			fmin = minimizer_theta_component([reconstructed_viewing_angle, reconstructed_energy_theta], R, n_index, time_shift_Vpol, hilbert = False, fit = 'both')
			plot_range = [fmin-10 , fmin + 50]
			
			cs = ax2[0].pcolor(xplot.reshape(len(E), len(theta)), np.rad2deg(yplot).reshape(len(E), len(theta)), zplot.reshape(len(E), len(theta)), cmap = 'gnuplot_r', vmin = plot_range[0], vmax = plot_range[1])
			ax2[0].axhline(np.rad2deg(viewing_angle), color = 'red',
						   label = 'simulated viewing angle', linewidth = 2)
			
			
			ax2[0].axhline(np.rad2deg(reconstructed_viewing_angle), color = 'orange', 
						   label = 'reconstructed viewing angle', linewidth = 2)
			ax2[0].axhline(cherenkov_angle, color = 'white',
						  label = 'cherenkov angle', linewidth = .5)
			ax2[0].axvline(reconstructed_energy_theta, color = 'orange', linewidth = 2)
			ax2[0].legend(fontsize = 'large')
			ax2[0].set_xlabel("energy theta component", fontsize = 'large')
			ax2[0].set_ylabel("viewing angle [degrees]", fontsize = 'large')
			fig2.colorbar(cs, ax=ax2[0])
			ax2[0].set_xscale('log')
			
			polarization = np.arange(np.deg2rad(1), np.deg2rad(90), np.deg2rad(1))
			xplot = []
			yplot = []
			for ip, p in enumerate(polarization):
				xplot.append(p)
				yplot.append(minimizer([reconstructed_energy_theta*np.tan(p)], R, 
												 n_index, time_shift_Hpol, time_shift_Hpol_neg, minimizer = True))
			ax2[1].plot(np.rad2deg(xplot), -1*np.array(yplot), 'o-')
			ax2[1].set_ylim((max(-1*np.array(yplot))- 50, max(-1*np.array(yplot))))
			ax2[1].set_ylabel("probability (not normalized)", fontsize = 'large')
			ax2[1].axvline(np.rad2deg(np.arctan2((reconstructed_energy_phi), (reconstructed_energy_theta))), 
						   label = 'reconstructed polarization', color = 'orange', linewidth = 2)
			ax2[1].set_xlabel("polarization angle [degrees]", fontsize = 'large')
			ax2[1].set_ylim((max(-1*np.array(yplot))-1, max(-1*np.array(yplot))))
			ax2[1].legend(fontsize = 'large')
			fig2.tight_layout()
                
	def end(self):
		pass
			
			
				
			
			
			
			
			

				
			
				
			
			
