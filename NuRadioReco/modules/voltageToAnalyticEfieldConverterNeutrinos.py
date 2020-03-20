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
import NuRadioReco.modules.channelResampler
from NuRadioReco.utilities.geometryUtilities import get_time_delay_from_direction

from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalGen.parametrizations import get_time_trace

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()



class voltageToAnalyticEfieldConverterNeutrinos:
	"""
	this module reconstructs the electric-field by forward folding an analytic 
	pulse function through the antenna.
	
	This module is specified for neutrinos. The function used to describe the 
	electric field is the Alvarez2009 parametrization. 
	
	Before the reconstruction, the channels are upsampled.
	
	The module is optimized for an electric-field reconstruction using two 
	channels. First, the energy and the viewing angle are fitted using a Vpol 
	antenna. Then the viewing angle is fixed in the Hpol antenna, and the energy
	is fitted. 
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
			passband = [200* units.MHz, 500 * units.MHz], bandpass = True, 
			use_MC=True, attenuation = True,  attenuation_model = 'GL1', 
			sampling_rate = 5*units.GHz, parametrization = 'Alvarez2009'):
	
		"""
		run method. This function is executed for each event
		
		Parameters
		----------
		evt
		station 
		det
		icemodel
		shower_type
		debug: bool
			if True debug plotting is enables
		channel_Vpol: int
			the channel id for the VPol used for the electric field reconstruction
		channel_Hpol: int
			the channel id for the VPol used for the electric field reconstruction
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
		sampling_rate:
			sampling rate for upsampling the channel traces
		parametrization:
			parametrization model used for the reconstruction. At the moment 
			only optimized for Alvarez2009
		"""
		
		use_channels_efield = [channel_Vpol, channel_Hpol]
		if phasing: use_channels_efield += channels_phasing_Vpol + channels_phasing_Hpol
		use_channels_efield.sort()
		
		station_id = station.get_id()
		if use_MC and (station.get_sim_station() is not None):
			#find ray type with maximum amplitude
			efield_max = 0 
			for i_efield, efield in enumerate(station.get_sim_station().get_electric_fields()):
				if efield.get_channel_ids()[0] in use_channels_efield:
					max_efield = max(np.sqrt( efield.get_trace()[1]**2 + efield.get_trace()[2]**2))
					if max_efield > efield_max:
						raytype = efield[efp.ray_path_type]
						zenith = efield[efp.zenith]
						azimuth = efield[efp.azimuth]
						efield_max = max_efield
						viewing_angle = efield[efp.nu_viewing_angle]
			ray_solution = ['direct', 'refracted', 'reflected'].index(raytype) + 1
			vertex_position = station.get_sim_station()[stnp.nu_vertex]
			n_index = icemodel.get_index_of_refraction(vertex_position)
			cherenkov_angle = np.rad2deg(np.arccos(1./n_index))
		
		phasing_dict = {channel_Vpol: channels_phasing_Vpol + [channel_Vpol], 
						channel_Hpol: channels_phasing_Hpol + [channel_Hpol]}
		noise_RMS = det.get_noise_RMS(station.get_id(), 0) #assume noise is the same in all channels
		
		channelResampler.run(evt, station, det, sampling_rate=sampling_rate) # upsample the traces 

		sampling_rate = station.get_channel(0).get_sampling_rate()
		
		dt = 1./sampling_rate
		order = 10
		efield_antenna_factor, V, V_timedomain = get_array_of_channels(station,
																	   use_channels_efield,
																	   det, 
																	   zenith, 
																	   azimuth, 
																	   antennapattern.AntennaPatternProvider(), 
																	   time_domain=True) 
		
		N_V_time_domain = len(V_timedomain[0])
		ff = np.fft.rfftfreq(N_V_time_domain, dt)
		b, a = signal.butter(order, passband, 'bandpass', analog=True)
		w, h = signal.freqs(b, a, ff)


		def minimizer_theta_component(params, R, n_index, toffset, hilbert = False):

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
			"""
			N = 50
			if len(params) == 1:
				E_theta = params[0]
				E_phi = E_theta
				theta = np.deg2rad(55)
			else:
				E_theta = params[0]
				theta = params[1]

			if phasing: E_theta *= len(phasing_dict[channel_Vpol])*np.sqrt(2)
			time_trace_thet = get_time_trace(E_theta, theta, N_V_time_domain, 
											 dt, shower_type, n_index, R,  parametrization)
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
					if bandpass:
						f = fft.time2freq(trace, 1/dt) * h
						trace = fft.freq2time(f, 1/dt)
					k = np.argmax(trace)
					analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([trace_freq_thet, trace_freq_phi]), axis = 0)
					analytic_trace = fft.freq2time(analytic_trace_fft, 1/dt)

					if hilbert: 
						chi2 += np.sum(abs((abs(signal.hilbert(np.roll(analytic_trace, 
																	   int(toffset))))[k - N: k+2*N] - abs(signal.hilbert(trace))[k - N: k+2*N]))**2)
					else:
						chi2 += np.sum((np.roll(analytic_trace, 
												int(toffset))[k-N:k+3*N] - trace[k-N:k+3*N])**2)
					
			return chi2/(2*noise_RMS**2)

		def minimizer(params, R, n_index, toffset, 
					  hilbert = False, minimizer = True):
			
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
			minimizer: bool
				if True, the output of minimization is returned. 
				if False, the sign of the Hpol component is returned. 
			"""
			N = 50
		
			if len(params) == 3:
				E_theta, theta, E_phi = params
			else:
				E_phi = params[0]
				E_theta = results.x[0]
				theta = results.x[1]

			if phasing: E_phi *= len(phasing_dict[channel_Hpol])*np.sqrt(2)
			chi2 = np.inf

			for sign_tmp in [-1, 1]:
				tmp = 0
				time_trace_thet = get_time_trace(E_theta, theta, N_V_time_domain, 
												 dt, shower_type, n_index, R, parametrization)
				trace_freq_thet = fft.time2freq(time_trace_thet, 1/dt)
				time_trace_phi = sign_tmp*get_time_trace(E_phi, theta, N_V_time_domain, 
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
						k = np.argmax(trace)
					if use_channels_efield[iCh] == channel_Hpol:
						if phasing: trace = get_phased_trace(use_channels_efield[iCh],
															 phasing_dict[use_channels_efield[iCh]])
						analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([trace_freq_thet,
																						   trace_freq_phi]), axis = 0)
						analytic_trace = fft.freq2time(analytic_trace_fft, 1/dt)

						if bandpass: 
							f = fft.time2freq(trace, 1/dt) * h
							trace = fft.freq2time(f, 1/dt)

						if hilbert: 
							tmp += np.sum(abs((abs(signal.hilbert(np.roll(analytic_trace,
																		  int(toffset))))[k - N: k+N] - abs(signal.hilbert(trace))[k - N: k+N]))**2)
						else:
							tmp += np.sum((abs(np.roll(analytic_trace, 
													   int(toffset))[k - N: k+N] - trace[k - N: k+N]))**2)
							

				if tmp < chi2:
					chi2 = tmp
					sign = sign_tmp

			if not minimizer:
				return sign
			else:
				return chi2/(2*noise_RMS**2)
				

		def get_phased_trace( channel_id, channel_ids_phasing, band_pass = bandpass):
			phased_trace= np.zeros(len(V_timedomain[0]))
			for iCh, trace in enumerate(V_timedomain): 
				if use_channels_efield[iCh] in channel_ids_phasing:
					positions = np.array([det.get_relative_position(101,channel_id),
										  det.get_relative_position(101,use_channels_efield[iCh])])
					time_delay1 = get_time_delay_from_direction(zenith, azimuth, 
																positions[0], n = n_index)
					time_delay2 = get_time_delay_from_direction(zenith, azimuth, 
																positions[1], n = n_index)
					roll = int(-(time_delay2-time_delay1))
					trace_phasing = np.roll(trace, roll)
					phased_trace += trace_phasing
			return phased_trace

		prop = propagation.get_propagation_module(name = 'analytic')
		antenna_position = det.get_relative_position(101, channel_Vpol)
		r = prop(vertex_position, antenna_position, icemodel, 
				 attenuation_model = attenuation_model)
		r.find_solutions()
		n = r.get_number_of_solutions()
		for iS in range(n):
			
			if r.get_solution_type(iS) == ray_solution:
				R = r.get_path_length(iS)
				attn = r.get_attenuation(iS, ff, 0.5*sampling_rate)

				
		time_trace = get_time_trace(10**18,viewing_angle, N_V_time_domain, 
									dt, shower_type, n_index, R, parametrization) 
		trace_freq = fft.time2freq(time_trace, 1/dt)

		if attenuation:
				trace_freq *= attn
		if bandpass:
				trace_freq *= h
		
		analytic_trace_fft = np.sum(efield_antenna_factor[use_channels_efield.index(channel_Vpol)]*np.array([trace_freq,
																											 trace_freq]), axis = 0)
		for iCh, trace in enumerate(V_timedomain):
			if use_channels_efield[iCh] == channel_Vpol: # use Vpol to determine offset
				if phasing: trace = get_phased_trace(use_channels_efield[iCh], phasing_dict[use_channels_efield[iCh]])
				if bandpass:
					f = fft.time2freq(trace, 1/dt) * h
					trace = fft.freq2time(f, 1/dt)
				
				analytic_trace = fft.freq2time(analytic_trace_fft, 1/dt)
	
				corr = signal.hilbert(signal.correlate(trace, analytic_trace))

				toffset = np.argmax(corr) - (len(corr)/2) 
		
		options = {'maxiter':500, 'disp':True}
		bnds = ((10**17, 10**20))
		results = opt.minimize(minimizer_theta_component, x0 = (10**18), 
							   method = 'Nelder-Mead', args = (R, n_index, toffset),
							   bounds = bnds, options = options) #Fit amplitude theta component with viewing angle fixed to 55 degrees
		print("results", results)

		bnds = ((10**17, 10**19), (np.deg2rad(40), np.deg2rad(70)))
		results = opt.minimize(minimizer_theta_component, 
							   x0 = [results.x[0], np.deg2rad(55)], 
							   method = 'Nelder-Mead', args = (R, n_index, toffset),
							   bounds = bnds) #Fit amplitude theta component and viewing angle
		print("results ", results)

		
		
		#adapt the time for the fit for the shift of the Hpol wrt Vpol
		positions = np.array([det.get_relative_position(101,channel_Vpol),
							  det.get_relative_position(101,channel_Hpol)])
		time_delay1 = get_time_delay_from_direction(zenith, azimuth, positions[0], 
													n = n_index)
		time_delay2 = get_time_delay_from_direction(zenith, azimuth, positions[1], 
													n = n_index)
		time_shift_Hpol = int(-(time_delay2-time_delay1))
		time_shift_Hpol += toffset
		
		bnds = ((10**17, 10**19))
		results1 = opt.minimize(minimizer, x0=(results.x[0]), method = 'Nelder-Mead', 
								args = (R, n_index, time_shift_Hpol, False, False, True), 
								bounds = bnds, options = options) #Fit amplitude phi component
		print("results", results1)


		
		#store electricfield 
		time_trace_theta = get_time_trace(results.x[0], results.x[1], N_V_time_domain, 
										  dt, shower_type, n_index, R, parametrization)
		time_trace_phi = get_time_trace(results1.x[0], results.x[1], N_V_time_domain, 
										dt, shower_type, n_index, R, parametrization)
		time_trace_theta = np.roll(time_trace_theta, int(toffset))
		time_trace_phi = np.roll(time_trace_phi, int(toffset))
		station_trace = np.array([np.zeros_like(time_trace_theta), time_trace_phi, time_trace_phi])
		electric_field = NuRadioReco.framework.electric_field.ElectricField([channel_Vpol, channel_Hpol])
		electric_field.set_trace(station_trace, sampling_rate)
		electric_field.set_parameter(efp.polarization_angle, np.arctan2(np.sqrt(results1.x[0]), np.sqrt(results.x[0])))
		
		# figure out the timing of the electric field
		voltages_from_efield = trace_utilities.get_channel_voltage_from_efield(station, 
																			   electric_field, 
																			   [channel_Vpol, channel_Hpol], 
																			   det, 
																			   zenith, 
																			   azimuth, 
																			   self.antenna_provider, False)
		correlation = np.zeros(voltages_from_efield.shape[1] + station.get_channel(channel_Vpol).get_trace().shape[0] - 1)
		channel_trace_start_times = []
		for channel_id in [channel_Vpol, channel_Hpol]:
			channel_trace_start_times.append(station.get_channel(channel_id).get_trace_start_time())
		average_trace_start_time = np.average(channel_trace_start_times)
		for i_trace, v_trace in enumerate(voltages_from_efield):
			channel = station.get_channel([channel_Vpol, channel_Hpol][i_trace])
			time_shift = get_time_delay_from_direction(zenith, 
													   azimuth, 
													   det.get_relative_position(station.get_id(), 
																						  [channel_Vpol, channel_Hpol][i_trace])) - (channel.get_trace_start_time() - average_trace_start_time)
			voltage_trace = np.roll(np.copy(v_trace), int(time_shift * electric_field.get_sampling_rate()))
			correlation += signal.correlate(voltage_trace, channel.get_trace())
		t_offset = (np.arange(0, correlation.shape[0]) - channel.get_trace().shape[0]) / electric_field.get_sampling_rate()
		electric_field.set_trace_start_time(-t_offset[np.argmax(correlation)] + average_trace_start_time)
		station.add_electric_field(electric_field)


		if debug:
			E_theta = results.x[0]
			E_phi = results1.x[0]
			if phasing: E_theta *= len(phasing_dict[channel_Vpol])*np.sqrt(2)
			if phasing: E_phi *= len(phasing_dict[channel_Hpol])*np.sqrt(2)
			view_angle = results.x[1]
			time_trace_thet = get_time_trace(E_theta, view_angle, N_V_time_domain, 
											 dt, shower_type, n_index, R, parametrization)
			sign_phi_component = minimizer([E_theta, view_angle, results1.x[0]], 
										   R, n_index, toffset, minimizer = False)
			time_trace_phi = sign_phi_component * get_time_trace(E_phi, view_angle, 
																 N_V_time_domain, dt, 
																 shower_type, n_index, 
																 R, parametrization)
			trace_freq_thet = fft.time2freq(time_trace_thet, 1/dt)
			trace_freq_phi = fft.time2freq(time_trace_phi, 1/dt)

			if attenuation: 
				trace_freq_thet *= attn
				trace_freq_phi *= attn

			if bandpass: 
				trace_freq_thet *= h
				trace_freq_phi *= h

			fig, ax = plt.subplots(2, 2, sharex=False, figsize=(20, 10))
			N = 50
			for iCh, trace in enumerate(V_timedomain):
				if ((use_channels_efield[iCh] == channel_Vpol) or (use_channels_efield[iCh] == channel_Hpol)):
						if phasing: trace = get_phased_trace(use_channels_efield[iCh], 
															 phasing_dict[use_channels_efield[iCh]])
					
						if bandpass: 
							f = fft.time2freq(trace, 1/dt) * h
							trace = fft.freq2time(f, 1/dt)
						if iCh == use_channels_efield.index(channel_Vpol):
							k = np.argmax(trace)
						if iCh == use_channels_efield.index(channel_Hpol):
							toffset = time_shift_Hpol
						analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([trace_freq_thet, 
																						   trace_freq_phi]), axis = 0)
						time_trace = fft.freq2time(analytic_trace_fft, 1/dt)
						ax[iCh][0].plot(trace, label = 'voltage trace')
						ax[iCh][1].plot(ff/units.MHz, abs(fft.time2freq(trace, 1/dt)), label = 'voltage trace')
						ax[iCh][1].plot(ff/units.MHz, abs(analytic_trace_fft), label = 'fit')
						ax[iCh][0].plot(np.roll(time_trace, int(toffset)), label = 'fit')
						ax[iCh][0].set_xlim(([k - N - 50, k + 2*N + 50]))
						ax[iCh][0].axvline([k+2*N], color = 'purple', label = 'fitting area')
						ax[iCh][0].axvline([k-N], color = 'purple')
						ax[iCh][0].axvline([k], color = 'green', label = 'maximum')
						ax[iCh][0].legend()
						ax[iCh][1].legend()
						ax[iCh][0].set_xlabel("time [ns]")
						ax[iCh][1].set_xlabel("frequecy [MHz]")
				
				fig.tight_layout()
			
			fig2, ax2 = plt.subplots(1, 2, sharex=False, figsize=(20, 10))
			E = np.logspace(17, 20, num = 50)
			theta = np.arange(np.deg2rad(30), np.deg2rad(80), np.deg2rad(1))
			xplot = np.zeros(len(E)*len(theta))
			yplot = np.zeros(len(E)*len(theta))
			zplot = np.zeros(len(E)*len(theta))
			i = 0
			for energy in E:
				for view_angle in theta:
					zplot[i] = np.exp(-1*minimizer_theta_component([energy, view_angle], 
																   R, n_index, toffset, 
																   minimizer = True))
					xplot[i] = energy
					yplot[i] = view_angle
					i += 1
			
			
			cs = ax2[0].scatter(xplot, np.rad2deg(yplot), c = zplot, cmap = 'gnuplot2', 
								vmin = min(zplot), vmax = max(zplot), s = 100, alpha = .8)
			ax2[0].axhline(np.rad2deg(viewing_angle), color = 'red', 
						   label = 'simulated viewing angle', linewidth = 3)
			ax2[0].axhline(np.rad2deg(results.x[1]), color = 'orange', 
						   label = 'reconstructed values', linewidth = 3)
			ax2[0].axvline(results.x[0], color = 'orange', linewidth = 3)
			ax2[0].legend(fontsize = 'xx-large')
			ax2[0].set_xlabel("energy theta component", fontsize = 'xx-large')
			ax2[0].set_ylabel("viewing angle [degrees]", fontsize = 'xx-large')
			fig2.colorbar(cs, ax=ax2[0])
			ax2[0].set_xscale('log')
			
			polarization = np.arange(np.deg2rad(1), np.deg2rad(70), np.deg2rad(1))
			xplot = []
			yplot = []
			for ip, p in enumerate(polarization):
				xplot.append(p)
				yplot.append(np.exp(-1*minimizer([results.x[0]*np.tan(p)**2], R, 
												 n_index, toffset, minimizer = True)))
			ax2[1].plot(np.rad2deg(xplot), yplot, 'o-')
			ax2[1].set_ylabel("probability (not normalized)", fontsize = 'xx-large')
			print("results x1", results1.x[0])
			ax2[1].axvline(np.rad2deg(np.arctan2(np.sqrt(results1.x[0]), np.sqrt(results.x[0]))), 
						   label = 'reconstructed polarization', color = 'orange', linewidth = 3)
			ax2[1].set_xlabel("polarization angle [degrees]", fontsize = 'xx-large')
			ax2[1].legend(fontsize = 'xx-large')
			fig2.tight_layout()
		

		
	def end(self):
		pass
			
			
				
			
			
			
			
			

				
			
				
			
			
