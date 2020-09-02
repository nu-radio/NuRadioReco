from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalGen import askaryan as signalgen
from NuRadioReco.detector import detector
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from radiotools import coordinatesystems as cstrans
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.detector import antennapattern
from scipy import signal
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator
import NuRadioReco.modules.io.eventReader
import datetime
from NuRadioReco.framework.parameters import stationParameters as stnp
from radiotools import helper as hp
import numpy as np
import logging
logger = logging.getLogger("sim")
from NuRadioReco.detector import antennapattern

raytracing = {}
eventreader = NuRadioReco.modules.io.eventReader.eventReader()

ice = medium.get_ice_model('greenland_simple')
prop = propagation.get_propagation_module('analytic')
attenuate_ice = True
#sampling_rate = 5

#det = detector.Detector(json_filename = "/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/station_amp.json")
#det.update(datetime.datetime(2018, 1, 1))



hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

class simulation():
	
	def __init__(self):
		self.antenna_provider = antennapattern.AntennaPatternProvider()
		return 
	
	def begin(self, det, station, use_channels):
		""" initialize filter and amplifier """
		print("hallo")
		sim_to_data = True

		channl = station.get_channel(use_channels[0])
		self._n_samples = channl.get_number_of_samples()
		self._sampling_rate = channl.get_sampling_rate()
		self._dt = 1./self._sampling_rate
		
	
		self._ff = np.fft.rfftfreq(self._n_samples, self._dt)
		tt = np.arange(0, self._n_samples * self._dt, self._dt)
        
		mask = self._ff > 0
		order = 8
		passband = [50* units.MHz, 1150 * units.MHz]
		b, a = signal.butter(order, passband, 'bandpass', analog=True)
		w, ha = signal.freqs(b, a, self._ff[mask])
		order = 10
		passband = [0* units.MHz, 700 * units.MHz]
		b, a = signal.butter(order, passband, 'bandpass', analog=True)
		w, hb = signal.freqs(b, a, self._ff[mask])
		fa = np.zeros_like(self._ff, dtype=np.complex)
		fa[mask] = ha
		fb = np.zeros_like(self._ff, dtype = np.complex)
		fb[mask] = hb
		self._h = fb*fa
		self._amp = {}
		for channel_id in use_channels:
			self._amp[channel_id] = {}
			
			self._amp[channel_id] = hardwareResponseIncorporator.get_filter(self._ff, station.get_id(), channel_id, det, sim_to_data)
				
		self._f = np.ones_like(self._ff)
		passband = [20* units.MHz, 1150 * units.MHz]
		self._f[np.where(self._ff < passband[0])] = 0.
		self._f[np.where(self._ff > passband[1])] = 0.
		
		pass


	def _calculate_polarization_vector(self, channel_id, iS):
		polarization_direction = np.cross(raytracing[channel_id][iS]["launch vector"], np.cross(self._shower_axis, raytracing[channel_id][iS]["launch vector"]))
		polarization_direction /= np.linalg.norm(polarization_direction)
		cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*raytracing[channel_id][iS]["launch vector"]))
		return cs.transform_from_ground_to_onsky(polarization_direction)


	def simulation(self, det, station, vertex_x, vertex_y, vertex_z, nu_zenith, nu_azimuth, energy, use_channels, fit = 'seperate', first_iter = False):
		
		
		polarization = True
		reflection = True

			
		x1 = [vertex_x, vertex_y, vertex_z]
		fhad = station.get_sim_station()[stnp.inelasticity]
		
		self._shower_axis = -1 * hp.spherical_to_cartesian(nu_zenith, nu_azimuth)
		n_index = ice.get_index_of_refraction(x1)
		cherenkov_angle = np.arccos(1. / n_index)
		
		
#		sampling_rate_detector = det.get_sampling_frequency(station.get_id(), 0)
		
		global raytracing ## define dictionary to store the ray tracing properties	
		global launch_vectors
		global launch_vector
		if(first_iter):
			launch_vectors = []

			for channel_id in use_channels:
				raytracing[channel_id] = {}
				x2 = det.get_relative_position(station.get_id(), channel_id) + det.get_absolute_position(station.get_id())
				r = prop(x1, x2, ice, 'GL1')

				r.find_solutions()
				if(not r.has_solution()):
					print("warning: no solutions")
					continue

				# loop through all ray tracing solution
				import matplotlib.pyplot as plt
		#		fig = plt.figure()
				for iS in range(r.get_number_of_solutions()):
					raytracing[channel_id][iS] = {}
					self._launch_vector = r.get_launch_vector(iS)
					raytracing[channel_id][iS]["launch vector"] = self._launch_vector
					
					R = r.get_path_length(iS)
					raytracing[channel_id][iS]["trajectory length"] = R
					T = r.get_travel_time(iS)  # calculate travel time
					if (R == None or T == None):
						continue
					raytracing[channel_id][iS]["travel time"] = T
					receive_vector = r.get_receive_vector(iS)
					zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
					raytracing[channel_id][iS]["receive vector"] = receive_vector
					raytracing[channel_id][iS]["zenith"] = zenith
					raytracing[channel_id][iS]["azimuth"] = azimuth
				
					attn = r.get_attenuation(iS, self._ff)#, 0.5 * self._sampling_rate)
				
					raytracing[channel_id][iS]["attenuation"] = attn

					
					zenith_reflections = np.atleast_1d(r.get_reflection_angle(iS))
					raytracing[channel_id][iS]["reflection angle"] = zenith_reflections
					viewing_angle = hp.get_angle(self._shower_axis,raytracing[channel_id][iS]["launch vector"])
					print("VIEWING ANGLE", np.rad2deg(viewing_angle))
					if channel_id == 9: 
						launch_vectors.append( self._launch_vector)
	#			fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/attn.pdf")


		traces = {}
		timing = {}

		for channel_id in use_channels:
			#print("CHANNEL", channel_id)

			traces[channel_id] = {}
			timing[channel_id] = {}
			for iS in raytracing[channel_id]:


				#print("IS", iS)
				traces[channel_id][iS] = {}
				timing[channel_id][iS] = {}
				viewing_angle = hp.get_angle(self._shower_axis,raytracing[channel_id][iS]["launch vector"])

				spectrum = signalgen.get_frequency_spectrum(
				energy * fhad, viewing_angle, self._n_samples, self._dt, "HAD", n_index, raytracing[channel_id][iS]["trajectory length"],
				'Alvarez2009')
			

				import matplotlib.pyplot as plt
				
				# apply frequency dependent attenuation
				if attenuate_ice:
					spectrum *= raytracing[channel_id][iS]["attenuation"]
					
                #print(stop)
				if polarization:
					polarization_direction_onsky = self._calculate_polarization_vector(channel_id, iS)

					cs_at_antenna = cstrans.cstrafo(*hp.cartesian_to_spherical(*raytracing[channel_id][iS]["receive vector"]))
					polarization_direction_at_antenna = cs_at_antenna.transform_from_onsky_to_ground(polarization_direction_onsky)
					logger.debug('receive zenith {:.0f} azimuth {:.0f} polarization on sky {:.2f} {:.2f} {:.2f}, on ground @ antenna {:.2f} {:.2f} {:.2f}'.format(
						raytracing[channel_id][iS]["zenith"] / units.deg, raytracing[channel_id][iS]["azimuth"] / units.deg, polarization_direction_onsky[0],
						polarization_direction_onsky[1], polarization_direction_onsky[2],
						*polarization_direction_at_antenna))
				eR, eTheta, ePhi = np.outer(polarization_direction_onsky, spectrum)

				## correct for reflection 
				r_theta = None
				r_phi = None
				
				n_surface_reflections = np.sum(raytracing[channel_id][iS]["reflection angle"] != None)
				if reflection:
					x2 = det.get_relative_position(station.get_id(), channel_id) + det.get_absolute_position(station.get_id())
					for zenith_reflection in raytracing[channel_id][iS]["reflection angle"]:  # loop through all possible reflections
							if(zenith_reflection is None):  # skip all ray segments where not reflection at surface happens
								continue
							r_theta = geo_utl.get_fresnel_r_p(
								zenith_reflection, n_2=1., n_1=ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))
							r_phi = geo_utl.get_fresnel_r_s(
								zenith_reflection, n_2=1., n_1=ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))

							eTheta *= r_theta
							ePhi *= r_phi
							logger.debug("ray hits the surface at an angle {:.2f}deg -> reflection coefficient is r_theta = {:.2f}, r_phi = {:.2f}".format(zenith_reflection / units.deg,
								r_theta, r_phi))
				
							
		
				
                ##### Get filter (this is the filter used for the trigger for RNO-G)
				
                #### get antenna respons for direction
				efield_antenna_factor = trace_utilities.get_efield_antenna_factor(station, self._ff, [channel_id], det, raytracing[channel_id][iS]["zenith"],  raytracing[channel_id][iS]["azimuth"], self.antenna_provider)
				
                ### convolve efield with antenna reponse
				analytic_trace_fft = np.sum(efield_antenna_factor[0] * np.array([eTheta, ePhi]), axis = 0)
                ### filter the trace
			
				
				analytic_trace_fft *=self._h
				
		#### add amplifier

				analytic_trace_fft *= self._amp[channel_id]#hardwareResponseIncorporator.get_filter(self._ff, station.get_id(), channel_id, det, sim_to_data)
            # zero first bins to avoid DC offs

				analytic_trace_fft[0] = 0
		#### filter becuase of amplifier response 
				

				analytic_trace_fft *= self._f
				
                ### store traces
				traces[channel_id][iS] = fft.freq2time(analytic_trace_fft, 1/self._dt)
                ### store timing
				timing[channel_id][iS] =raytracing[channel_id][iS]["travel time"]
                
		if(first_iter): ## seelct viewing angle due to channel with largest amplitude 
		
			maximum_channel = 0
			for i in range(iS+1):
				maximum_trace = max(abs(traces[9][i]))
				#print("iS", iS)
				#print("maximum trace", maximum_trace)
				#print("launch_vector", launch_vectors[i])
				if maximum_trace > maximum_channel:
					launch_vector = launch_vectors[i]
					maximum_channel = maximum_trace

		


		
		return traces, timing, launch_vector       	











