import numpy as np
import scipy.signal
import pickle
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft
import NuRadioReco.utilities.io_utilities
import NuRadioReco.framework.electric_field
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import radiotools.helper as hp
import NuRadioMC.SignalProp.analyticraytracing
import NuRadioMC.utilities.medium

class neutrino2DVertexReconstructor:

    def __init__(self, lookup_table_location):
        """
        Constructor for the vertex reconstructor

        Parameters
        --------------
        lookup_table_location: string
            path to the folder in which the lookup tables for the signal travel
            times are stored
        """
        self.__lookup_table_location = lookup_table_location

    def begin(self, station_id, channel_ids, detector, passband=None):
        """
        General settings for vertex reconstruction

        Parameters
        -------------
        station_id: integer
            ID of the station to be used for the reconstruction
        channel_ids: array of integers
            IDs of the channels to be used for the reconstruction
        detector: Detector or GenericDetector
            Detector description for the detector used in the reconstruction
        filter_passband: array of float or None
            Passband of the filter that should be applied to channel traces before
            calculating the correlation. If None is passed, no filter is applied
        """
        first_channel_position = detector.get_relative_position(station_id, channel_ids[0])
        for channel_id in channel_ids:
            pos = detector.get_relative_position(station_id, channel_id)
            # Check if channels are on the same string. Allow some tolerance for
            # uncertainties from deployment
            if np.abs(pos[0] - first_channel_position[0]) > 1.*units.m or np.abs(pos[1] - first_channel_position[1]) > 1.*units.m:
                raise ValueError('All channels have to be on the same string')
        self.__detector = detector
        self.__channel_ids = channel_ids
        self.__station_id = station_id
        self.__channel_pairs = []
        for i in range(len(channel_ids) - 1):
            for j in range(i+1, len(channel_ids)):
                self.__channel_pairs.append([channel_ids[i], channel_ids[j]])
        self.__lookup_table = {}
        self.__header = {}
        self.__passband = passband
        self.__ray_types = [['direct', 'direct'], ['reflected', 'reflected'], ['refracted', 'refracted'], ['direct', 'reflected'],['reflected','direct'],['direct','refracted'],['refracted','direct']]
        for channel_id in channel_ids:
            channel_z = abs(detector.get_relative_position(station_id, channel_id)[2])
            if channel_z not in self.__lookup_table.keys():
                f = NuRadioReco.utilities.io_utilities.read_pickle('{}/lookup_table_{}.p'.format(self.__lookup_table_location, int(abs(channel_z))))
                self.__header[int(channel_z)] = f['header']
                self.__lookup_table[int(abs(channel_z))] = f['antenna_{}'.format(channel_z)]


    def run(self, station, max_distance, z_width, grid_spacing, direction_guess=None, debug=False):
        """
        Execute the 2D vertex reconstruction

        Parameters
        ---------------
        station: Station
            The station for which the vertex shall be reconstructed
        max_distance: number
            Maximum distance up to which the vertex position shall be searched
        z_width: number
            Vertical size of the search area. If direction_guess is specified, a
            stripe of z_width to each side of the initial direction will be searched.
            If direction_guess is not specified, z_width is the maximum depth up
            to which the vertex will be searched.
        grid_spacing: number
            Distance between two points of the grid on which the vertex is searched
        direction_guess: number, defaults to None
            Zenith for an initial guess of the vertex direction. If specified,
            a strip if width 2*z_width around the guessed direction will be searched
        debug: boolean
            If True, debug plots will be produced
        """
        distances = np.arange(50.*units.m, max_distance, grid_spacing)
        if direction_guess is None:
            heights = np.arange(-z_width, 0, grid_spacing)
        else:
            heights = np.arange(-z_width, z_width, grid_spacing)
        x_0, z_0 = np.meshgrid(distances, heights)
        if direction_guess is None:
            x_coords = x_0
            z_coords = z_0
        else:
            x_coords = np.cos(direction_guess-90.*units.deg) * x_0 + np.sin(direction_guess-90.*units.deg) * z_0
            z_coords = -np.sin(direction_guess-90.*units.deg) * x_0 + np.cos(direction_guess-90.*units.deg) * z_0

        correlation_sum = np.zeros(x_coords.shape)

        corr_range = 50.*units.ns
        max_corr_index = None
        for i_pair, channel_pair in enumerate(self.__channel_pairs):
            ch1 = station.get_channel(channel_pair[0])
            ch2 = station.get_channel(channel_pair[1])

            #snr1 = ch1.get_parameter(chp.SNR)['peak_2_peak_amplitude']
            #snr2 = ch2.get_parameter(chp.SNR)['peak_2_peak_amplitude']
            snr1 = np.max(np.abs(ch1.get_trace()))
            snr2 = np.max(np.abs(ch2.get_trace()))
            if snr1 == 0 or snr2 == 0:
                continue
            spec1 = np.copy(ch1.get_frequency_spectrum())
            spec2 = np.copy(ch2.get_frequency_spectrum())
            if self.__passband is not None:
                b, a = scipy.signal.butter(10, self.__passband, 'bandpass', analog=True)
                w, h = scipy.signal.freqs(b, a, ch1.get_frequencies())
                spec1 *= h
                spec2 *= h

            trace1 = fft.freq2time(spec1, ch1.get_sampling_rate())
            t_max1 = ch1.get_times()[np.argmax(np.abs(trace1))]
            trace2 = fft.freq2time(spec2, ch2.get_sampling_rate())
            t_max2 = ch2.get_times()[np.argmax(np.abs(trace2))]
            if snr1 > snr2:
                trace1[np.abs(ch1.get_times()-t_max1)>corr_range] = 0
                t_max = t_max1
                max_channel = ch1
            else:
                trace2[np.abs(ch2.get_times()-t_max2)>corr_range] = 0
                t_max = t_max2
                max_channel = ch2
            self.__correlation = np.abs(scipy.signal.correlate(trace1, trace2))
            if np.sum(np.abs(self.__correlation)) > 0:
                self.__correlation /= np.sum(np.abs(self.__correlation))
            corr_snr = np.max(self.__correlation)/np.mean(self.__correlation[self.__correlation>0])
            arg_max_corr = np.argmax(self.__correlation)
            toffset = -(np.arange(0, self.__correlation.shape[0]) - self.__correlation.shape[0] / 2.) / ch1.get_sampling_rate()
            self.__sampling_rate = ch1.get_sampling_rate()
            self.__channel_pair = channel_pair
            self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_pair[0]), self.__detector.get_relative_position(self.__station_id, channel_pair[1])]
            correlation_array = np.zeros_like(correlation_sum)
            for i_ray in range(len(self.__ray_types)):
                self.__current_ray_types = self.__ray_types[i_ray]
                correlation_array = np.maximum(self.get_correlation_array_2d(x_coords, z_coords), correlation_array)
            if np.max(correlation_array) > 0:
                correlation_sum += correlation_array / np.max(correlation_array) * corr_snr


            max_corr_index = np.unravel_index(np.argmax(correlation_sum), correlation_sum.shape)
            max_corr_r = x_coords[max_corr_index[0]][max_corr_index[1]]
            max_corr_z = z_coords[max_corr_index[0]][max_corr_index[1]]

            if debug:

                fig1 = plt.figure(figsize=(12,4))
                fig2 = plt.figure(figsize=(8,12))
                ax1_1 = fig1.add_subplot(1, 3, 1)
                ax1_2 = fig1.add_subplot(1, 3, 2, sharey=ax1_1)
                ax1_3 = fig1.add_subplot(1, 3, 3)
                ax1_1.plot(ch1.get_times(), ch1.get_trace()/units.mV, c='C0', alpha=.3)
                ax1_2.plot(ch2.get_times(), ch2.get_trace()/units.mV, c='C1', alpha=.3)
                ax1_1.plot(ch1.get_times()[np.abs(trace1)>0], trace1[np.abs(trace1)>0]/units.mV, c='C0', alpha=1)
                ax1_2.plot(ch2.get_times()[np.abs(trace2)>0], trace2[np.abs(trace2)>0]/units.mV, c='C1', alpha=1)
                ax1_1.set_xlabel('t [ns]')
                ax1_1.set_ylabel('U [mV]')
                ax1_1.set_title('Channel {}'.format(self.__channel_pair[0]))
                ax1_2.set_xlabel('t [ns]')
                ax1_2.set_ylabel('U [mV]')
                ax1_2.set_title('Channel {}'.format(self.__channel_pair[1]))


                ax1_3.plot(toffset, self.__correlation)
                ax1_3.set_title('$SNR_{corr}$=%.2f'%(corr_snr))
                ax1_1.grid()
                ax1_2.grid()
                ax1_3.grid()
                fig1.tight_layout()
                ax2_1 = fig2.add_subplot(211)
                ax2_2 = fig2.add_subplot(212)
                corr_plots = ax2_1.pcolor(x_coords, z_coords, correlation_array)
                sum_plots = ax2_2.pcolor(x_coords, z_coords, correlation_sum)
                fig2.colorbar(corr_plots, ax=ax2_1)
                fig2.colorbar(sum_plots, ax=ax2_2)
                if station.has_sim_station():
                    sim_station = station.get_sim_station()
                    sim_vertex = sim_station.get_parameter(stnp.nu_vertex)
                    ax2_1.axvline(np.sqrt(sim_vertex[0]**2+sim_vertex[1]**2), c='r', linestyle=':')
                    ax2_1.axhline(sim_vertex[2], c='r', linestyle=':')
                    ax2_2.axvline(np.sqrt(sim_vertex[0]**2+sim_vertex[1]**2), c='r', linestyle=':')
                    ax2_2.axhline(sim_vertex[2], c='r', linestyle=':')


                ax2_1.axvline(max_corr_r, c='k', linestyle=':')
                ax2_1.axhline(max_corr_z, c='k', linestyle=':')
                ax2_2.axvline(max_corr_r, c='k', linestyle=':')
                ax2_2.axhline(max_corr_z, c='k', linestyle=':')

                fig2.tight_layout()
                plt.show()

                plt.close('all')
        if max_corr_index is None:
            return
        self.__rec_x = x_coords[max_corr_index[0]][max_corr_index[1]]
        self.__rec_z = z_coords[max_corr_index[0]][max_corr_index[1]]
        station.set_parameter(stnp.vertex_2D_fit, [self.__rec_x, self.__rec_z])
        for channel_id in self.__channel_ids:
            ray_type = self.find_ray_type(station, station.get_channel(channel_id))
            zenith = self.find_receiving_zenith(station, ray_type, channel_id)
            efield = NuRadioReco.framework.electric_field.ElectricField([channel_id], self.__detector.get_relative_position(station.get_id(), channel_id))
            efield.set_parameter(efp.ray_path_type, ray_type)
            if zenith is not None:
                efield.set_parameter(efp.zenith, zenith)
            station.add_electric_field(efield)

        return

    def get_correlation_array_2d(self, x, z):
        channel_pos1 = self.__channel_positions[0]
        channel_pos2 = self.__channel_positions[1]
        d_hor1 = np.sqrt((x - channel_pos1[0])**2 + (channel_pos1[1])**2)
        d_hor2 = np.sqrt((x - channel_pos2[0])**2 + (channel_pos2[1])**2)
        res = self.get_correlation_for_pos(np.array([d_hor1, d_hor2]), z)
        return res

    def get_correlation_for_pos(self, d_hor, z):
        #get_time_func = np.vectorize(self.get_signal_travel_time)
        t1 = self.get_signal_travel_time(d_hor[0], z, self.__current_ray_types[0], self.__channel_pair[0])
        t2 = self.get_signal_travel_time(d_hor[1], z, self.__current_ray_types[1], self.__channel_pair[1])
        delta_t = t1 - t2
        delta_t = delta_t.astype(float)
        corr_index = self.__correlation.shape[0]/2 + np.round(delta_t*self.__sampling_rate)
        res = np.zeros_like(d_hor[0])
        corr_index[np.isnan(corr_index)] = 0
        mask = (~np.isnan(delta_t)) & (corr_index > 0) & (corr_index < self.__correlation.shape[0]) & (~np.isinf(delta_t))
        corr_index[~mask] = 0
        res = np.take(self.__correlation, corr_index.astype(int))
        res[~mask] = 0
        return res

    def get_signal_travel_time(self, d_hor, z, ray_type, channel_id):
        channel_pos = self.__detector.get_relative_position(self.__station_id, channel_id)
        channel_type = int(abs(channel_pos[2]))
        travel_times = np.zeros_like(d_hor)
        mask = np.ones_like(travel_times).astype(bool)
        i_x = np.array(np.round((-d_hor - self.__header[channel_type]['x_min'])/self.__header[channel_type]['d_x'])).astype(int)
        mask[i_x > self.__lookup_table[channel_type][ray_type].shape[0] - 1] = False
        i_z = np.array(np.round((z - self.__header[channel_type]['z_min'])/self.__header[channel_type]['d_z'])).astype(int)
        mask[i_z > self.__lookup_table[channel_type][ray_type].shape[1] - 1] = False
        i_x[~mask] = 0
        i_z[~mask] = 0
        indices = np.array([i_x.flatten(), i_z.flatten()])
        travel_times = self.__lookup_table[channel_type][ray_type][(i_x, i_z)]
        travel_times[~mask] = np.nan
        return travel_times

    def find_ray_type(self, station, ch1):
        corr_range = 50.*units.ns
        ray_types = ['direct', 'refracted', 'reflected']
        ray_type_correlations = np.zeros(3)
        for i_ray_type, ray_type in enumerate(ray_types):
            for channel_id in self.__channel_ids:
                if channel_id != ch1.get_id():
                    ch2 = station.get_channel(channel_id)
                    snr1 = np.max(np.abs(ch1.get_trace()))
                    snr2 = np.max(np.abs(ch2.get_trace()))
                    trace1 = np.copy(ch1.get_trace())
                    t_max1 = ch1.get_times()[np.argmax(np.abs(trace1))]
                    trace2 = np.copy(ch2.get_trace())
                    t_max2 = ch2.get_times()[np.argmax(np.abs(trace2))]
                    if snr1 > snr2:
                        trace1[np.abs(ch1.get_times()-t_max1)>corr_range] = 0
                        t_max = t_max1
                        max_channel = ch1
                    else:
                        trace2[np.abs(ch2.get_times()-t_max2)>corr_range] = 0
                        t_max = t_max2
                        max_channel = ch2
                    correlation = np.abs(scipy.signal.hilbert(scipy.signal.correlate(trace1, trace2)))
                    correlation /= np.sum(np.abs(correlation))
                    t_1 = self.get_signal_travel_time(np.array([self.__rec_x]), np.array([self.__rec_z]), ray_type, ch1.get_id())[0]
                    t_2 = self.get_signal_travel_time(np.array([self.__rec_x]), np.array([self.__rec_z]), ray_type, ch2.get_id())[0]
                    if np.isnan(t_1) or np.isnan(t_2):
                        return None
                    delta_t = t_1 - t_2
                    corr_index = correlation.shape[0]/2 + np.round(delta_t*self.__sampling_rate)
                    if np.isnan(corr_index):
                        return None
                    corr_index = int(corr_index)
                    if corr_index > 0 and corr_index < len(correlation):
                        ray_type_correlations[i_ray_type] += correlation[int(corr_index)]
        return ray_types[np.argmax(ray_type_correlations)]
    def find_receiving_zenith(self, station, ray_type, channel_id):
        solution_types = {1: 'direct',
                          2: 'refracted',
                          3: 'reflected'}
        nu_vertex_2D = station.get_parameter(stnp.vertex_2D_fit)
        nu_vertex = [nu_vertex_2D[0], 0, nu_vertex_2D[1]]
        ray_tracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(
            nu_vertex,
            self.__detector.get_relative_position(station.get_id(),channel_id)+self.__detector.get_absolute_position(station.get_id()),
            NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
        )
        ray_tracer.find_solutions()
        for i_solution, solution in enumerate(ray_tracer.get_results()):
            if solution_types[ray_tracer.get_solution_type(i_solution)] == ray_type:
                receive_vector = ray_tracer.get_receive_vector(i_solution)
                return hp.cartesian_to_spherical(receive_vector[0], receive_vector[1], receive_vector[2])[0]
        return None