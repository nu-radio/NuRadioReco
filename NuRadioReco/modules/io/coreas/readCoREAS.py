import h5py
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
from radiotools import helper as hp
from radiotools import coordinatesystems as cstrafo
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units
import numpy as np
import logging
import time
import os
logger = logging.getLogger('readCoREAS')


class readCoREAS:

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0

    def begin(self, input_files, station_id, n_cores=10, max_distance=2 * units.km, seed=0):
        """
        begin method

        initialize readCoREAS module

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        station_id: station id
            id number of the station
        n_cores: number of cores (integer)
            the number of random core positions to generate for each input file
        max_distance: radius of random cores (double or None)
            if None: max distance is set to the maximum ground distance of the
            star pattern simulation
        """
        self.__input_files = input_files
        self.__station_id = station_id
        self.__n_cores = n_cores
        self.__max_distace = max_distance
        self.__current_input_file = 0
        np.random.seed(seed)

    def run(self, detector, output_mode=0):
        """
        read in CoREAS simulation

        Parameters
        ----------
        output_mode: integer (default 0)
            0: only the event object is returned
            1: the function reuturns the event object, the current inputfilename, the distance between the choosen station and the requested core position,
               and the area in which the core positions are randomly distributed


        """
        while (self.__current_input_file < len(self.__input_files)):
            t = time.time()
            t_per_event = time.time()
            filesize = os.path.getsize(self.__input_files[self.__current_input_file])
            if(filesize < 18456 * 2):
                logger.warning("file {} seems to be corrupt, skipping to next file".format(self.__input_files[self.__current_input_file]))
                self.__current_input_file += 1
                continue
            corsika = h5py.File(self.__input_files[self.__current_input_file], "r")
            logger.info("using coreas simulation {} with E={:2g} theta = {:.0f}".format(self.__input_files[self.__current_input_file],
                                                                                                      corsika['inputs'].attrs["ERANGE"][0] * units.GeV,
                                                                                                      corsika['inputs'].attrs["THETAP"][0]))
            positions = []
            for i, observer in enumerate(corsika['CoREAS']['observers'].values()):
                position = observer.attrs['position']
                positions.append(np.array([-position[1], position[0], 0]) * units.cm)
    #             logger.debug("({:.0f}, {:.0f})".format(position[0], position[1]))
            positions = np.array(positions)

            max_distance = self.__max_distace
            if(max_distance is None):
                max_distance = np.max(np.abs(positions[:, 0:2]))
            area = np.pi * max_distance ** 2

            if(output_mode == 0):
                n_cores = self.__n_cores * 100  # for output mode 1 we want always n_cores in star pattern. Therefore we generate more core positions to be able to select n_cores in the star pattern afterwards
            elif(output_mode == 1):
                n_cores = self.__n_cores

            theta = np.random.rand(n_cores) * 2 * np.pi
            r = (np.random.rand(n_cores)) ** 0.5 * max_distance
            cores = np.array([r * np.cos(theta), r * np.sin(theta), np.zeros(n_cores)]).T

            zenith, azimuth, magnetic_field_vector = coreas.get_angles(corsika)
            cs = cstrafo.cstrafo(zenith, azimuth, magnetic_field_vector)
            positions_vBvvB = cs.transform_from_magnetic_to_geographic(positions.T)
            positions_vBvvB = cs.transform_to_vxB_vxvxB(positions_vBvvB).T
            dd = (positions_vBvvB[:, 0] ** 2 + positions_vBvvB[:, 1] ** 2) ** 0.5
            ddmax = dd.max()
            logger.info("star shape from: {} - {}".format(-dd.max(), dd.max()))

            cores_vBvvB = cs.transform_from_magnetic_to_geographic(cores.T)
            cores_vBvvB = cs.transform_to_vxB_vxvxB(cores_vBvvB).T
            dcores = (cores_vBvvB[:, 0] ** 2 + cores_vBvvB[:, 1] ** 2) ** 0.5
            mask_cores_in_starpattern = dcores <= ddmax

            if((not np.sum(mask_cores_in_starpattern)) and (output_mode == 1)):  # handle special case of no core position being generated within star pattern
                observer = corsika['CoREAS']['observers'].values()[0]

                evt = NuRadioReco.framework.event.Event(corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])  # create empty event
                station = NuRadioReco.framework.station.Station(self.__station_id)
                sim_station = coreas.make_sim_station(self.__station_id, corsika, observer)

                station.set_sim_station(sim_station)
                evt.set_station(station)
                yield evt, self.__current_input_file, None, area

            cores_to_iterate = cores_vBvvB[mask_cores_in_starpattern]
            if(output_mode == 0):  # select first n_cores that are in star pattern
                if(np.sum(mask_cores_in_starpattern) < self.__n_cores):
                    logger.warning("only {0} cores contained in star pattern, returning {0} cores instead of {1} cores that were requested".format(np.sum(mask_cores_in_starpattern), self.__n_cores))
                else:
                    cores_to_iterate = cores_vBvvB[mask_cores_in_starpattern][:self.__n_cores]

            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t

            for iCore, core in enumerate(cores_to_iterate):
                t = time.time()
                # check if out of bounds

                distances = np.linalg.norm(core[:2] - positions_vBvvB[:, :2], axis=1)
                index = np.argmin(distances)
                distance = distances[index]
                key = list(corsika['CoREAS']['observers'].keys())[index]
                logger.info("generating core at ground ({:.0f}, {:.0f}), vBvvB({:.0f}, {:.0f}), nearest simulated station is {:.0f}m away at ground ({:.0f}, {:.0f}), vBvvB({:.0f}, {:.0f})".format(cores[iCore][0], cores[iCore][1], core[0], core[1], distance / units.m,
                                                                                                                                        positions[index][0], positions[index][1], positions_vBvvB[index][0], positions_vBvvB[index][1]))
#                 import matplotlib.pyplot as plt
#                 indexes = np.array(range(len(cores)))
#                 fig, ax = plt.subplots(1, 1)
#                 ax.plot(positions_vBvvB[:, 0], positions_vBvvB[:, 1], 'o')
#                 ax.plot(positions[:, 0], positions[:, 1], 'd')
#                 ax.plot(core[0], core[1], '*')
#                 ax.plot(cores[indexes[mask_cores_in_starpattern][iCore]][0], cores[indexes[mask_cores_in_starpattern][iCore]][1], 'x')
#                 ax.plot(positions_vBvvB[index][0], positions_vBvvB[index][1], 'd')
#                 ax.plot(positions[index][0], positions[index][1], 'D')
#                 ax.set_aspect("equal")
#                 ax.set_title("zen {:.0f}, az {:.0f}".format(zenith / units.deg, azimuth / units.deg))
#                 plt.show()

                t_event_structure = time.time()
                observer = corsika['CoREAS']['observers'].get(key)

                evt = NuRadioReco.framework.event.Event(self.__current_input_file, iCore)  # create empty event
                station = NuRadioReco.framework.station.Station(self.__station_id)
                channel_ids = detector.get_channel_ids(self.__station_id)
                sim_station = coreas.make_sim_station(self.__station_id, corsika, observer, channel_ids)
                station.set_sim_station(sim_station)
                evt.set_station(station)
                sim_shower = coreas.make_sim_shower(corsika)
                evt.add_sim_shower(sim_shower)
                rd_shower = NuRadioReco.framework.radio_shower.RadioShower([station.get_id()])
                evt.add_shower(rd_shower)
                if(output_mode == 0):
                    self.__t += time.time() - t
                    self.__t_event_structure += time.time() - t_event_structure
                    yield evt
                elif(output_mode == 1):
                    self.__t += time.time() - t
                    self.__t_event_structure += time.time() - t_event_structure
                    yield evt, self.__current_input_file, distance, area
                else:
                    logger.debug("output mode > 1 not implemented")
                    raise NotImplementedError

            self.__current_input_file += 1

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        logger.info("\per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt
