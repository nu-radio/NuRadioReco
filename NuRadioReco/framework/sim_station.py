from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_station
from six import iteritems
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('SimStation')


class SimStation(NuRadioReco.framework.base_station.BaseStation):
    """
    Class to store sim station object
    """
    def __init__(self, station_id, sampling_rate=None, trace=None, position=None):
        """
        Init function of sim station class

        Parameters
        ----------

        station_id: int
            station ID
        sampling_rate: float (default: None)
            sampling rate of station
        trace: array (default:None)
            waveform as function of time
        position: array (default:None)
            position of simulated station

        """
        NuRadioReco.framework.base_station.BaseStation.__init__(self, station_id)
        self.__position = position
        self.__magnetic_field_vector = None
        self.__simulation_weight = None
        if(trace is not None and sampling_rate is not None):
            self.set_electric_fields(trace, sampling_rate)

    def get_position(self):
        """
        Get station position

        Return: array


        """
        return self.__position

    def get_magnetic_field_vector(self):
        """
        Get magnetic field vector in which the simulation was run
        Relevant for cosmic rays

        Return: array


        """
        return self.__magnetic_field_vector

    def set_magnetic_field_vector(self, magnetic_field_vector):
        """
        Set magnetic field vector in which the simulation was run
        Relevant for cosmic rays

        Parameters
        ----------
        magnetic_field_vector: array

        """
        self.__magnetic_field_vector = magnetic_field_vector

    def get_simulation_weight(self):
        """
        Get weight of station from simulation (how often would
        such a station trigger)

        return: float

        """
        return self.__simulation_weight

    def set_simulation_weight(self, simulation_weight):
        """
        Set weight of station from simulation (how often would
        such a station trigger)

        Parameters
        ----------
        simulation_weight: float


        """
        self.__simulation_weight = simulation_weight

    def serialize(self, mode):
        """
        Serialize the station data so it can be saved

        Parameters
        ----------
        key: str
            'mirco', 'mini' or 'full'

        """
        base_station_pkl = NuRadioReco.framework.base_station.BaseStation.serialize(self, mode)
        data = {'__magnetic_field_vector': self.__magnetic_field_vector,
                '__simulation_weight': self.__simulation_weight,
                '__position': self.__position,
                'base_station': base_station_pkl}
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        """
        Deserialize the station data from file

        Parameters
        ----------
        data_pkl: data

        """
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.base_station.BaseStation.deserialize(self, data['base_station'])
        self.__magnetic_field_vector = data['__magnetic_field_vector']
        self.__simulation_weight = data['__simulation_weight']
        self.__position = data['__position']
