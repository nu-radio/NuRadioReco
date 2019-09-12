from __future__ import absolute_import, division, print_function

import NuRadioReco.framework.base_station
import numpy as np
from six import iteritems
import pickle
from aenum import Enum


import logging
logger = logging.getLogger('AERAStation')


class AERAStation(NuRadioReco.framework.base_station.BaseStation):

    def __init__(self, station_id, position=None):
        NuRadioReco.framework.base_station.BaseStation.__init__(self, station_id)
        self.__position = position

    def get_position(self):
        return self.__position

    def get_parameter(self, key):
        if not isinstance(key, aerastationParameter):
            raise ValueError("")
        return self._parameters[key]

    def has_parameter(self, key):
        if not isinstance(key, aerastationParameter):
            raise ValueError("")
        return key in self._parameters.keys()

    def set_parameter(self, key, value):
        if not isinstance(key, aerastationParameter):
            raise ValueError("")
        self._parameters[key] = value

    def set_parameter_error(self, key, value):
        if not isinstance(key, aerastationParameter):
            raise ValueError("")
        self._parameter_covariances[(key, key)] = value ** 2

    def get_parameter_error(self, key):
        if not isinstance(key, aerastationParameter):
            raise ValueError("")
        return self._parameter_covariances[(key, key)] ** 0.5

    def serialize(self, mode):
        base_station_pkl = NuRadioReco.framework.base_station.BaseStation.serialize(self, mode)
        data = {'__position': self.__position,
                'base_station': base_station_pkl}
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.base_station.BaseStation.deserialize(self, data['base_station'])
        self.__position = data['__position']


class aerastationParameter(Enum):
    energy_fluence = 1  # reconstructed station signal
    signal_to_noise_ratio = 2
    status = 3  # has_signal, is_rejected, is_saturted, ... (string)
