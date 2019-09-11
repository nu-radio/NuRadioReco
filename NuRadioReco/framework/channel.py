from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.parameters as parameters
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('channel')


class Channel(NuRadioReco.framework.base_trace.BaseTrace):

    """
    framework of channel object
    """

    def __init__(self, channel_id):
        """
        Init function of channel object

        Parameters
        ----------
        channel_id: int
            ID of channel
        """
        NuRadioReco.framework.base_trace.BaseTrace.__init__(self)
        self._parameters = {}
        self._id = channel_id
        self.__electric_field = None

    def get_parameter(self, key):
        """
        Access channel parameter

        Parameters
        ----------
        key: NuRadioReco.framework.parameters.channelParameters
            channel parameter, for a new parameter add it to the framework first

        return: value

        """
        if not isinstance(key, parameters.channelParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
        return self._parameters[key]

    def get_parameters(self):
        """
        Get all channel parameters, parameters.channelParameters

        """
        return self._parameters

    def set_parameter(self, key, value):
        """
        Set channel parameter

        Parameters
        ----------
        key: NuRadioReco.framework.parameters.channelParameters
            channel parameter, for a new parameter add it to the framework first
        value: var
            as defined in channel parameters

        """
        if not isinstance(key, parameters.channelParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
        self._parameters[key] = value

    def has_parameter(self, key):
        """
        Check for station parameter

        Parameters
        ----------
        key: NuRadioReco.framework.parameters.channelParameters
            channel parameter, for a new parameter add it to the framework first

        """
        if not isinstance(key, parameters.channelParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
        return key in self._parameters

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def get_id(self):
        """
        Get channel id

        return: int
        """
        return self._id

    def serialize(self, mode):
        """
        Serialize the channel data so it can be saved

        Parameters
        ----------
        key: str
            'mirco', 'mini' or 'full'

        """
        if(mode == 'micro' or mode == 'mini'):
            base_trace_pkl = None
        else:
            base_trace_pkl = NuRadioReco.framework.base_trace.BaseTrace.serialize(self)
        data = {'parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
                'id': self.get_id(),
                'base_trace': base_trace_pkl}
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        """
        Deserialize the channel data from file

        Parameters
        ----------
        data_pkl: data

        """
        data = pickle.loads(data_pkl)
        if(data['base_trace'] is not None):
            NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['base_trace'])
        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(data['parameters'], parameters.channelParameters)
        self._id = data['id']
        if 'electric_field' in data.keys():
            self.__electric_field = NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['electric_field'])
