from NuRadioReco.utilities import units
from NuRadioReco.framework.trigger import SimplePhasedTrigger
import numpy as np
import time
import logging
logger = logging.getLogger('phasedTriggerSimulator')


class triggerSimulator:
    """
    Calculates the trigger for a phased array

    explain the module in more detail here
    """

    def __init__(self):
        self.__t = 0
        self.begin()


    def begin(self, debug=False, pre_trigger_time=100 * units.ns):
        self.__pre_trigger_time = pre_trigger_time
	self.__debug = debug

    def run(self, evt, station, det,
            threshold=60 * units.mV,
            triggered_channels=None,
            trigger_name='simple_phased_threshold'):
        """
        simulates phased array trigger for each event

        describe the run method in more detail here (if neccessary).
        I left triggered_channels in as an optional parameter, don't know if you
        need it.

        Parameters
        ----------
        triggered_channels: array of ints
            channels ids that are triggered on
        """

        sampling_rate = station.get_channel(0).get_sampling_rate()

        phased_trace = None

	if (triggered_channels == None):

		triggered_channels = [channel._id for channel in station.iter_channels()]

        for channel in station.iter_channels():  # loop over all channels (i.e. antennas) of the station
            channel_id = channel.get_id()
            if channel_id not in triggered_channels:  # skip all channels that do not participate in the trigger decision
		logger.debug("skipping channel{}".format(channel_id))
                continue

            trace = channel.get_trace()  # get the time trace (i.e. an array of amplitudes)
            times = channel.get_times()  # get the corresponding time bins

            antenna_position = det.get_relative_position(station.get_id(), channel.get_id())  # ask the detector to get the antenna position, the function returns an array of [x, y,z] coordinates

            # phase up the signal, assuming no time delays between the channels
            if(phased_trace is None):
                phased_trace = trace
            else:
                phased_trace += trace

	is_triggered = False
        if(np.max(np.abs(phased_trace)) > threshold):  # define a simple threshold trigger
            is_triggered = True
	    logger.debug("Station has triggered")

	trigger = SimplePhasedTrigger(trigger_name, threshold, triggered_channels)
 	trigger.set_triggered(is_triggered)
	station.set_trigger(trigger)

    def end(self):
        pass
