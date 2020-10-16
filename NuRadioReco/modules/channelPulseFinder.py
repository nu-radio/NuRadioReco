from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.framework.parameters import channelParameters as chp
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
import time
import logging
logger = logging.getLogger('channelPulseFinder')


class channelPulseFinder:
    """
    Finds pulses and records windows into channel parameters
    """

    def __init__(self):
        self.__t = 0
        self.logger = logging.getLogger('NuRadioReco.channelPulseFinder')
        self.__debug = None
        self.begin()

    def begin(self, debug=False, log_level=logging.WARNING):
        """
        Parameters
        -----------
        debug: bool
            Set module to debug output
        """
        self.__debug = debug
        self.logger.setLevel(log_level)

    def thresholding(trace, lag, threshold, influence):
        """
        Parameters
        -----------
        trace: array
            event trace
        lag: int
            Lag of window for mean and std calculations
        threshold: float
            Sigma threshold
        influence: float
            influence of pulses on other measurments

        Returns
        ----------
        Pulses: dict
            dictionary of Pulse locations
        """
        signals = np.zeros(len(trace))
        filtered = np.array(trace)
        avgFilter = [0]*len(trace)
        stdFilter = [0]*len(trace)
        avgFilter[lag - 1] = np.mean(y[0:lag])
        stdFilter[lag - 1] = np.std(y[0:lag])
        for i in range(lag, len(trace) - 1):
            if abs(trace[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
                if trace[i] > avgFilter[i-1]:
                    signals[i] = 1
                else:
                    signals[i] = -1

                filtered[i] = influence * y[i] + (1 - influence) * filtered[i-1]
                avgFilter[i] = np.mean(filtered[(i-lag):i])
                stdFilter[i] = np.std(filtered[(i-lag):i])
            else:
                signals[i] = 0
                filtered[i] = trace[i]
                avgFilter[i] = np.mean(filtered[(i-lag):i])
                stdFilter[i] = np.std(filtered[(i-lag):i])

        return dict(signals=np.asarray(signals),
                    avgFilter=np.asarray(avgFilter),
                    stdFilter=np.asarray(stdFilter))

    @register_run()
    def run(self, evt, station, det):
        """
        Run the channelPulseFinder

        Parameters
        ---------

        evt, station, det
            Event, Station, Detector
        """
        t = time.time()
        for channel in station.iter_channels():
            start = channel.get_trace_start_time()
            trace = channel.get_trace()*units.mV
            times = channel.get_times()*units.ns - start
            #trace = np.roll(trace,700)[0:400] #to cut the trace to the middle
            peaks, _ = find_peaks(trace, distance=20, width=2, height=.00005*units.mV)
            envelope = np.abs(hilbert(trace))
            pulses = self.thresholding(envelope, lag=150, threshold=4, influence=0)
            channel[chp.pulses] = pulses


            plt.plot(peaks, trace[peaks], "xr", label = 'peaks')
            plt.plot(trace, label='trace')
            plt.plot(envelope, label='envelope')
            plt.plot(pulses["signals"]*np.max(envelope), label="signals")
            plt.legend()
            #plt.show()

        self.__t = time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
