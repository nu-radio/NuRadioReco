from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector.ARIANNA import analog_components
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import numpy as np
from NuRadioReco.utilities import units, fft
import time
import logging


class cosmicRayIceShelfReflector:
    """
    Describe module here


    """

    def __init__(self, debug=False):
        self.logger = logging.getLogger("NuRadioReco.cosmicRayIceShelfReflector")
        self.__debug = debug
        if(self.__debug):
            self.logger.setLevel(logging.DEBUG)

    @register_run()
    def run(self, evt, station, det):
        """
        Describe run method here here

        """
        sim_station = station.get_sim_station()  # get sim station that contains the simulated
        self.logger.debug("print out some debug output")
        for electric_field in sim_station.get_electric_fields():
            zenith = electric_field[efp.zenith]  # the propagation direction of the efield
            azimuth = electric_field[efp.azimuth]  # the propagation direction of the efield
            # modify the propagation direction
            self.logger.debug(f"current zenith angle is {zenith/units.deg:.1f}deg")
#             zenith = 2 * zenith
            # save modified parameters back to efield
            electric_field[efp.zenith] = zenith

            # get frequency spectrum of efield
            spectrum = electric_field.get_frequency_spectrum()
            # modify spectrum, e.g. reduce amplitude by additional propagation through the ice
            electric_field.set_frequency_spectrum(spectrum, sampling_rate=electric_field.get_sampling_rate())

    def end(self):
        pass
