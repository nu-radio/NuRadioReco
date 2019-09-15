import NuRadioReco.framework.event
import NuRadioReco.framework.hybrid_shower
import NuRadioReco.framework.radio_shower
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.base_station import reconstructionStatus as recs
from NuRadioReco.utilities import units

import numpy as np
from scipy import interpolate

import sys
import os
import time
import imp

import logging
logger = logging.getLogger("ADSTReader")

try:
    AugerOfflineRoot = os.environ["AUGEROFFLINEROOT"]
except KeyError as e:
    sys.exit('%s not in enviourment, please source Offline.' % e)

import ROOT
ROOT.gSystem.Load(os.path.join(AugerOfflineRoot, "lib/libRecEventKG.so"))

NuRadioRecoRoot = os.environ["NuRadioReco"]
path_to_enum = NuRadioRecoRoot + "NuRadioReco/modules/io/aera/convertenums.py"

if not os.path.exists(path_to_enum):
    sys.exit("Path to enum file not correct, cant handle enums! Aborting ...")

ce = imp.load_source('', path_to_enum)
rdstQ = ce.rdstQ
rdshQ = ce.rdshQ

class ADSTReader:

    """
    This is the AERAReader. Reads reconstructed AERA data from the ADST root format.


    """
    def begin(self, input_files, read_efield_traces=False, read_channel_traces=False):

        """
        Parameters
        ----------
        input_files: list
        paths

        read_efield_traces: bool
        if True, the reconstructed electric field traces (RdRecStation traces) are stored in an electric field (default: False)

        read_channel_traces: bool
        if True, the channel are stored (default: False)

        """

        self.__t = time.time()

        self.__read_efield_traces = read_efield_traces
        self.__read_channel_traces = read_channel_traces

        root_files = ROOT.std.vector('string')()
        for file in input_files:
            root_files.push_back(file)

        print('Read %d root file(s) ...' % len(input_files))
        self.__rec_files = ROOT.RecEventFile(root_files)
        self.__aera_rec_event = ROOT.RecEvent()
        self.__rec_files.SetBuffers(self.__aera_rec_event)

        self.__detector_geometry = ROOT.DetectorGeometry()
        self.__rec_files.ReadDetectorGeometry(self.__detector_geometry)

        self.__n_events = self.__rec_files.GetNEvents()
        print("Read in %d event(s)." % self.__n_events)


    def run(self):
        """
        Run function of the AERA rec. event reader
        """

        while self.__rec_files.ReadNextEvent() == ROOT.RecEventFile.eSuccess:

            info = ROOT.EventInfo()
            if not self.__rec_files.GetEventInfo(self.__rec_files.GetCurrentEvent(), info):
                sys.exit('Faild to read in EventInfo')

            evt = NuRadioReco.framework.event.Event(info.GetRdRunNumber(), info.GetRdEventId())

            if info.GetSDRecLevel() >= 3:  # has ldf fit
                sdshower = self._read_sdshower(self.__aera_rec_event)
                evt.get_hybrid_information().add_hybrid_shower(sdshower)

            if info.GetFDRecLevel(1) >= 10:  # has mono energy
                fdshower = self._read_fdshower(self.__aera_rec_event)
                evt.get_hybrid_information().add_hybrid_shower(fdshower)

            if self.__rec_files.IsMC():
                simshower = self._read_gshower(self.__aera_rec_event)
                evt.get_hybrid_information().add_hybrid_shower(simshower)  # maybe store that as sim_shower?

            # rshower = self._read_rshower(self.__aera_rec_event)
            # evt.add_shower(rshower)
            for station in self._read_radio_stations(self.__aera_rec_event):
                evt.set_station(station)

            yield evt


    def end(self):
        pass


    def store_to_nur(self, outputfilename):
        import NuRadioReco.modules.io.eventWriter
        eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()

        eventWriter.begin(outputfilename)

        for event in self.run():
            eventWriter.run(event)

        eventWriter.end()


    def _read_sdshower(self, recEvent):
        sdRecShower = recEvent.GetSDEvent().GetSdRecShower()

        # sdshower.set_parameter("sd_event_id", event.GetSDEvent().GetEventId())
        # sdshower.set_parameter("gps_sec", event.GetSDEvent().GetGPSSecond())

        sdshower = NuRadioReco.framework.hybrid_shower.HybridShower('SdRecShower')

        sdshower.set_parameter(shp.zenith, sdRecShower.GetZenith())
        sdshower.set_parameter(shp.azimuth, sdRecShower.GetAzimuth())
        sdshower.set_parameter_error(shp.zenith, sdRecShower.GetZenithError())
        sdshower.set_parameter_error(shp.azimuth, sdRecShower.GetAzimuthError())

        sdshower.set_parameter(shp.core, np.array(sdRecShower.GetCoreSiteCS()))
        sdshower.set_parameter_error(shp.core, np.array([sdRecShower.GetCoreEastingError(), sdRecShower.GetCoreNorthingError(), sdRecShower.GetCoreNorthingEastingCorrelation()]))

        sdshower.set_parameter(shp.shower_size, sdRecShower.GetShowerSize())
        sdshower.set_parameter_error(shp.shower_size, sdRecShower.GetShowerSizeError())
        # sdshower.set_parameter(shp.shower_size_label, sdRecShower.GetShowerSizeLabel())
        # sdshower.set_parameter(shp.ldf_chi, sdRecShower.GetLDFChi2())
        # sdshower.set_parameter(shp.ldf_ndf, sdRecShower.GetLDFNdof())

        sdshower.set_parameter(shp.energy, sdRecShower.GetEnergy())
        sdshower.set_parameter_error(shp.energy, sdRecShower.GetEnergyError())
        # sdshower.set_parameter(shp.declination, sdRecShower.GetDeclination())

        return sdshower


    def _read_fdshower(self, recEvent):
        # needs implementation

        fdshower = NuRadioReco.framework.hybrid_shower.HybridShower('FdRecShower')

        return fdshower


    def _read_gshower(self, recEvent):
        genShower = recEvent.GetGenShower()

        simshower = NuRadioReco.framework.hybrid_shower.HybridShower('SimShower')

        simshower.set_parameter(shp.zenith, genShower.GetZenith())
        simshower.set_parameter(shp.azimuth, genShower.GetAzimuth())
        simshower.set_parameter(shp.shower_maximum, genShower.GetXmaxGaisserHillas())

        simshower.set_parameter(shp.primary_particle, genShower.GetPrimary())
        simshower.set_parameter(shp.distance_shower_maximum_geometric, genShower.GetDistanceOfShowerMaximum() * units.cm)  # conversion into meter
        simshower.set_parameter(shp.core, np.array(genShower.GetCoreSiteCS()))
        # simshower.set_parameter(shp.magenetic_field_declination, genShower.GetMagneticFieldDeclination())

        simshower.set_parameter(shp.electromagnetic_energy, genShower.GetElecEnergy())
        simshower.set_parameter(shp.energy, genShower.GetEnergy())

        return simshower


    def _read_rshower(self, recEvent):
        recRShower = recEvent.GetRdEvent().GetRdRecShower()

        rshower = NuRadioReco.framework.radio_shower.RadioShower()

        # from reference system (PA)
        rshower.set_parameter(shp.core, np.array([recRShower.GetParameter(rdshQ.eCoreX),
                                                  recRShower.GetParameter(rdshQ.eCoreY),
                                                  recRShower.GetParameter(rdshQ.eCoreZ)]))

        rshower.set_parameter(shp.zenith, recRShower.GetZenith())
        rshower.set_parameter(shp.azimuth, recRShower.GetAzimuth())

        rshower.set_parameter(shp.magnetic_field_vector, np.array(recRShower.GetMagneticFieldVector()))

        # rshower.set_parameter("chi2", recRShower.GetParameter(rdshQ.eHASLDFChi2))
        # rshower.set_parameter("ndf", recRShower.GetParameter(rdshQ.eHASLDFNDF))
        #
        # rshower.set_parameter("A", recRShower.GetParameter(rdshQ.eHASLDFA))
        # rshower.set_parameter("B", recRShower.GetParameter(rdshQ.eHASLDFB))
        # rshower.set_parameter("C", recRShower.GetParameter(rdshQ.eHASLDFC))
        # rshower.set_parameter("D", recRShower.GetParameter(rdshQ.eHASLDFD))
        # rshower.set_parameter("r0", recRShower.GetParameter(rdshQ.eHASLDFr0))
        #
        # rshower.set_parameter("density_at_xmax", recRShower.GetParameter(rdshQ.eHASLDFDensityAtXmax))  # in kg/m3
        # rshower.set_parameter("distance_xmax_geometric", recRShower.GetParameter(rdshQ.eHASLDFDistanceToXmaxGeometric))  # in meter
        #
        # rshower.set_parameter("core_fit_vB", np.array([recRShower.GetParameter(rdshQ.eHASLDFCoreX),
        #                                                recRShower.GetParameter(rdshQ.eHASLDFCoreY),
        #                                                0]))

        # rshower.set_parameter("E_geo", recRShower.GetParameter(rdshQ.eGeomagneticRadiationEnergy))
        # rshower.set_parameter("density_correction", recRShower.GetParameter(rdshQ.eDensityCorrectionFactor))
        # rshower.set_parameter("S_geo_rad", recRShower.GetParameter(rdshQ.eRadioEnergyEstimator))
        # rshower.set_parameter("electromagnetic_energy", recRShower.GetParameter(rdshQ.eReconstructedElectromagneticEnergy))

        return rshower


    def _read_radio_stations(self, recEvent):
        rDetector = recEvent.GetDetector()
        rStations = recEvent.GetRdEvent().GetRdStationVector()

        for rec_station in np.array(rStations):
            station_position = np.array(rDetector.GetRdStationPosition(rec_station.GetId()))

            station = NuRadioReco.framework.station.Station(rec_station.GetId(), position=station_position)

            if rec_station.IsRejected():
                station.set_reconstruction_status(recs.is_rejected)

            if rec_station.IsSaturated():
                station.set_reconstruction_status(recs.is_saturated)

            if rec_station.HasPulse():
                station.set_reconstruction_status(recs.has_signal)

            electric_field = NuRadioReco.framework.electric_field.ElectricField([0, 1, 2], position=station_position)
            electric_field.set_parameter(efp.signal_time, rec_station.GetParameter(rdstQ.eSignalTime))
            electric_field.set_parameter(efp.signal_energy_fluence, np.array([rdstQ.eSignalEnergyFluenceNS,
                                                                              rdstQ.eSignalEnergyFluenceEW,
                                                                              rdstQ.eSignalEnergyFluenceV]))

            electric_field.set_parameter(efp.noise_energy_fluence, np.array([rdstQ.eNoiseEnergyFluenceNS,
                                                                              rdstQ.eNoiseEnergyFluenceEW,
                                                                              rdstQ.eNoiseEnergyFluenceV]))

            if self.__read_efield_traces:
                # yep this assumes the the sampling rates are the same for all polarisations, lets pray to god that this is always true ...
                electric_field.set_trace(np.array([np.array(rec_station.GetRdTrace(0).GetTimeTrace()),
                                                   np.array(rec_station.GetRdTrace(1).GetTimeTrace()),
                                                   np.array(rec_station.GetRdTrace(2).GetTimeTrace())]),
                                                   sampling_rate=rec_station.GetRdTrace(0).GetSamplingRate())
                electric_field.set_trace_start_time(rec_station.GetParameter(rdstQ.eTraceStartTime))

            station.set_electric_fields([electric_field])

            '''
            TODO:
            Signal / Noise SearchTime
            polarisation angles ...
            noise energy fluence
            '''

            if self.__read_channel_traces:
                raise NotImplementedError

            copy_parameters = [(stp.station_signal, rdstQ.eSignalEnergyFluenceMag),
                               (stp.signal_to_noise_ratio, rdstQ.eSignalToNoise)]

            for para in copy_parameters:
                try:
                    station.set_parameter(para[0], rec_station.GetParameter(para[1]))
                except KeyError as e:
                    print("Key could not be copied: %s" % e)

            yield station

        # stations = np.array([x for x in rStations if x.HasPulse() and not x.IsRejected()])  #  and not x.IsSaturated()
        #
        # rshower.set_parameter("station_ids", np.array([x.GetId() for x in stations]))
        #
        # rshower.set_parameter("f_geo", get_stations_parameter(rdstQ.eGeomagneticEnergyFluence, stations))
        # rshower.set_parameter("f_vxB_fit", get_stations_parameter(rdstQ.ePredictedEnergyFluenceVxB, stations))
        #
        # rshower.set_parameter("f_vxB", get_stations_parameter(rdstQ.eSignalEnergyFluenceVxB, stations))
        # rshower.set_parameter_error("f_vxB", get_stations_parameter_error(rdstQ.eSignalEnergyFluenceVxB, stations))
        #
        # rshower.set_parameter("c_early_late", get_stations_parameter(rdstQ.eEarlyLateCorrectionFactor, stations))
        # rshower.set_parameter("charge_excess_fraction", get_stations_parameter(rdstQ.eChargeExcessFraction, stations))
        #
        # station_position_vBvvB = np.empty((len(stations), 3))
        # station_position = np.empty((len(stations), 3))
        # for idx, station in enumerate(stations):
        #     x_vxB = station.GetParameter(rdstQ.eLDFFitStationPositionVxB)
        #     y_vxB = station.GetParameter(rdstQ.eLDFFitStationPositionVxVxB)
        #     z_vxB = station.GetParameter(rdstQ.eLDFFitStationPositionV)
        #     station_position_vBvvB[idx] = np.array([x_vxB, y_vxB, z_vxB])
        #
        #     station_position[idx] = np.array(rDetector.GetRdStationPosition(station.GetId()))
        #
        # rshower.set_parameter("station_position_vBvvB", station_position_vBvvB)
        #
        # # checked position with offline
        # rshower.set_parameter("station_position", station_position)

        # # TMP:
        # gSh = recEvent.GetGenShower()
        # zenith = gSh.GetZenith()
        # azimuth = gSh.GetAzimuth()
        # core = np.array(gSh.GetCoreSiteCS())
        # station_position_vBvvB = get_station_vb_vvb_position(station_position, core, zenith, azimuth)
        # rshower.set_parameter("station_position_vBvvB", station_position_vBvvB)
