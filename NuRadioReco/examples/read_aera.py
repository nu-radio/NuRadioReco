import NuRadioReco.modules.io.aera.adstReader
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.io.eventReader
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.electricFieldBandPassFilter
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.utilities import units

from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

from AERAutilities import pyplots

import argparse
import numpy as np

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='ADST root')
parser.add_argument('inputfilename', type=str, nargs='*', help='path')
parser.add_argument('-o', '--outputfilename', type=str, nargs='?', help='path')
parser.add_argument('-l', '--label', type=str, nargs='?', help='label')
parser.add_argument('--nur', action='store_true')

args = parser.parse_args()

# initialize modules
ADSTReader = NuRadioReco.modules.io.aera.adstReader.ADSTReader()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventReader = NuRadioReco.modules.io.eventReader.eventReader()
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()

def write_nur_file():
    ADSTReader.begin(args.inputfilename, read_efield_traces=True)
    eventWriter.begin(args.outputfilename)

    for iE, event in enumerate(ADSTReader.run()):
        sd_shower = event.get_hybrid_information().get_hybrid_shower('SdRecShower')

        sdcore = sd_shower.get_parameter(shp.core)
        distance_core = []

        for station in event.get_stations():
            distance_core.append(np.linalg.norm(station.get_position() - sdcore))

        if not np.all(np.array(distance_core) < 500):
            continue

        eventWriter.run(event)

def analyse_nur_file():
    eventReader.begin(args.inputfilename)

    energy_fluence_23 = []
    energy_fluence_24 = []

    zeniths = []
    azimuths = []
    dts = []
    # for iE, event in enumerate(eventReader.run()):
    for iE, event in enumerate(eventReader.run()):
        sd_shower = event.get_hybrid_information().get_hybrid_shower('SdRecShower')
        zenith = sd_shower.get_parameter(shp.zenith)
        azimuth = sd_shower.get_parameter(shp.azimuth)
        # if np.rad2deg(zenith) > 10:  # or np.rad2deg(zenith) > 55:
        #     continue
        # if np.rad2deg(azimuth) < 260 or np.rad2deg(azimuth) > 280:
        #     continue
        ef23 = event.get_station(123).get_electric_fields()[0]
        ef24 = event.get_station(124).get_electric_fields()[0]
        # dt23 = ef23.get_parameter(efp.signal_time) - ef23.get_parameter(efp.signal_search_window_start)
        # dt24 = ef24.get_parameter(efp.signal_time) - ef24.get_parameter(efp.signal_search_window_start)
        # dts.append(dt23 - dt24)
        event.get_station(124).get_electric_fields()[0].get_parameter(efp.signal_time)
        # print(event.get_station(124).get_electric_fields()[0].get_parameter(efp.signal_time))
        energy_fluence_23.append(event.get_station(123).get_parameter(stp.station_signal))
        energy_fluence_24.append(event.get_station(124).get_parameter(stp.station_signal))
        zeniths.append(sd_shower.get_parameter(shp.zenith))
        azimuths.append(sd_shower.get_parameter(shp.azimuth))
        # dts.append(dt)
    axis_label = r"$\frac{f_{BF} - f_{LPDA}}{f_{LPDA}}$"
    deviation = (np.array(energy_fluence_23) - np.array(energy_fluence_24)) / np.array(energy_fluence_24)
    fig, ax = pyplots.rphp.get_histogram(deviation, bins=np.arange(-2, 3, 0.2), xlabel=axis_label)
    pyplots.save_fig(fig, 'antenna_deviation_%s.png' % args.label)

    # fig, ax = pyplots.rphp.get_histogram(dts, bins=np.arange(-5, 125, 2), xlabel="")
    # pyplots.save_fig(fig, 'time_deviation_%s.png' % args.label)


    pyplots.polar_hist(np.array(azimuths), np.array(zeniths),
                       clim=(0, None), phi_bin=8, #phi_bin=np.linspace(np.pi / 4, 2*np.pi + np.pi / 4, 8 + 1),
                       zenith_bin=np.arange(0, 70, 20), fname='arrival_direction_events_%s.png' % args.label)
    pyplots.polar_hist(np.array(azimuths), np.array(zeniths), var=deviation,
                       clabel=axis_label, norm=True, phi_bin=8, #phi_bin=np.linspace(np.pi / 4, 2*np.pi + np.pi / 4, 8 + 1),
                       zenith_bin=np.arange(0, 70, 20), fname='deviation_arrival_direction_%s.png' % args.label)


def analyse_nur_file_with_io():
    data_provider = NuRadioRecoio.NuRadioRecoio(args.inputfilename, parse_header=True)
    header = data_provider.get_header()

    station_123 = header[123]
    station_124 = header[124]

    mask = np.all([station_123[stp.signal_to_noise_ratio] > 10, station_124[stp.signal_to_noise_ratio] > 10], axis=0)

    axis_label = r"$\frac{f_{BF} - f_{LPDA}}{f_{LPDA}}$"
    deviation = (station_123[stp.station_signal] - station_124[stp.station_signal]) / station_124[stp.station_signal]

    fig, ax = pyplots.rphp.get_histogram(deviation[mask], bins=np.arange(-2, 3, 0.2), xlabel=axis_label)
    pyplots.save_fig(fig, 'antenna_deviation_%s_high.png' % args.label)

    # pyplots.polar_hist(np.array(azimuths), np.array(zeniths),
    #                    clim=(0, None), phi_bin=8, #phi_bin=np.linspace(np.pi / 4, 2*np.pi + np.pi / 4, 8 + 1),
    #                    zenith_bin=np.arange(0, 70, 20), fname='arrival_direction_events_%s.png' % args.label)
    # pyplots.polar_hist(np.array(azimuths), np.array(zeniths), var=deviation,
    #                    clabel=axis_label, norm=True, phi_bin=8, #phi_bin=np.linspace(np.pi / 4, 2*np.pi + np.pi / 4, 8 + 1),
    #                    zenith_bin=np.arange(0, 70, 20), fname='deviation_arrival_direction_%s.png' % args.label)
    #

def print_traces():
    eventReader.begin(args.inputfilename)
    from matplotlib import pyplot as plt
    # for iE, event in enumerate(eventReader.run()):
    for iE, event in enumerate(eventReader.run()):
        station = event.get_station(123)
        station2 = event.get_station(124)
        efield = station.get_electric_fields()[0]
        efield2 = station2.get_electric_fields()[0]
        traces = efield.get_trace()
        traces2 = efield2.get_trace()

        specs = np.abs(efield.get_frequency_spectrum())
        freq = efield.get_frequencies() * 1000  # from GHz to MHz
        print(freq, specs.shape)

        # plt.plot(np.arange(len(traces[0])), traces[0])
        plt.plot(np.arange(len(traces[0])), traces[1])
        # plt.plot(np.arange(len(traces[0])), traces2[0])
        plt.plot(np.arange(len(traces[0])), traces2[1])
        # plt.plot(np.arange(len(traces[0])), traces[2])
        plt.show()

def print_event_ids():
    eventReader.begin(args.inputfilename)
    for iE, event in enumerate(eventReader.run()):
        print(iE, event.get_run_number(), event.get_id())

def calculate_energyfluence():
    eventReader.begin(args.inputfilename)
    noise_window = 350
    electricFieldSignalReconstructor.begin(signal_window_pre=50 * units.ns, signal_window_post=50 * units.ns, noise_window=noise_window * units.ns)

    N = eventReader.get_n_events()
    energy_fluence_offline = np.zeros((N, 2))
    energy_fluence_offline_mag = np.zeros((N, 2))
    energy_fluence_nuradioreco = np.zeros((N, 2))
    energy_fluence_bandpassed = np.zeros((N, 2))
    low, up = 35, 75
    for iE, event in enumerate(eventReader.run()):
        for idx, station in enumerate(event.get_stations()):

            if (idx == 0 and station.get_id() == 124) or (idx == 1 and station.get_id() == 123):
                sys.exit("ERROROR")

            energy_fluence_offline_mag[tuple((iE, idx))] = station.get_parameter(stp.station_signal)

            ef = station.get_electric_fields()[0]
            energy_fluence_offline[tuple((iE, idx))] = np.sum(ef.get_parameter(efp.signal_energy_fluence))

            electricFieldSignalReconstructor.run(event, station, det=None, debug=False)
            ef = station.get_electric_fields()[0]
            energy_fluence_nuradioreco[tuple((iE, idx))] = np.sum(ef.get_parameter(efp.signal_energy_fluence))

            electricFieldBandPassFilter.run(event, station, det=None, passband=[low * units.MHz, up * units.MHz], filter_type='rectangular', order=2, debug=False)
            electricFieldSignalReconstructor.run(event, station, det=None, debug=False)
            ef = station.get_electric_fields()[0]
            energy_fluence_bandpassed[tuple((iE, idx))] = np.sum(ef.get_parameter(efp.signal_energy_fluence))

    label = "meas"

    difflpda = 2 * (energy_fluence_offline[:, 1] -  energy_fluence_nuradioreco[:, 1]) / (energy_fluence_offline[:, 1] +  energy_fluence_nuradioreco[:, 1])
    fig, ax = pyplots.rphp.get_histogram(difflpda, bins=10, xlabel=r"2 $\frac{f_{off} - f_{nurd}}{f_{off} + f_{nurd}}$", stat_kwargs={'additional_text': 'LPDA '})
    pyplots.save_fig(fig, "energy_fluence_compared_offline_nuradioreco_lpda_%.0f_%s.png" % (noise_window, label))

    difflpda2 = 2 * (energy_fluence_offline[:, 1] -  energy_fluence_offline_mag[:, 1]) / (energy_fluence_offline[:, 1] +  energy_fluence_offline_mag[:, 1])
    fig, ax = pyplots.rphp.get_histogram(difflpda2, bins=10, xlabel=r"2 $\frac{f_{off} - f_{x}}{f_{off} + f_{x}}$", stat_kwargs={'additional_text': 'LPDA '})
    pyplots.save_fig(fig, "energy_fluence_compared_offline_lpda_%.0f_%s.png" % (noise_window, label))

    diffbandpass = (energy_fluence_nuradioreco[:, 1] - energy_fluence_bandpassed[:, 1]) / (energy_fluence_nuradioreco[:, 1] )
    fig, ax = pyplots.rphp.get_histogram(diffbandpass, bins=10, xlabel=r"$\frac{f_{30 - 80MHz} - f_{%.0f-%.0fMHz}}{f_{30 - 80MHz}}$" % (low, up), stat_kwargs={'additional_text': 'Bandpassed LPDA '})
    pyplots.save_fig(fig, "energy_fluence_compared_bandpassed_lpda_%.0f_%.0f_%s.png" % (low, up, label))

    diffbf = 2 * (energy_fluence_offline[:, 0] -  energy_fluence_nuradioreco[:, 0]) / (energy_fluence_offline[:, 0] +  energy_fluence_nuradioreco[:, 0])
    fig, ax = pyplots.rphp.get_histogram(diffbf, bins=10, xlabel=r"2 $\frac{f_{off} - f_{nurd}}{f_{off} + f_{nurd}}$", stat_kwargs={'additional_text': 'Butterfly '})
    pyplots.save_fig(fig, "energy_fluence_compared_offline_nuradioreco_bf_%.0f_%s.png" % (noise_window, label))

    diffantennas = (energy_fluence_nuradioreco[:, 0] -  energy_fluence_nuradioreco[:, 1]) / (energy_fluence_nuradioreco[:, 1])
    fig, ax = pyplots.rphp.get_histogram(diffantennas, bins=np.arange(-2, 3, 0.2), xlabel=r" $\frac{f_{BF} - f_{LPDA}}{f_{LPDA}}$")
    pyplots.save_fig(fig, "energy_fluence_compared_antennas_%.0f_%s.png" % (noise_window, label))

    # diffantennas2 = (energy_fluence_offline_mag[:, 0] -  energy_fluence_offline_mag[:, 1]) / energy_fluence_offline_mag[:, 1]
    # fig, ax = pyplots.rphp.get_histogram(diffantennas2, bins=np.arange(-2, 3, 0.2), xlabel=r" $\frac{f_{BF} - f_{LPDA}}{f_{LPDA}}$")
    # pyplots.save_fig(fig, "energy_fluence_compared_antennas_offline_%.0f_%s.png" % (noise_window, label))


    diffbandpassantennas = (energy_fluence_bandpassed[:, 0] -  energy_fluence_bandpassed[:, 1]) / (energy_fluence_bandpassed[:, 1])
    fig, ax = pyplots.rphp.get_histogram(diffbandpassantennas, bins=np.arange(-2, 3, 0.2), xlabel=r" $\frac{f_{BF} - f_{LPDA}}{f_{LPDA}}$", stat_kwargs={'additional_text': 'Bandpassed %.0f-%.0f MHz '% (low, up)})
    pyplots.save_fig(fig, "energy_fluence_compared_bandpassed_antennas_%.0f_%.0f_%s.png" % (low, up, label))

    # diffdiff = difflpda - diffbf
    # fig, ax = pyplots.rphp.get_histogram(diffdiff, bins=10)
    # pyplots.save_fig(fig, "energy_fluence_compared_diff_%.0f.png" % (noise_window, label))


if args.nur:
    write_nur_file()
else:
    calculate_energyfluence()
    # analyse_nur_file_with_io()
    # analyse_nur_file()
