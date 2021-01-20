import NuRadioReco.modules.io.eventReader
from radiotools import helper as hp
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import fft
from NuRadioReco.framework.parameters import stationParameters as stnp
import h5py
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities import propagated_analytic_pulse
import matplotlib
from scipy import signal
from scipy import optimize as opt
from matplotlib import rc
from matplotlib.lines import Line2D
#from lmfit import Minimizer, Parameters, fit_report
import datetime
import math
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation



class rayTypeSelecter:


    def __init__(self):
        self.begin()

    def begin(self):
        """
        begin method. This function is executed before the event loop.

        The antenna pattern provider is initialized here.
        """
        pass

    def run(self, event, shower_id, station, det,
            use_channels=[9, 14]):
        ice = medium.get_ice_model('greenland_simple')
        prop = propagation.get_propagation_module('analytic')
        sim_station = True
        if sim_station: vertex = event.get_sim_shower(shower_id)[shp.vertex]
        debug_plots = True
        #vertex= station[stnp.nu_vertex] 
        x1 = vertex
        sampling_rate = station.get_channel(0).get_sampling_rate() ## assume same for all channels
        if debug_plots: fig, axs = plt.subplots(3)
        ich = 0
        T_ref = np.zeros(3)
        max_totaltrace = np.zeros(3)
        position_max_totaltrace = np.zeros(3)
        for raytype in [ 1, 2,3]:#
            total_trace = np.zeros(len(station.get_channel(0).get_trace()))
            for channel in station.iter_channels():
                if channel.get_id() in use_channels:
                    channel_id = channel.get_id()
                    x2 = det.get_relative_position(station.get_id(), channel_id) + det.get_absolute_position(station.get_id())
                    r = prop(x1, x2, ice, 'GL1')

                    r.find_solutions()
                    for iS in range(r.get_number_of_solutions()):
                       if r.get_solution_type(iS) == raytype:
                           
                           T = r.get_travel_time(iS)
                           print("ref channel", use_channels[0])
                           if channel.get_id() == use_channels[0]: T_ref[iS] = T
                           print("T REF", T_ref)
                           dt = T - T_ref[iS]
                           dn_samples = dt * sampling_rate
                           dn_samples = math.ceil(-1*dn_samples)
                           trace = np.roll(np.copy(channel.get_trace()), dn_samples)
                           total_trace += trace

                           if debug_plots: axs[raytype-1].plot(trace)
        #    if debug_plots: axs[raytype].set_xlim((1500, 2250))
            if debug_plots: axs[raytype-1].set_title("raytype {}".format(['direct', 'refracted', 'reflected'][raytype-1]))
            if debug_plots: axs[raytype-1].plot(total_trace, label = 'total {}'.format(raytype))
            if debug_plots: axs[raytype-1].legend()
            max_totaltrace[raytype-1] = max(abs(total_trace))
     
            position_max_totaltrace[raytype-1] = np.argmax((abs(total_trace)))
        print("max total trace", max_totaltrace)
        reconstructed_raytype = ['direct', 'refracted', 'reflected'][np.argmax(max_totaltrace)]
        if debug_plots: fig.tight_layout()
        if debug_plots: fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/Uncertainties/1_direction_simulations/test.pdf")
        station.set_parameter(stnp.raytype, reconstructed_raytype) 
        print("position max totaltrace", position_max_totaltrace)
        station.set_parameter(stnp.pulse_position, position_max_totaltrace[np.argmax(max_totaltrace)])#pulse location channel 6
        print("reconstructed raytype", reconstructed_raytype)
    def end(self):
        pass