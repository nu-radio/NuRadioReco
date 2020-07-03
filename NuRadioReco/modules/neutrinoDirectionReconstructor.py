import NuRadioReco.modules.io.eventReader
from radiotools import helper as hp
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import fft
from NuRadioReco.framework.parameters import stationParameters as stnp
import h5py
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities import propagated_analytic_pulse
import matplotlib
from scipy import signal
from scipy import optimize as opt
from matplotlib import rc
from matplotlib.lines import Line2D


###### still to do:
###### - sampling rate, test for 5 GHZ
###### - precision is not good enough ( event 449 19 3)
###### - for overlapping pulses we should only include 1 ray
##### - noise has same contribution, so only include for first iteration channels > 3

simulation = propagated_analytic_pulse.simulation()


class neutrinoDirectionReconstructor:
    
    
    def __init__(self):
        self.begin()

    def begin(self):
        """
        begin method. This function is executed before the event loop.

        The antenna pattern provider is initialized here.
        """
        pass
    
    def run(self, event, station, det, debug=False, debug_plotpath=None,
            use_channels=[9, 13]):
        
        
        sampling_rate = 5

        combined_fit = False #direction and energy are fitted simultaneously
        seperate_fit = True ##direction is first fitted and then values are used to fit energy
        if seperate_fit:
            fitprocedure = 'seperate'
        if combined_fit: ## Often a local minima is found
            fitprocedure = 'combined'

        if combined_fit == seperate_fit:
            print(" WARNING: decision needs to be made what kind of fit is performed" )

        
    
        def minimizer(params, vertex_x, vertex_y, vertex_z, minimize = True, fit = 'seperate', timing_k = False, first_iter = False):
            sigma = 1.7*10**(-5)
            if len(params) == 2: ##hadronic shower direction fit
                if fit == 'seperate':
                    zenith, azimuth = params
                    energy = 10**19
                if fit == 'combined':
                    zenith, azimuth = params
                    energy = rec_energy
            if len(params) == 1: ##hadronic shower energy fit with input fitted direction
                energy = params[0]
                zenith, azimuth = rec_zenith, rec_azimuth
            if fit == 'seperate':
                if len(params) == 3: ## hadronic or electromagnetic total fit
                    zenith, azimuth, energy = params
            if fit == 'combined':
                if len(params) == 3: ## hadronic or electromagnetic total fit
                    zenith, azimuth, energy = params

            ###### Get channel with maximum amplitude to find reference timing
            maximum_trace1 = 0
            for ich, channel in enumerate(station.iter_channels()):
                if channel.get_id() in use_channels:
                    maximum_trace = max(np.copy(channel.get_trace()))
                    if maximum_trace > maximum_trace1:
                        maxchannelid = channel.get_id()
                        maximum_trace1 = maximum_trace
        
        
            traces_sim, timing_sim = simulation.simulation(station, vertex_x, vertex_y, vertex_z, zenith, azimuth, energy, use_channels, fit, first_iter = first_iter)
            traces, timing = simulation.simulation(station, vertex_x, vertex_y, vertex_z, zenith, azimuth, energy, use_channels, fit, first_iter = first_iter)

            
            
            
            
            
            chi2 = 0

            rec_traces = []
            normalization_factors = {} ## dictionary to store the normalzation factors


           
            data_trace = np.copy(station.get_channel(maxchannelid).get_trace())
            max_trace = 0
            rec_trace = np.zeros(len(data_trace))## if for the simulated trace there is no solution
            for key in traces_sim[maxchannelid]:
                rec_trace_i = traces_sim[maxchannelid][key]

                if max(abs(rec_trace_i)) > max_trace: ### maximum of the reconstrucetd is chosen, meaning that T ref can be different for different directions. When we include noise, this should not matter. 
                    rec_trace = rec_trace_i
                    max_trace = max(abs(rec_trace_i))
                    T_ref_sim = timing_sim[maxchannelid][key]## for maximum ray type, take timing as reference timing
                    key_used = key

                    
            T_ref = T_ref_sim     
            
            for key in traces[maxchannelid]:
                if key == key_used:
                    rec_trace = traces[maxchannelid][key]
            
            
            
            k_ref = np.argmax(abs(data_trace))
            trace_range = 100 * sampling_rate
            data_trace[ abs(np.arange(0, len(data_trace)) - k_ref) > trace_range] = 0
            rec_trace2 = np.pad(rec_trace, (0, len(data_trace) - len(rec_trace)), 'constant') ## add zeros to simulated trace to mmake same length as data
            corr = signal.hilbert(signal.correlate(data_trace, rec_trace2)) # correlate the two to find offset for simulated trace
            toffset = np.argmax(corr) - (len(corr)/2) +1

            ks = {}

            for ich, channel in enumerate(station.iter_channels()):
                if channel.get_id() in use_channels: #iterate over channels
                    data_trace = np.copy(channel.get_trace())
                    max_trace = 0
                    ### if no solution exist, than analytic voltage is zero
                    rec_trace = np.zeros(len(data_trace))# if there is no raytracing solution, the trace is only zeros

                    rec_trace3 = np.zeros(len(data_trace))# if there is no raytracing solution, the trace is only zeros
                    delta_k = [] ## if no solution type exist then channel is not included
                    num = 0
                    for key in traces[channel.get_id()]: ## iterate over ray type solutions
                        rec_trace_i = traces[channel.get_id()][key]

                        rec_trace = rec_trace_i
                        max_trace = max(abs(rec_trace_i))
                        delta_T =  timing[channel.get_id()][key] - T_ref
                        ## before correlating, set values around maximum voltage trace data to zero
                        delta_toffset = delta_T * sampling_rate

                        rec_trace2 = np.pad(rec_trace, (0, len(data_trace) - len(rec_trace)), 'constant') ## add zeros to simulated trace to mmke same length as data

                        ### figuring out the time offset for specfic trace
                        dk = int(k_ref + delta_toffset)
                        

                        ## for ARZ, we need to correlate and cannot just use delta_toffset
                        rec_trace1 = np.roll(rec_trace2, int(toffset+delta_toffset)) # roll simulated trace

                        ### now correlate rectrace1 with the cut data, and look for the offset
                        ## cut data:
                        data_trace_timing = np.copy(data_trace) ## cut data around timing
                        data_trace_timing[ abs(np.arange(0, len(data_trace_timing)) - dk) > 20] = 0
                        rec_trace_timing = np.copy(rec_trace1) ## cut rec around timing
                        rec_trace_timing[ abs(np.arange(0, len(data_trace_timing)) - dk) > 20] = 0
                        normdata = max(abs(data_trace_timing))
                        normrec = max(abs(rec_trace_timing))
                        corr = signal.hilbert(signal.correlate(data_trace_timing, rec_trace_timing*normdata/normrec)) ## correlate
                        dt = np.argmax(corr) - (len(corr)/2) +1 ## find offset
                        ## rotate for ARZ correction
                        rec_trace1 = np.roll(rec_trace1, int(dt)) # roll reconstruction trace with ARZ extra offset

                        delta_k.append(int(k_ref + delta_toffset + dt )) ## for overlapping pulses this does not work
                        rec_trace3 += rec_trace1 ## add two voltage traces in time domain
                    ks[channel.get_id()] = delta_k

                    N= 50
                    if fit == 'seperate':
                        if 1:#abs(np.diff([delta_k[0], delta_k[1]])[0]) > 50):    ## for now exclude overlapping pulses, because i don't know how to treat them
                            if len(params) == 2: # if we fit direction, we scale the pulse
                                normalization_factor = 1
                            
                                if len(delta_k) > 0:
                                    SNR = (abs(abs(min(data_trace[delta_k[0]-N: delta_k[0]+3*N])) + max(data_trace[delta_k[0]-N: delta_k[0]+3*N]))) / (2*Vrms)
                                    #plt.plot(data_trace[delta_k[0]-N: delta_k[0]+3*N])
                                    #plt.show()
                                    if SNR > 4:## for first iteration only use channels with high SNR
                                        if (max(rec_trace3[delta_k[0]-N: delta_k[0]+3*N]) != 0): ## determine normalization factor from maximum pulse
                                            norm = max(abs(data_trace[delta_k[0]-N: delta_k[0]+3*N]))
                                            normalization_factor = norm/ max(abs(rec_trace3[delta_k[0]-N: delta_k[0]+3*N]))

                                            chi2 += np.sum(-1*abs((rec_trace3*normalization_factor)[delta_k[0]-N: delta_k[0]+3*N] - data_trace[delta_k[0]-N:delta_k[0]+3*N])**2/(2*sigma**2)) ### first ray tracing solution
                                            #print("chi2", np.sum(-1*abs((rec_trace3*normalization_factor)[delta_k[0]-N: delta_k[0]+3*N] - data_trace[delta_k[0]-N:delta_k[0]+3*N])**2/(2*sigma**2)))
                                           
                                          
                                           
                                    rec_traces.append(rec_trace3*normalization_factor)
                                
                                if len(delta_k) > 1:
                                    SNR = (abs(abs(min(data_trace[delta_k[1]-N: delta_k[1]+3*N])) + max(data_trace[delta_k[1]-N: delta_k[1]+3*N]))) / (2*Vrms)
                                    if SNR > 4:
                                        if (max(rec_trace3[delta_k[1]-N: delta_k[1]+3*N]) != 0): ## determine normalization factor from maximum pulse
                                            norm = max(abs(data_trace[delta_k[1]-N: delta_k[1]+3*N]))
                                            normalization_factor = norm/ max(abs(rec_trace3[delta_k[1]-N: delta_k[1]+3*N]))
                                            chi2 += np.sum(-1*abs((rec_trace3*normalization_factor)[delta_k[1]-N: delta_k[1]+3*N] - data_trace[delta_k[1]-N:delta_k[1]+3*N])**2/(2*sigma**2)) ### second ray tracing solution

                                    rec_traces[-1][delta_k[1]-N: delta_k[1]+3*N] = (rec_trace3*normalization_factor)[delta_k[1]-N: delta_k[1]+3*N]
                                if len(delta_k) == 0: ## no solutions
                                    rec_traces.append(rec_trace3)
                            if len(params) != 2: # we fit energy for seperate fit, so no scaling factor
                                rec_traces.append(rec_trace3)
                                if len(delta_k) > 0:
                                    chi2 += np.sum(-1*abs((rec_trace3)[delta_k[0]-N: delta_k[0]+3*N] - data_trace[delta_k[0]-N:delta_k[0]+3*N])**2/(2*sigma**2))##
                                    #print("chi2", np.sum(-1*abs((rec_trace3)[delta_k[0]-N: delta_k[0]+3*N] - data_trace[delta_k[0]-N:delta_k[0]+3*N])**2/(2*sigma**2)))

                                if len(delta_k) > 1:
                                    chi2 += np.sum(-1*abs((rec_trace3)[delta_k[1]-N: delta_k[1]+3*N] - data_trace[delta_k[1]-N:delta_k[1]+3*N])**2/(2*sigma**2)) ## second ray type solution


                    if fit == 'combined':
                        rec_traces.append(rec_trace3)
                        if len(delta_k) > 0:
                            chi2 += np.sum(-1*abs((rec_trace3)[delta_k[0]-N: delta_k[0]+3*N] - data_trace[delta_k[0]-N:delta_k[0]+3*N])**2/(2*sigma**2))##
                        if len(delta_k) > 1:
                            chi2 += np.sum(-1*abs((rec_trace3)[delta_k[1]-N: delta_k[1]+3*N] - data_trace[delta_k[1]-N:delta_k[1]+3*N])**2/(2*sigma**2)) ## second ray type solution

            if timing_k:
                return ks
            if not minimize:
                return rec_traces

            return -1*chi2
    
        station.set_is_neutrino()
        if station.has_sim_station():
            simulated_zenith = station.get_sim_station()[stnp.nu_zenith]
            simulated_azimuth = station.get_sim_station()[stnp.nu_azimuth]
            simulated_energy = station.get_sim_station()[stnp.nu_energy]
            simulated_vertex = station.get_sim_station()[stnp.nu_vertex]
            print("simulated vertex position is", simulated_vertex)
            for ich, channel in enumerate(station.iter_channels()): ## checks SNR of channels
                Vrms = 1.6257*10**(-5)
                print("channel {}, SNR {}".format(channel.get_id(),(abs(min(channel.get_trace())) + max(channel.get_trace())) / (2*Vrms) ))
                if channel.get_id() == 13:
                    Vrms = 1.6257*10**(-5) # 16 micro volt
                    SNR = (abs(abs(min(channel.get_trace()))) + max(channel.get_trace())) / (2*Vrms)
                    print("SNR", SNR)

        if SNR > 4:
            uncertainties = 0 ## check influence of vertex position
            if uncertainties:
                for ie, efield in enumerate(station.get_sim_station().get_electric_fields()):
                    if efield.get_channel_ids()[0] == 1:
                        simulated_vertex = station.get_sim_station()[stnp.nu_vertex]
                        vertex_R = np.sqrt(simulated_vertex[0]**2 + simulated_vertex[1]**2 + (simulated_vertex[2]+94)**2)
                        print("simualted vertex", simulated_vertex)
                        vertex_zenith = hp.cartesian_to_spherical(simulated_vertex[0], simulated_vertex[1], (simulated_vertex[2]+94))[0]
                        vertex_azimuth = hp.cartesian_to_spherical(simulated_vertex[0], simulated_vertex[1], (simulated_vertex[2]+94))[1]
                        
                        #### add uncertainties
                        zenith_uncertainty = 0
                        azimuth_uncertainty =0
                        R_uncertainty = 0
                        vertex_zenith += zenith_uncertainty
                        vertex_azimuth += azimuth_uncertainty
                        vertex_R += R_uncertainty
                        new_vertex = vertex_R *hp.spherical_to_cartesian(vertex_zenith, vertex_azimuth)
                        simulated_vertex = [new_vertex[0], new_vertex[1], new_vertex[2]-94]
                        print("vertex including uncertainties", simulated_vertex)
                        print("simulated neutrino zenith", np.rad2deg(station.get_sim_station()[stnp.nu_zenith]))
                        print("simulated neutrino azimuth", np.rad2deg(station.get_sim_station()[stnp.nu_azimuth]))
        
        
            options = {'maxiter':500, 'disp':True}
            
            print("SIMULATED INPUT VALUES")
            tracsim = minimizer([simulated_zenith,simulated_azimuth, simulated_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  False, fit = fitprocedure, first_iter = True)
         
            
            #fsim = minimizer([simulated_zenith,simulated_azimuth, simulated_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = fitprocedure, first_iter = True)
            #fsim = minimizer([simulated_zenith,simulated_azimuth, simulated_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = fitprocedure, first_iter = True)
            #print("fsim", fsim)
            #print(stop)
            
            
            if seperate_fit:
                print("###### Direction and energy fit are performed seperately, first fit neutrino direction ...")
                #results = opt.minimize(minimizer, x0 = ( [simulated_zenith, simulated_azimuth ]), args = ( simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], 'True', 'seperate'), method = 'Nelder-Mead', options = options)
                #print("Results", results)
                #print("##### Then, use reconstructed neutrino direction to fit energy. ")
                #rec_zenith = results.x[0]
                #rec_azimuth = results.x[1]
                #bnds = ((10**17, 10**19 ), )
                
                energies = np.logspace(17, 19, 50)
                zvalues = []
                az = np.arange(simulated_azimuth - np.deg2rad(5), simulated_azimuth + np.deg2rad(5), np.deg2rad(.5))
                zen = np.arange(simulated_zenith - np.deg2rad(5), simulated_zenith + np.deg2rad(5), np.deg2rad(.5))
                azimuth = []
                zenith = []
                zplot = []
                zvalue_lowest = 1000000
                print("first iteration for direction fit  ...")
                for a in az:
                    for z in zen:
                        zvalue = minimizer([z, a], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = fitprocedure)
                        if zvalue < zvalue_lowest:
                            global_az = a
                            global_zen = z
                            zvalue_lowest = zvalue
                
                
                
                rec_zenith = global_zen
                rec_azimuth = global_az
                print("     reconstructed zenith = {}".format(np.rad2deg(rec_zenith)))
                print("     reconstructed azimuth = {}".format(np.rad2deg(rec_azimuth)))
                print("min value", zvalue_lowest)
                
                energies = np.logspace(17, 19, 50)
                zvalues = []
                for ener in energies:
                    zvalues.append(minimizer([ener],  simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize = True, fit = 'seperate', timing_k = False, first_iter = False))
            
                #results = opt.minimize(minimizer, x0 = ( [1*10**19]), args = ( simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], 'True', 'seperate'), method = 'Nelder-Mead', options = options, bounds = bnds)
                rec_energy = energies[np.argmin(zvalues)]#results.x[0] ## we need the L-BFGS-B method because we want to set bounds
                
                zvalues = []
                az = np.arange(simulated_azimuth - np.deg2rad(5), simulated_azimuth + np.deg2rad(5), np.deg2rad(.5))
                zen = np.arange(simulated_zenith - np.deg2rad(5), simulated_zenith + np.deg2rad(5), np.deg2rad(.5))
                azimuth = []
                zenith = []
                zplot = []
                zvalue_lowest = 1000000
                print("second iteration for direction fit  ...")
                for a in az:
                    for z in zen:
                        zvalue = minimizer([z, a, rec_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = fitprocedure)
                        zplot.append(zvalue)
                        azimuth.append(a)
                        zenith.append(z)
                        if zvalue < zvalue_lowest:
                            global_az = a
                            global_zen = z
                            zvalue_lowest = zvalue
                
                
                
                rec_zenith = global_zen
                rec_azimuth = global_az
                
                fig = plt.figure()
                matplotlib.rc('xtick', labelsize = 10)
                matplotlib.rc('ytick', labelsize = 10)
                ax =fig.add_subplot(111)
                cs = ax.pcolor(np.rad2deg(np.array(azimuth)).reshape(len(az), len(zen)), np.rad2deg(np.array(zenith)).reshape(len(az), len(zen)), np.array(zplot).reshape(len(az), len(zen)), cmap = 'Greys_r')
                
                ax.contour(np.rad2deg(np.array(azimuth)).reshape(len(az), len(zen)), np.rad2deg(np.array(zenith)).reshape(len(az), len(zen)), np.array(zplot).reshape(len(az), len(zen)), [min(zplot) + 0.5, min(zplot) + 2, min(zplot) + 4.5], colors = 'red')


                ax.axvline(np.rad2deg(rec_azimuth), label = 'reconstructed fit', color = 'red')
                ax.axhline(np.rad2deg(rec_zenith),  color = 'red')

                ax.axvline(np.rad2deg(simulated_azimuth), label = 'simulated direction', color = 'orange')
                ax.axhline(np.rad2deg(simulated_zenith), color = 'orange')
                ax.axvline(np.rad2deg(global_az), color = 'lightblue', ls = '--')
                ax.axhline(np.rad2deg(global_zen), label = 'reconstructed global', color = 'lightblue', ls = '--')
                ax.legend()
                ax.set_xlabel("azimuth [degrees]")
                ax.set_ylabel("zenith [degrees]")
                ax.set_title("reconstructed energy and varying azimuth and zenith")
                minor_ticksx = np.round(np.arange(min(np.rad2deg(azimuth)), max(np.rad2deg(azimuth)), 2), 1)
                minor_ticksy = np.round(np.arange(min(np.rad2deg(zenith)), max(np.rad2deg(zenith)), 0.5), 1)
                ax.set_xticks(minor_ticksx)
                ax.set_yticks(minor_ticksy)
                ax.grid(alpha = .5)

                cbar = fig.colorbar(cs)
                cbar.set_label('-1 * log(L)', rotation=270)
                fig.tight_layout()
                
                fig.savefig("direction_{}.pdf".format(event.get_id()))
                
                
                print("     reconstructed zenith = {}".format(np.rad2deg(rec_zenith)))
                print("     reconstructed azimuth = {}".format(np.rad2deg(rec_azimuth)))
                
                print("###### seperate fit reconstructed valus")
                print("         zenith = {}".format(np.rad2deg(rec_zenith)))
                print("         azimuth = {}".format(np.rad2deg(rec_azimuth)))
                print("        energy = {}".format(rec_energy))
                print("         simualted zenith {}".format(np.rad2deg(simulated_zenith)))
                print("         simualted azimuth {}".format(np.rad2deg(simulated_azimuth)))
            
            
            
            
            if combined_fit:
                print("#### Neutrino direction and energy are fit simultaneously.")
                bnds = ((simulated_zenith - np.deg2rad(5), simulated_zenith + np.deg2rad(5)),(simulated_azimuth - np.deg2rad(5), simulated_azimuth + np.deg2rad(5)), (10**17, 10**19 )) ## these do not work for Nelder-Mead
                
                results = opt.minimize(minimizer, x0 = ( [rec_zenith, rec_azimuth, rec_energy]), args = ( simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], 'True', 'combined'), method = 'Nelder-Mead', options = options)
                rec_zenith = results.x[0]
                rec_azimuth = results.x[1]
                rec_energy = results.x[2]
                print("simulated energy", simulated_energy)
                print("Results", results)
            
            
            
            
            print("RECONSTRUCTED DIRECTION ZENITH {} AZIMUTH {}".format(np.rad2deg(rec_zenith), np.rad2deg(rec_azimuth)))
            print("RECONSTRUCTED ENERGY", rec_energy)

            
            ## get the traces for the reconstructed energy and direction
            tracrec = minimizer([rec_zenith, rec_azimuth, rec_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize = False, fit = fitprocedure)
            
            ## get the min likelihood value for the simulated values
            fminsim = minimizer([simulated_zenith, simulated_azimuth, simulated_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = 'combined')
            fminrec = minimizer([rec_zenith, rec_azimuth, rec_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = 'combined')
            print("FMIN SIMULATED VALUE", fminsim)
            print("FMIN RECONSTRUCTED VALUE", fminrec)
            
            
            
            
            
            
            minimizer_plot = 0
            if minimizer_plot:
                az = np.arange(simulated_azimuth - np.deg2rad(5), simulated_azimuth + np.deg2rad(5), np.deg2rad(.5))
                zen = np.arange(simulated_zenith - np.deg2rad(5), simulated_zenith + np.deg2rad(5), np.deg2rad(.5))
                azimuth = []
                zenith = []
                zplot = []
                zvalue_lowest = 1000000
                print("creating direction plot ...")
                for a in az:
                    for z in zen:
                        print("         azimuth {} and zenith {}".format(np.rad2deg(a), np.rad2deg(z)))
                        zvalue = minimizer([z, a, rec_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = fitprocedure)
                        zplot.append(zvalue)
                        if zvalue < zvalue_lowest:
                            global_az = a
                            global_zen = z
                            zvalue_lowest = zvalue
                        azimuth.append(a)
                        zenith.append(z)
            
                fig = plt.figure()
                matplotlib.rc('xtick', labelsize = 5)
                matplotlib.rc('ytick', labelsize = 5)
                ax =fig.add_subplot(111)
                cs = ax.pcolor(np.rad2deg(np.array(azimuth)).reshape(len(az), len(zen)), np.rad2deg(np.array(zenith)).reshape(len(az), len(zen)), np.array(zplot).reshape(len(az), len(zen)), cmap = 'Greys_r')
                
                ax.contour(np.rad2deg(np.array(azimuth)).reshape(len(az), len(zen)), np.rad2deg(np.array(zenith)).reshape(len(az), len(zen)), np.array(zplot).reshape(len(az), len(zen)), [min(zplot) + 0.5, min(zplot) + 2, min(zplot) + 4.5], colors = 'red')


                ax.axvline(np.rad2deg(rec_azimuth), label = 'reconstructed fit', color = 'red')
                ax.axhline(np.rad2deg(rec_zenith),  color = 'red')

                ax.axvline(np.rad2deg(simulated_azimuth), label = 'simulated direction', color = 'orange')
                ax.axhline(np.rad2deg(simulated_zenith), color = 'orange')
                ax.axvline(np.rad2deg(global_az), color = 'lightblue', ls = '--')
                ax.axhline(np.rad2deg(global_zen), label = 'reconstructed global', color = 'lightblue', ls = '--')
                ax.legend(fontsize = 'large')
                ax.set_xlabel("azimuth [degrees]")
                ax.set_ylabel("zenith [degrees]")
                ax.set_title("simulated energy and varying azimuth and zenith")
                minor_ticksx = np.round(np.arange(min(np.rad2deg(azimuth)), max(np.rad2deg(azimuth)), 2), 1)
                minor_ticksy = np.round(np.arange(min(np.rad2deg(zenith)), max(np.rad2deg(zenith)), 0.5), 1)
                ax.set_xticks(minor_ticksx)
                ax.set_yticks(minor_ticksy)
                ax.grid(alpha = .5)

                cbar = fig.colorbar(cs)
                cbar.set_label('-1 * log(L)', rotation=270)
                fig.tight_layout()
                
                fig.savefig("direction_{}.pdf".format(event.get_id()))
            
            debug_plot = 1
            if debug_plot:
                print("reconstruction value global", minimizer([global_zen, global_az], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize = True, fit = fitprocedure))
                print("reconstruction value simulation", minimizer([simulated_zenith,simulated_azimuth], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = fitprocedure))
                tracglobal = minimizer([global_zen, global_az], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize = False, fit = fitprocedure)
                
                tracglobal_k = minimizer([rec_zenith, rec_azimuth, rec_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize = False, fit = fitprocedure, timing_k = True)
                
                fig, ax = plt.subplots(len(use_channels), 3, sharex=False, figsize=(40, 20))
                matplotlib.rc('xtick', labelsize = 30)
                matplotlib.rc('ytick', labelsize = 30)
                ich = 0
                for channel in station.iter_channels():
                    if channel.get_id() in use_channels: # use channels needs to be sorted
                        
                        k1 = tracglobal_k[channel.get_id()]
                        print("k1", k1)
                        if len(k1) > 0:
                            k = k1[0]
                            ax[ich][0].plot(channel.get_trace(), label = 'data', color = 'black')
                            ax[ich][2].plot( np.fft.rfftfreq(len(channel.get_trace()), 1/sampling_rate), abs(fft.time2freq(channel.get_trace(), sampling_rate)), color = 'black')
                            ax[ich][0].plot(tracsim[ich], label = 'simulation', color = 'orange')
                            ax[ich][0].plot(tracrec[ich], label = 'reconstruction', color = 'green')
                            #ax[ich][0].plot(tracglobal[ich], label = 'global reconstruction', color = 'lightblue')
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[ich]), 1/sampling_rate), abs(fft.time2freq(tracsim[ich], sampling_rate)), color = 'orange')
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[ich]), 1/sampling_rate), abs(fft.time2freq(tracrec[ich], sampling_rate)), color = 'green')
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracglobal[ich]), 1/sampling_rate), abs(fft.time2freq(tracglobal[ich], sampling_rate)), color = 'lightblue')
                            N = 50
                            ax[ich][0].axvline((k-N), label = 'fitting area')
                            ax[ich][0].axvline((k+2*N))
                            ax[ich][0].set_xlim((k-2*N, k+4*N))
                            ax[ich][0].legend(fontsize = 'x-large')
                        if len(k1) > 1:
                            k = k1[1]
                            ax[ich][1].plot(channel.get_trace(), label = 'data', color = 'black')
                            ax[ich][2].plot( np.fft.rfftfreq(len(channel.get_trace()), 1/sampling_rate), abs(fft.time2freq(channel.get_trace(), sampling_rate)), color = 'black')
                            ax[ich][1].plot(tracsim[ich], label = 'simulation', color = 'orange')
                            ax[ich][1].plot(tracrec[ich], label = 'reconstruction', color = 'green')
                            #ax[ich][1].plot(tracglobal[ich], label = 'global reconstruction', color = 'lightblue')
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[ich]), 1/sampling_rate), abs(fft.time2freq(tracsim[ich], sampling_rate)), color = 'orange')
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[ich]), 1/sampling_rate), abs(fft.time2freq(tracrec[ich], sampling_rate)), color = 'green')
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracglobal[ich]), 1/sampling_rate), abs(fft.time2freq(tracglobal[ich], sampling_rate)), color = 'lightblue')
                            
                            
                            N = 50
                            ax[ich][1].axvline((k-N), label = 'fitting area')
                            ax[ich][1].axvline((k+2*N))
                            ax[ich][1].set_xlim((k-2*N, k+4*N))
                            ax[ich][1].legend(fontsize = 'large')
                        
                        
                        ich += 1
                fig.tight_layout()
                fig.savefig("fit_{}.pdf".format(event.get_id()))





    
        
        
    def end(self):
        pass
