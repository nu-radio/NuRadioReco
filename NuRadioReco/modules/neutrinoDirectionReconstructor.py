from scipy import constants
import scipy.stats as stats 
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
import datetime
import math
from NuRadioReco.utilities import units

class neutrinoDirectionReconstructor:
    
    
    def __init__(self):
        pass

    def begin(self, station, det, event, shower_id, use_channels=[6, 14]):
        """
        begin method. This function is executed before the event loop.

        We do not use this function for the reconsturctions. But only to determine uncertainties.
        """

        self._station = station
        self._use_channels = use_channels
        self._det = det
        self._sampling_rate = station.get_channel(0).get_sampling_rate()
        simulated_energy = event.get_sim_shower(shower_id)[shp.energy]

        self._simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
        self._simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]
        vertex =event.get_sim_shower(shower_id)[shp.vertex] #station[stnp.nu_vertex]
        simulation = propagated_analytic_pulse.simulation(True, vertex)#event.get_sim_shower(shower_id)[shp.vertex])
        simulation.begin(det, station, use_channels, raytypesolution = 2)#[1, 2, 3] [direct, refracted, reflected]
        print("simulated zenith", np.rad2deg(self._simulated_zenith))
        print("simulatd azimuth", np.rad2deg(self._simulated_azimuth))        
        a, b, self._launch_vector_sim, c, d, e =  simulation.simulation(det, station, vertex[0],vertex[1], vertex[2], self._simulated_zenith, self._simulated_azimuth, simulated_energy, use_channels, first_iter = True)
        print("LAN VECTOR SIM", self._launch_vector_sim)
        print("viewing angle", np.rad2deg(c))
        #print(stop)
        self._simulation = simulation
        pass
    
    def run(self, event, shower_id, station, det,
            use_channels=[6, 14], filenumber = 1, debugplots_path = None, template = False, sim_vertex = True, Vrms = 0.0114):
        
        
        
        
        ## there are 3 options for analytic models for the fit: 
        # - the Alvarez2009 model
        # - ARZ read in by templates
        # - ARZ average model 
        
        
        only_simulation = True# if True, no fit is performed, but only the script is tested with the simulated values

        restricted_input = True#False## For testing, we can restrict the neutrino direction with 10 degrees around the simulated one, (60 energy steps). For 2 channels this takes 6 min. Without restrictring it takes 10 min for 2 channels, (10 energy steps)
        
        self._station = station
        self._use_channels = use_channels
        self._det = det
        self._model_sys = 0.0
        if sim_vertex:
            reconstructed_vertex = event.get_sim_shower(shower_id)[shp.vertex]
        else:
            reconstructed_vertex = station[stnp.nu_vertex]
        
        simulation = propagated_analytic_pulse.simulation(template, reconstructed_vertex) ### if the templates are used, than the templates for the correct distance are loaded
        rt = ['direct', 'refracted', 'reflected'].index(self._station[stnp.raytype]) + 1 ## raytype from the triggered pulse
        simulation.begin(det, station, use_channels, raytypesolution = rt)
        self._simulation = simulation
       
    
    
        sigma = Vrms
      
        
    
        station.set_is_neutrino()
        if station.has_sim_station():
           
            sim_station = True
            simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]
            simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
            self._simulated_azimuth = simulated_azimuth
            simulated_energy = event.get_sim_shower(shower_id)[shp.energy]
            self.__simulated_energy = simulated_energy
            simulated_vertex = event.get_sim_shower(shower_id)[shp.vertex]
            ### values for simulated vertex and simulated direction
            tracsim, timsim, lv_sim, vw_sim, a, pol_sim = simulation.simulation(det, station, event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], simulated_zenith, simulated_azimuth, simulated_energy, use_channels, first_iter = True) 
       
            ## check SNR of channels
            SNR = []
            for ich, channel in enumerate(station.iter_channels()): ## checks SNR of channels
                Vrms = sigma#1.6257*10**(-5)
                print("channel {}, SNR {}".format(channel.get_id(),(abs(min(channel.get_trace())) + max(channel.get_trace())) / (2*Vrms) ))
                if channel.get_id() in use_channels:
                    Vrms = sigma#1.6257*10**(-5) # 16 micro volt
                    SNR.append((abs(abs(min(channel.get_trace()))) + max(channel.get_trace())) / (2*Vrms))
                    print("SNR", SNR)
        
       
    
        channl = station.get_channel(use_channels[0])
        n_samples = channl.get_number_of_samples()
        self._sampling_rate = channl.get_sampling_rate()
        sampling_rate = self._sampling_rate
        
        combined_fit = True #direction and energy are fitted simultaneously
        if combined_fit: ## Often a local minima is found
            fitprocedure = 'combined'
        else:
            print("Fit procedure is not implemented..")
       

        
        ### helper functions for plotting
        def mollweide_azimuth(az):
            az -= (simulated_azimuth - np.deg2rad(180)) ## put simulated azimuth at 180 degrees
            az = np.remainder(az, np.deg2rad(360)) ## rotate values such that they are between 0 and 360
            az -= np.deg2rad(180)
            return az
        
        def mollweide_zenith(zen):
            zen -= (simulated_zenith  - np.deg2rad(90)) ## put simulated azimuth at 90 degrees
            zen = np.remainder(zen, np.deg2rad(180)) ## rotate values such that they are between 0 and 180
            zen -= np.deg2rad(90) ## hisft to mollweide projection
            return zen


        def get_normalized_angle(angle, degree=False, interval=np.deg2rad([0, 360])):
            import collections
            if degree:
                interval = np.rad2deg(interval)
            delta = interval[1] - interval[0]
            if(isinstance(angle, (collections.Sequence, np.ndarray))):
                angle[angle >= interval[1]] -= delta
                angle[angle < interval[0]] += delta
            else:
                while (angle >= interval[1]):
                    angle -= delta
                while (angle < interval[0]):
                    angle += delta
            return angle
        
        
      
       
      
            
           
        if 1:
            add_vertex_uncertainties = False
            if add_vertex_uncertainties:
                for ie, efield in enumerate(station.get_sim_station().get_electric_fields()):
                    if efield.get_channel_ids()[0] == 1:
                        #simulated_vertex = station.get_sim_station()[stnp.nu_vertex]
                        vertex_R = np.sqrt((simulated_vertex[0] )**2 + simulated_vertex[1]**2 + (simulated_vertex[2]+100)**2)
                        vertex_zenith = hp.cartesian_to_spherical(simulated_vertex[0] , simulated_vertex[1], (simulated_vertex[2]+100))[0]
                        vertex_azimuth = hp.cartesian_to_spherical(simulated_vertex[0] , simulated_vertex[1], (simulated_vertex[2]+100))[1]

                        #### add uncertainties in radians 
                        zenith_uncertainty = 0#np.deg2rad(100)
                        azimuth_uncertainty = 0  #np.deg2rad(1)
                        R_uncertainty = 0
                        vertex_zenith += zenith_uncertainty
                        vertex_azimuth += azimuth_uncertainty
                        vertex_R += R_uncertainty
                        new_vertex = vertex_R *hp.spherical_to_cartesian(vertex_zenith, vertex_azimuth)
                        new_vertex = [(new_vertex[0] ), new_vertex[1], new_vertex[2]-100]

                        vertex_R = np.sqrt((new_vertex[0] )**2 + new_vertex[1]**2 + (new_vertex[2]+100)**2)
        
            print("simulated vertex", simulated_vertex)
            print('reconstructed', reconstructed_vertex)
           
            
           
            
            
            #### values for reconstructed vertex and simulated direction
            if sim_station:
                traces_sim, timing_sim, self._launch_vector, viewingangles_sim, rayptypes, a = simulation.simulation( det, station, reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], simulated_zenith, simulated_azimuth, simulated_energy, use_channels, fit = 'combined', first_iter = True)

         
                fsimsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  True, fit = fitprocedure, first_iter = True) 
                all_fsimsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, fit = fitprocedure, first_iter = True)[3]
                print("ALL FSIMSIM", all_fsimsim)
                tracsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = fitprocedure, first_iter = True)[0]
          
                fsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  True, fit = fitprocedure, first_iter = True)
              
                all_fsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = fitprocedure, first_iter = True)[3]
                print("Chi2 values for simulated direction and with/out simulated vertex are {}/{}".format(fsimsim, fsim))
            
                sim_reduced_chi2_Vpol = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = fitprocedure)[4][0]
                #print("sim reduced chi2 VPol", sim_reduced_chi2_Vpol)
                #print(stop)
                sim_reduced_chi2_Hpol = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = fitprocedure, first_iter = True)[4][1]
        
        
#sim_reduced_chi2_Vpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = 'combined')[4][0]
 #           sim_reduced_chi2_Hpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = 'combined')[4][1]            
            
           # trac = self.minimizer([simulated_zenith,simulated_azimuth, simulated_energy], reconstructed_vertex[0],reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = fitprocedure, first_iter = True) ## this is to store the correct launch vector for the new vertex position 
           # print(stop)

          
          
            
          
            
            
            if combined_fit:
              
                signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*self._launch_vector) ## due to 
                sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)


                cherenkov = 56        
                viewing_start = np.deg2rad(cherenkov) - np.deg2rad(15) 
                viewing_end = np.deg2rad(cherenkov) + np.deg2rad(15)
                theta_start = np.deg2rad(-180)
                theta_end =  np.deg2rad(180)

                import datetime
                cop = datetime.datetime.now()
                print("SIMULATED DIRECTION {} {}".format(np.rad2deg(simulated_zenith), np.rad2deg(simulated_azimuth)))

                if only_simulation:
                    print("no reconstructed is performed. The script is tested..")
                elif not restricted_input:
                    results = opt.brute(self.minimizer, ranges=(slice(viewing_start, viewing_end, np.deg2rad(1)), slice(theta_start, theta_end, np.deg2rad(1)), slice(np.log10(simulated_energy) - .5, np.log10(simulated_energy) + .5, .1)), full_output = True, finish = opt.fmin , args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True,fitprocedure, False, False, True, False))
                else: 
                    zenith_start = simulated_zenith - np.deg2rad(10)
                    zenith_end = simulated_zenith +  np.deg2rad(10)
                    azimuth_start = simulated_azimuth - np.deg2rad(10)
                    azimuth_end = simulated_azimuth + np.deg2rad(10)
                    energy_start = np.log10(simulated_energy) - 1
                    energy_end = np.log10(simulated_energy) + 1
                    results = opt.brute(self.minimizer, ranges=(slice(zenith_start, zenith_end, np.deg2rad(1)), slice(azimuth_start, azimuth_end, np.deg2rad(1)), slice(energy_start, energy_end, .1)), finish = opt.fmin, full_output = True, args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True,fitprocedure, False, False, False, False))
                    
                print('start datetime', cop)
                print("end datetime", datetime.datetime.now() - cop)
					
           


                if only_simulation:
                    rec_zenith = simulated_zenith
                    rec_azimuth = simulated_azimuth
                    rec_energy = simulated_energy
                
                elif not restricted_input:
                    rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))
                    cherenkov_angle = results[0][0]
                    angle = results[0][1]

                    p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                    p3 = rotation_matrix.dot(p3)
                    global_az = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                    global_zen = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                    global_zen = np.deg2rad(180) - global_zen
                    

                    rec_zenith = global_zen
                    rec_azimuth = global_az
                    rec_energy = 10**results[0][2]
                else:
                    rec_zenith = results[0][0]
                    rec_azimuth = results[0][1]
                    rec_energy = 10**results[0][2]

                
                print("reconstructed energy {}".format(rec_energy))
                print("reconstructed zenith {} and reconstructed azimuth {}".format(np.rad2deg(rec_zenith), np.rad2deg(self.transform_azimuth(rec_azimuth))))
                print("         simualted zenith {}".format(np.rad2deg(simulated_zenith)))
                print("         simualted azimuth {}".format(np.rad2deg(simulated_azimuth)))      
                
                print("     reconstructed zenith = {}".format(np.rad2deg(rec_zenith)))
                print("     reconstructed azimuth = {}".format(np.rad2deg(self.transform_azimuth(rec_azimuth))))
                
                print("###### seperate fit reconstructed valus")
                print("         zenith = {}".format(np.rad2deg(rec_zenith)))
                print("         azimuth = {}".format(np.rad2deg(self.transform_azimuth(rec_azimuth))))
                print("        energy = {}".format(rec_energy))
                print("         simualted zenith {}".format(np.rad2deg(simulated_zenith)))
                print("         simualted azimuth {}".format(np.rad2deg(simulated_azimuth)))
                print("         simulated energy {}".format(simulated_energy)) 
            
            
            
        
            
            
            
            
            print("RECONSTRUCTED DIRECTION ZENITH {} AZIMUTH {}".format(np.rad2deg(rec_zenith), np.rad2deg(self.transform_azimuth(rec_azimuth))))
            print("RECONSTRUCTED ENERGY", rec_energy)
            
           
            
            ## get the traces for the reconstructed energy and direction
            tracrec = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, fit = 'combined')[0]
            fit_reduced_chi2_Vpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, fit = 'combined')[4][0]
            fit_reduced_chi2_Hpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, fit = 'combined')[4][1]             
           
            fminfit = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  True, fit = 'combined')            
           
            all_fminfit = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = 'combined')[3]
       
            #sim_reduced_chi2_Vpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = 'combined')[4][0]
            #sim_reduced_chi2_Hpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = 'combined')[4][1]

            print("FMIN SIMULATED direction with reconstructed vertex", fsim)
            print("FMIN RECONSTRUCTED VALUE FIT", fminfit)
            
            
           
            
          
            debug_plot = 1
            if debug_plot:
                
                tracdata = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, fit = fitprocedure)[1]
             
                timingdata = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, fit = fitprocedure)[2]
                timingsim = self.minimizer([simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, fit = fitprocedure)[2]
                              

                fig, ax = plt.subplots(len(use_channels), 3, sharex=False, figsize=(40, 20))
                matplotlib.rc('xtick', labelsize = 30)
                matplotlib.rc('ytick', labelsize = 30)
                ich = 0
                SNRs = np.zeros((len(use_channels), 2)) 
                fig_sim, ax_sim = plt.subplots(len(use_channels), 1, sharex = True, figsize = (20, 10))
                fig_test, ax_test = plt.subplots(len(use_channels), 1, sharex = True, figsize = (20, 10))

                for channel in station.iter_channels():
                    if channel.get_id() in use_channels: # use channels needs to be sorted
                        isch = 0
                        for sim_channel in self._station.get_sim_station().get_channels_by_channel_id(channel.get_id()):
                            if isch == 0:
                                        
                                sim_trace = sim_channel
                                ax_test[ich].plot(sim_channel.get_times(), sim_channel.get_trace())
                                ax_sim[ich].plot(sim_channel.get_times(), sim_channel.get_trace())
                            if isch == 1:
                                ax_test[ich].plot(sim_channel.get_times(), sim_channel.get_trace())
                                ax_sim[ich].plot(sim_channel.get_times(), sim_channel.get_trace())
                                sim_trace += sim_channel
                            isch += 1
                       
                        ax_sim[ich].plot(channel.get_times(), channel.get_trace())
                        ax_sim[ich].plot(sim_trace.get_times(), sim_trace.get_trace())
                       
                        
                        if len(tracdata[channel.get_id()]) > 0:
                            #print('max', max(tracdata[channel.get_id()][0]))
                            #SNR = abs(max(tracdata[channel.get_id()][0]) - min(tracdata[channel.get_id()][0])) / (2*sigma)
                            
                            #print("LEN DATA", len(tracdata[channel.get_id()][0]))
                            #SNRs[ich, 0] = SNR
                            #ax[ich][0].plot(channel.get_trace(), label = 'data', color = 'black')
                            ax[ich][0].plot(channel.get_times(), channel.get_trace(), label = 'data', color = 'black')
                            #ax[ich][0].plot(timingdata[channel.get_id()][0], tracdata[channel.get_id()][0], label = 'data', color = 'black')
                            ax[ich][0].fill_between(timingsim[channel.get_id()][0],tracsim[channel.get_id()][0]- sigma, tracsim[channel.get_id()][0] + sigma, color = 'red', alpha = 0.2 )
                            ax[ich][0].fill_between(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0] - self._model_sys*tracrec[channel.get_id()][0], tracrec[channel.get_id()][0] + self._model_sys * tracrec[channel.get_id()][0], color = 'green', alpha = 0.2)
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq( tracdata[channel.get_id()][0], sampling_rate)), color = 'black')
                            ax[ich][0].plot(timingsim[channel.get_id()][0], tracsim[channel.get_id()][0], label = 'simulation', color = 'orange')
                            ax[ich][0].plot(sim_trace.get_times(), sim_trace.get_trace(), label = 'sim channel', color = 'red', lw = 3)
                            ax[ich][0].set_xlim((timingsim[channel.get_id()][0][0], timingsim[channel.get_id()][0][-1]))
            
                            if 1:#channel.get_id() in [6]:#,7,8,9,13,14]: 
                                 ax[ich][0].plot(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0], label = 'reconstruction', color = 'green')
                            #elif SNR > 3.5:
                                 ax[ich][0].plot(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0], label = 'reconstruction', color = 'green')


                            ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][0], sampling_rate)), color = 'orange')
                            if 1:#channel.get_id() in [6]:
                                 ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][0], sampling_rate)), color = 'green')
                         
                            #elif SNR > 3.5:
                            #     ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][0], sampling_rate)), color = 'green')

                            ax[ich][0].legend(fontsize = 'x-large')
                           
                        if len(tracdata[channel.get_id()]) > 1:
                            #SNR = abs(max(tracdata[channel.get_id()][1]) - min(tracdata[channel.get_id()][1])) / (2*sigma)
                            #SNRs[ich, 1] = SNR
                            ax[ich][1].plot(channel.get_times(), channel.get_trace(), label = 'data', color = 'black')
                            #print("ich", ich)
                            #ax[ich][1].plot(channel.get_times(), channel.get_trace(), label = 'data', color = 'black')
                            #ax[ich][1].plot(timingdata[channel.get_id()][1], tracdata[channel.get_id()][1], label = 'data', color = 'black')
                            ax[ich][1].fill_between(timingsim[channel.get_id()][1],tracsim[channel.get_id()][1]- sigma, tracsim[channel.get_id()][1] + sigma, color = 'red', alpha = 0.2 )

                            ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracdata[channel.get_id()][1], sampling_rate)), color = 'black')
                            ax[ich][1].plot(timingsim[channel.get_id()][1], tracsim[channel.get_id()][1], label = 'simulation', color = 'orange')
                            ax[ich][1].plot(sim_trace.get_times(), sim_trace.get_trace(), label = 'sim channel', color = 'red', lw = 3)
                            if 1:#channel.get_id() in [6]:#,7,8,9]: 
                                ax[ich][1].plot(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1], label = 'reconstruction', color = 'green')
                                ax[ich][1].fill_between(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1] - self._model_sys*tracrec[channel.get_id()][1], tracrec[channel.get_id()][1] + self._model_sys * tracrec[channel.get_id()][1], color = 'green', alpha = 0.2)

                            #elif SNR > 3.5:
                            #    ax[ich][1].plot(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1], label = 'reconstruction', color = 'green')
                             #   ax[ich][1].fill_between(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1] - self._model_sys*tracrec[channel.get_id()][1], tracrec[channel.get_id()][1] +self._model_sys * tracrec[channel.get_id()][1], color = 'green', alpha = 0.2)

                            ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][1], sampling_rate)), color = 'orange')
                            ax[ich][1].set_xlim((timingsim[channel.get_id()][1][0], timingsim[channel.get_id()][1][-1]))
                            if 1:#channel.get_id() in [6]:
                                 ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][1], sampling_rate)), color = 'green')
                            #elif SNR > 3.5:
                            #     ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][1], sampling_rate)), color = 'green')

                            
                        
                        
                        ich += 1
                fig.tight_layout()
                fig.savefig("{}/fit_{}_{}.pdf".format(debugplots_path, filenumber, shower_id))
                fig_sim.savefig('{}/sim_{}_{}.pdf'.format(debugplots_path, filenumber, shower_id))
                fig_test.savefig('{}/test_{}_{}.pdf'.format(debugplots_path, filenumber, shower_id))
                
                
                
             ### values for reconstructed vertex and reconstructed direction
            traces_rec, timing_rec, launch_vector_rec, viewingangle_rec, a, pol_rec =  simulation.simulation( det, station, reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], rec_zenith, rec_azimuth, rec_energy, use_channels, fit = 'combined', first_iter = True)  

            ### store parameters 

            station.set_parameter(stnp.nu_zenith, rec_zenith)
            station.set_parameter(stnp.nu_azimuth, self.transform_azimuth(rec_azimuth))
            station.set_parameter(stnp.nu_energy, rec_energy)
            station.set_parameter(stnp.chi2, [fsim, fminfit, fsimsim, self.__dof, sim_reduced_chi2_Vpol, sim_reduced_chi2_Hpol, fit_reduced_chi2_Vpol, fit_reduced_chi2_Hpol])
            station.set_parameter(stnp.launch_vector, [lv_sim, launch_vector_rec])
            station.set_parameter(stnp.polarization, [pol_sim, pol_rec])
            station.set_parameter(stnp.viewing_angle, [vw_sim, viewingangle_rec])
            print("chi2 for simulated rec vertex {}, simulated sim vertex {} and fit {}".format(fsim, fsimsim, fminfit))#reconstructed vertex
            print("chi2 for all channels simulated rec vertex {}, simulated sim vertex {} and fit {}".format(all_fsim, all_fsimsim, all_fminfit))#reconstructed vertex
            print("launch vector for simulated {} and fit {}".format(lv_sim, launch_vector_rec))
            print("polarization for simulated {} and fit {}".format(pol_sim, pol_rec))
            print("viewing angle for simulated {} and fit {}".format(np.rad2deg(vw_sim), np.rad2deg(viewingangle_rec)))
            print("reduced chi2 Vpol for simulated {} and fit {}".format(sim_reduced_chi2_Vpol, fit_reduced_chi2_Vpol))
            print("reduced chi2 Hpol for simulated {} and fit {}".format(sim_reduced_chi2_Hpol, fit_reduced_chi2_Hpol))

    def transform_azimuth(self, azimuth): ## from [-180, 180] to [0, 360]
        azimuth = np.rad2deg(azimuth)
        if azimuth < 0:
            azimuth = 360 + azimuth
        return np.deg2rad(azimuth)
    
    
                  
    def minimizer(self, params, vertex_x, vertex_y, vertex_z, minimize = True, fit = 'seperate', timing_k = False, first_iter = False, banana = False,  direction = [0, 0]):
            model_sys = 0
            sigma = 0.0114 #noise Rms for with amplifier #1.7*10**(-5) * 10000 
            ff = np.fft.rfftfreq(600, .1)
            mask = ff > 0
            order = 8
            passband = [300* units.MHz, 400* units.MHz]
            b, a = signal.butter(order, passband, 'bandpass', analog=True)
            w, ha = signal.freqs(b, a, ff[mask])
            fa = np.zeros_like(ff, dtype=np.complex)
            fa[mask] = ha
            pol_filt = fa
#            print("FF", ff)
#            print(stop)
       



            if banana:
                cherenkov_angle, angle, log_energy = params 
                energy = 10**log_energy
                


                signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*self._launch_vector)

                sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
                rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))


                p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                p3 = rotation_matrix.dot(p3)
                azimuth = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                zenith = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                zenith = np.deg2rad(180) - zenith
               


                if np.rad2deg(zenith) > 100:
                    return np.inf ## not in field of view
            else: 
                if len(params) ==3:
                    zenith, azimuth, log_energy = params 
                    energy = 10**log_energy
                if len(params) == 1:
                    log_energy = params
                    energy = 10**log_energy
                    zenith, azimuth = direction
            
            azimuth = self.transform_azimuth(azimuth)
            #energy = self.__simulated_energy
            print("energy", energy)
            print("parameters zen {} az {} energy {}".format(np.rad2deg(zenith), np.rad2deg(azimuth), energy))
            traces, timing, launch_vector, viewingangles, raytypes, pol = self._simulation.simulation(self._det, self._station, vertex_x, vertex_y, vertex_z, zenith, azimuth, energy, self._use_channels, fit, first_iter = first_iter)
            chi2 = 0
            all_chi2 = []

            dof = -3


            rec_traces = {}
            data_traces = {}
            data_timing = {}
        

            #get timing for raytype of triggered pulse
            for iS in raytypes[6]:
                if raytypes[6][iS] == ['direct', 'refracted', 'reflected'].index(self._station[stnp.raytype]) + 1:
                    solution_number = iS
            T_ref = timing[6][solution_number]
           
            k_ref = self._station[stnp.pulse_position]# get pulse position for triggered pulse
            
            ks = {}
            

            ich = -1
            reduced_chi2_Vpol = 0
            reduced_chi2_Hpol = 0
            channels_Vpol = [1,4,6,10, 11, 12]
            channels_Hpol = [2, 5, 13] 
            dict_dt = {}
            for channel in self._station.iter_channels():
            #    print("channel", channel.get_id())
                if (channel.get_id() in channels_Vpol) and (channel.get_id() in self._use_channels):# and (channel.get_id() in [self._use_channels]):#self._use_channels: #iterate over channels
                   # print("CHANNLE", channel.get_id())                
                    ich += 1 ## number of channel
                    data_trace = np.copy(channel.get_trace())
                    rec_traces[channel.get_id()] = {}
                    data_traces[channel.get_id()] = {}
                    data_timing[channel.get_id()] = {}

                    ### if no solution exist, than analytic voltage is zero
                    rec_trace = np.zeros(len(data_trace))# if there is no raytracing solution, the trace is only zeros

                    delta_k = [] ## if no solution type exist then channel is not included
                    num = 0
                    chi2s = np.zeros(2)
                    max_timing_index = np.zeros(2)
                    max_data = []
                    ### get dt for phased array
                    for i_trace, key in enumerate(traces[channel.get_id()]):#get dt for phased array pulse  
#                        print("i trace",i_trace) 

                        rec_trace_i = traces[channel.get_id()][key]
                        rec_trace = rec_trace_i

                        max_trace = max(abs(rec_trace_i))
                        delta_T =  timing[channel.get_id()][key] - T_ref
                      
                        if 1:#(np.round(delta_T) == 0):
                    #        print("delta T", delta_T)
                            ## before correlating, set values around maximum voltage trace data to zero
                            delta_toffset = delta_T * self._sampling_rate

                            ### figuring out the time offset for specfic trace
                            dk = int(k_ref + delta_toffset )
                            rec_trace1 = rec_trace

                              
                            if ((dk > 300)&(dk < len(np.copy(data_trace)) - 500)):#channel.get_id() == 9:#ARZ:#(channel.get_id() == 10 and i_trace == 1):
                            #    print("YES")
                                data_trace_timing = np.copy(data_trace) ## cut data around timing
                                ## DETERMIINE PULSE REGION DUE TO REFERENCE TIMING 


                                #print("dk", dk)
                                data_timing_timing = np.copy(channel.get_times())#np.arange(0, len(channel.get_trace()), 1)#
                                dk_1 = data_timing_timing[dk]

                                data_timing_timing = data_timing_timing[dk - 300 : dk + 500]
                                data_trace_timing = data_trace_timing[dk -300 : dk + 500]
                                data_trace_timing_1 = np.copy(data_trace_timing)
                                ### cut data trace timing to make window to search for pulse smaller
                                data_trace_timing_1[data_timing_timing < (dk_1 - 150)] = 0
                                data_trace_timing_1[data_timing_timing > (dk_1 + 250)] = 0
                                ##### dt is the same for a channel+ corresponding Hpol. Dt will be determined by largest trace
                                max_data_2 = 0
                                if (i_trace ==0): 

                                 #   print("trace 1")
                                    max_data_1 =  max(data_trace_timing_1)
                                if i_trace ==1:
                                    max_data_2 = max(data_trace_timing_1)
                                if ((i_trace ==0) or (i_trace ==1 & (max_data_2 > max_data_1))):
            #                    print("CHANNLE ID", channel.get_id())
                                    library_channels ={}
                                #library_channels[6] = {}
                                    library_channels[6] = [6,13]
                                    library_channels[1] = [1,2]
                                    library_channels[4] = [4,5]
                                    library_channels[10] = [10]
                                    library_channels[11] = [11]
                                    library_channels[12] = [12]
                     #           print("library", library_channels)
                                    if 1: 
                                        corr = signal.hilbert(signal.correlate(rec_trace1, data_trace_timing_1))
                                    dt = np.argmax(corr) - (len(corr)/2) +1
                                #    print("DT", dt)   
                               
                                    corresponding_channels = library_channels[channel.get_id()]
             #                       print("corresponding channels", corresponding_channels)
                                    for ch in corresponding_channels:
                                        dict_dt[ch] = dt

 
            for channel in self._station.iter_channels():
                if channel.get_id() in self._use_channels:
                    rec_traces[channel.get_id()] = {}
                    data_traces[channel.get_id()] = {}
                    data_timing[channel.get_id()] = {}
                    weight = 1
                    data_trace = np.copy(channel.get_trace())
                    data_trace_timing = data_trace 
 
                    for i_trace, key in enumerate(traces[channel.get_id()]): ## iterate over ray type solutions
                        if 1:
                            rec_trace_i = traces[channel.get_id()][key]
                            rec_trace = rec_trace_i

                            max_trace = max(abs(rec_trace_i))
                            delta_T =  timing[channel.get_id()][key] - T_ref

                            ## before correlating, set values around maximum voltage trace data to zero
                            delta_toffset = delta_T * self._sampling_rate

                            ### figuring out the time offset for specfic trace
                            dk = int(k_ref + delta_toffset )
                            rec_trace1 = rec_trace

                            
                            if ((dk > 300)&(dk < len(np.copy(data_trace)) - 500)):#channel.get_id() == 9:#ARZ:#(channel.get_id() == 10 and i_trace == 1):
                            #    print("YES")
                                data_trace_timing = np.copy(data_trace) ## cut data around timing
                                ## DETERMIINE PULSE REGION DUE TO REFERENCE TIMING 
                                

                                data_timing_timing = np.copy(channel.get_times())#np.arange(0, len(channel.get_trace()), 1)#
                                dk_1 = data_timing_timing[dk]
                                data_timing_timing = data_timing_timing[dk - 300 : dk + 500]
                                data_trace_timing = data_trace_timing[dk -300 : dk + 500]
                                data_trace_timing_1 = np.copy(data_trace_timing)
                                ### cut data trace timing to make window to search for pulse smaller
                                data_trace_timing_1[data_timing_timing < (dk_1 - 150)] = 0
                                data_trace_timing_1[data_timing_timing > (dk_1 + 250)] = 0
                                
                               
				#if channel.get_id() in [#corr = signal.hilbert(signal.correlate(rec_trace1, data_trace_timing_1))
                                #dt = np.argmax(corr) - (len(corr)/2) +1
                              
                                dt = dict_dt[channel.get_id()]
                                rec_trace1 = np.roll(rec_trace1, math.ceil(-1*dt))
                             
                               # fig = plt.figure()
                               # ax = fig.add_subplot(111)
                               # ax.plot(rec_trace1)
                               # ax.plot(data_trace_timing)
                               # fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/Uncertainties/1_direction_simulations/hoi.pdf")
                               # print(stop)
                                rec_trace1 = rec_trace1[150 : 750] 
                                data_trace_timing = data_trace_timing[150:  750]
                                #print("len data timing timing", len(data_timing_timing))
                                data_timing_timing = data_timing_timing[150:  750]
                                




                                delta_k.append(int(k_ref + delta_toffset + dt )) ## for overlapping pulses this does not work
                                ks[channel.get_id()] = delta_k

                                if fit == 'combined':

                                    rec_traces[channel.get_id()][i_trace] = rec_trace1
                                    data_traces[channel.get_id()][i_trace] = data_trace_timing
                                    data_timing[channel.get_id()][i_trace] = data_timing_timing
                                    #print("data timing", i_trace)                                  
                                    
                                    

                                    SNR = abs(max(data_trace_timing) - min(data_trace_timing) ) / (2*sigma)
                                 #   if channel.get_id() == 6:
                                 #       if i_trace == 0:
                                 #           SNR_1 = SNR ## find offset
                                 #       if i_trace ==1:
                                 #           SNR_2 = SNR
                                    ff = np.fft.rfftfreq(600, .1)
                                    mask = ff > 0
                                    order = 8
                                    passband = [200* units.MHz, 300* units.MHz]
                                    b, a = signal.butter(order, passband, 'bandpass', analog=True)
                                    w, ha = signal.freqs(b, a, ff[mask])
                                    fa = np.zeros_like(ff, dtype=np.complex)
                                    fa[mask] = ha
                                    pol_filt = fa
                                # if vertex is wrong reconstructed, than it can be that data_timing_timing does not exist. In that case, set to zero. 
                                    try:
                                        max_timing_index[i_trace] = data_timing_timing[np.argmax(data_trace_timing)]
                                    except: 
                                        max_timing_index[i_trace] = 0   

                                    if  channel.get_id() == 6:#(int(delta_T) == 0):
                                        if 1:#   
                                            trace_trig = i_trace

                                            chi2s[i_trace] = np.sum((rec_trace1 - data_trace_timing)**2 / ((sigma+model_sys*abs(data_trace_timing))**2))#/len(rec_trace1)
             #                           print("len data trace timing", len(data_trace_timing))
                                            data_tmp = fft.time2freq(data_trace_timing, self._sampling_rate) * pol_filt 
                                            power_data_6 = np.sum(fft.freq2time(data_tmp, self._sampling_rate)**2)                   
                                            rec_tmp = fft.time2freq(rec_trace1, self._sampling_rate) * pol_filt                     
                                            power_rec_6 = np.sum(fft.freq2time(rec_tmp, self._sampling_rate) **2)
                                            reduced_chi2_Vpol +=  weight*0.5* np.sum((rec_trace1 - data_trace_timing)**2 / ((sigma+model_sys*abs(data_trace_timing))**2))/len(rec_trace1)

                                    if channel.get_id() == 13:
                                        if  1:#max(data_timing[channel.get_id()][0]) < min(data_timing[channel.get_id()][1]):#i_trace == trace_trig:
                                            mask_trace = 200
                                            chi2s[i_trace] = np.sum((rec_trace1[0:mask_trace] - data_trace_timing[0:mask_trace])**2 / ((sigma)**2))/len(rec_trace1[0:mask_trace])
                                            data_tmp = fft.time2freq(data_trace_timing, self._sampling_rate) * pol_filt
                                            power_data_13 = np.sum((fft.freq2time(data_tmp, self._sampling_rate) )**2)
                                            rec_tmp = fft.time2freq(rec_trace1, self._sampling_rate) * pol_filt       
                                            power_rec_13 = np.sum((fft.freq2time(rec_tmp, self._sampling_rate) )**2)
                                            R_rec = power_rec_6/power_rec_13
                                            R_data = power_data_6/power_data_13
                                           # print("CHI@ HPOL",  0.5*np.sum((rec_trace1[0:200] - data_trace_timing[0:200])**2 / ((sigma**2))))
                                            if 0:#i_trace ==0:
                                                fig = plt.figure()
                                                ax = fig.add_subplot(111)
                                                ax.plot(data_trace_timing[0:mask_trace])
                                                ax.plot(rec_trace1[0:mask_trace])
                                                #ax.plot((data_trace_timing - rec_trace1)/15 /sigma)
                                                fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/Uncertainties/1_direction_simulations/hoi.pdf")
                                            reduced_chi2_Hpol +=  weight* 0.5*np.sum((rec_trace1[0:mask_trace] - data_trace_timing[0:mask_trace])**2 / ((sigma)**2))/len(rec_trace1[0:mask_trace])
                                           # print("reduced chi2 Hpol", weight* 0.5*np.sum((rec_trace1[0:mask_trace] - data_trace_timing[0:mask_trace])**2 / ((sigma)**2))/len(rec_trace1[0:mask_trace]))
                                    #        print("R_rec", R_rec)
                                    #        print("R_data", R_data)
                                    elif (SNR > 3.5):
                                        if channel.get_id() in channels_Vpol:
                                            mask_trace = 400
                                        if channel.get_id() in channels_Hpol:
                                            chi2s[i_trace] = np.sum((rec_trace1[0:mask_trace] - data_trace_timing[0:mask_trace])**2 / ((sigma)**2))#/len(rec_trace1[0:mask_trace])

                  
                    
                        else:
                            rec_traces[channel.get_id()][i_trace] = np.zeros(400)
                            data_traces[channel.get_id()][i_trace] = np.zeros(400)
                            data_timing[channel.get_id()][i_trace] = np.zeros(400)
                        

                    if 0:# (abs(max_timing_index[ 0] - max_timing_index[ 1]) < 300): #check if both pulses correspond to the same data. If so, only add minimum chi2
                     
                        chi2 += min(chi2s)
                        dof += len(rec_trace1)
                        all_chi2.append(min(chi2s)/len(rec_trace1))
                    else: ## else, add both
                        #print("data timing channel id", data_timing[channel.get_id()])
                        if max(data_timing[channel.get_id()][0]) > min(data_timing[channel.get_id()][1]):
                                time = np.arange(min(data_timing[channel.get_id()][0]), max(data_timing[channel.get_id()][1]),.1)
                                ff = np.fft.rfftfreq(len(time), .1)
                                mask = ff > 0
                                order = 8
                                passband = [200* units.MHz, 500* units.MHz]
                                b, a = signal.butter(order, passband, 'bandpass', analog=True)
                                w, ha = signal.freqs(b, a, ff[mask])
                                fa = np.zeros_like(ff, dtype=np.complex)
                                fa[mask] = ha
                                pol_filt = fa

                                model_sys = 0 
                                #if channel.get_id() ==6: 
                                #time = np.arange(min(data_timing[channel.get_id()][0]), max(data_timing[channel.get_id()][1]),.1)
                                #print("timing", data_timing[channel.get_id()][0])
                                data1 = np.append(data_traces[channel.get_id()][0], np.zeros(len(time) - len(data_traces[channel.get_id()][0])))
                                data2 = np.append(np.zeros(len(time) - len(data_traces[channel.get_id()][1])), data_traces[channel.get_id()][1])
                                #print("timing", time)
                                data1[(len(data_traces[channel.get_id()][0]))::] = data2[len(data_traces[channel.get_id()][0])::]
                                rec1 = np.append(rec_traces[channel.get_id()][0], np.zeros(len(time) - len(rec_traces[channel.get_id()][0])))
                                rec2 = np.append(np.zeros(len(time) - len(rec_traces[channel.get_id()][1])), rec_traces[channel.get_id()][1])
                                rec = rec1 + rec2
                                add = 0
                                if channel.get_id() ==6:
                                    add = 1
                                    weight = 1
                                    reduced_chi2_Vpol =  weight* np.sum((rec - data1)**2 / ((sigma+model_sys*abs(data1))**2))/len(rec)
                                    data_tmp = fft.time2freq(data1, self._sampling_rate) * pol_filt
                                    power_data_6 = np.sum(fft.freq2time(data_tmp, self._sampling_rate)**2)
                                    rec_tmp = fft.time2freq(rec, self._sampling_rate) * pol_filt
                                    power_rec_6 = np.sum(fft.freq2time(rec_tmp, self._sampling_rate) **2)    

                                if channel.get_id() ==13:
                                    weight = 1
                                    add =1 
                                    filter_Hpol = True
                                    if filter_Hpol: 
                                        rec = fft.time2freq(rec, self._sampling_rate) #* pol_filt
                                        rec = fft.freq2time(rec, self._sampling_rate) 
                                        data1 = fft.time2freq(data1, self._sampling_rate)# *pol_filt
                                        data1 = fft.freq2time(data1, self._sampling_rate) 
                                        reduced_chi2_Hpol = weight * np.sum((rec - data1)**2 / ((sigma)**2))/len(rec)
                                        noise_temp = 300
                                        bandwidth = 652 
                                        noise_rms = sigma/1.5#(noise_temp* 50 * constants.k * bandwidth / units.Hz) ** 0.5
                                        #sigma = noise_rms
                                        #print("Noise rms", noise_rms)
                                        reduced_chi2_Hpol =  weight* np.sum((rec - data1)**2 / ((sigma)**2))/len(rec)
                                    data_tmp = fft.time2freq(data1, self._sampling_rate) * pol_filt
                                    power_data_13 = np.sum(fft.freq2time(data_tmp, self._sampling_rate)**2)
                                    rec_tmp = fft.time2freq(rec, self._sampling_rate) * pol_filt
                                    power_rec_13 = np.sum(fft.freq2time(rec_tmp, self._sampling_rate) **2)
                                    R_data =  power_data_6/power_data_13 
                                    R_rec =  power_rec_6/power_rec_13      
                                    #chi2 += weight* np.sum((rec - data1)**2 / ((sigma)**2))/len(rec)
                                if (SNR > 3.5):
                                    add = 1
                                    data_tmp = fft.time2freq(data1, self._sampling_rate) * pol_filt
                                    power_data_6 = np.sum(fft.freq2time(data_tmp, self._sampling_rate)**2)
                                    rec_tmp = fft.time2freq(rec, self._sampling_rate) * pol_filt
                                    power_rec_6 = np.sum(fft.freq2time(rec_tmp, self._sampling_rate) **2)
                                
                                if add: chi2 += weight* np.sum((rec - data1)**2 / ((sigma)**2))#/len(rec)
                                #print("SImg aHPol", sigma)
                                sigma = 0.0114
                                dof += len(time)
                                print("ADDD DOF {}, chi2 {}, chi2 /dof {}".format( dof, chi2, chi2/dof))
                                if 0:#channel.get_id() == 6:
                                    fontsize = 30
                                    fig = plt.figure(figsize = (40, 40))
                                    ax = fig.add_subplot(421)
                                    ax.plot(time, data1, label = 'data')
                                    #ax.plot(time, data2)
                                    ax.set_title("Chi2: {}, dof: {}, chi2/dof: {}".format(np.round(np.sum((rec - data1)**2 / ((sigma+model_sys*abs(data1))**2)),2), np.round(dof,2), np.round(np.sum((rec - data1)**2 / ((sigma+model_sys*abs(data1))**2))/dof, 2)), fontsize = 30)
                                    ax.plot(time, rec, lw = 1, label = 'reconstruction')
                                    ax3 = fig.add_subplot(423)
                                    ax3.plot(time, (rec-data1)/(sigma + model_sys * abs(data1)), label = '(rec-data)/sigma')
                                    ax3.plot(time, (rec-data1), label = '(rec-data)')
                                    ax3.legend(fontsize = 30)
                                    ax3.axvline(time[np.argmax(((rec-data1)/(sigma + model_sys * abs(data1)))[0:400])])
                                    ax.axvline(time[np.argmax(((rec-data1)/(sigma + model_sys * abs(data1)))[0:400])])
                                    ax3.axvline(time[np.argmax(((rec-data1)/(sigma + model_sys * abs(data1))))])
                                    ax.axvline(time[np.argmax(((rec-data1)/(sigma + model_sys * abs(data1))))])
                                    ax.grid()
                                    ax.legend(fontsize = fontsize)
                                    ax3.grid()
                                    #ax3.set_xlim((11940, 11980))
                                    #aex.set_xlim((11940, 11980))
                                    ax1 = fig.add_subplot(422)
                                    ax1.hist((rec-data1)/(sigma + model_sys * abs(data1)), density = True)
                                    ax1.plot(np.arange(-5, 5, .1), stats.norm.pdf(np.arange(-5, 5, .1), 0, 1), color = 'red', label = 'standard normal')
                                
          #                      if channel.get_id() ==6:
           #                         reduced_chi2_Vpol = np.round(np.sum((rec - data1)**2 / ((sigma+model_sys*abs(data1))**2))/dof, 2)
                                if 0:#channel.get_id() == 13:
                                    ax = fig.add_subplot(425)
                                    ax.plot(data1)
                                    ax.plot(rec, lw = 1)
                                    ax1 = fig.add_subplot(426)
                                    ax1.hist((rec-data1)/(sigma/1.5 + model_sys * abs(data1)), density = True)
                                    ax1.plot(np.arange(-5, 5, .1), stats.norm.pdf(np.arange(-5, 5, .1), 0, 1), color = 'red', label = 'standard normal')
                                    ax1.legend()
                                    ax3 = fig.add_subplot(427)
                                    ax3.plot((rec-data1)/(sigma + model_sys * abs(data1)), label = '(rec-data)/sigma')
                                    ax3.legend(fontsize = fontsize)
                                    ax.set_title("Chi2: {}, dof: {}, chi2/dof: {}".format(np.round(np.sum((rec - data1)**2 / ((sigma+model_sys*abs(data1))**2)),2), np.round(len(time),2), np.round(np.sum((rec - data1)**2 / ((sigma+model_sys*abs(data1))**2))/len(time), 2)), fontsize = fontsize)#ax.set_title("Chi2: {}, dof {}, chi2/dof = {}:".format(np.round(np.sum((rec - data1)**2 / ((sigma+model_sys*abs(data1))**2), 2)), dof, np.round(np.sum((rec - data1)**2 / ((sigma+model_sys*abs(data1))**2))/dof, 2)))
                                    fig.tight_layout()
                                    #print(stop)
                                    fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/Uncertainties/1_direction_simulations/totaltrace.pdf")
        #                        
                                
                                     # if channel.get_id() ==13:
         #                           reduced_chi2_Hpol = np.round(np.sum((rec - data1)**2 / ((sigma+model_sys*abs(data1))**2))/len(time), 2)    			
                        else:
                                chi2 += chi2s[0] 
                                chi2 += chi2s[1]
                                dof += 2*len(rec_trace1)
                                all_chi2.append(chi2s[0])
                                all_chi2.append(chi2s[1])
            self.__dof = dof

            if timing_k:
                return ks
            if not minimize:
                #print("simtraces", simtraces)
                #print("simtraces_timing", simtraces_timing)
                #print("data timing 0", [data_timing[6][0][0], data_timing[6][0][-1]])
                #print("data timing 1", [data_timing[6][1][0], data_timing[6][1][-1]])
                return [rec_traces, data_traces, data_timing, all_chi2, [reduced_chi2_Vpol, reduced_chi2_Hpol]]
        #    print("CHI2 {}, dof {}, chi2/dof {}".format(chi2, dof, chi2/dof))
            return chi2 # - (dof - 3)/2 * np.log(2*np.pi*(sigma+self._model_sys*abs(data_trace_timing)**2))



    
        
        
    def end(self):
        pass
