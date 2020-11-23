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
        sampling_rate = station.get_channel(0).get_sampling_rate()
        self._sampling_rate = sampling_rate 
        simulated_energy = event.get_sim_shower(shower_id)[shp.energy]

        self._simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
        self._simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]
        vertex =event.get_sim_shower(shower_id)[shp.vertex] #station[stnp.nu_vertex]
        simulation = propagated_analytic_pulse.simulation(False, vertex)#event.get_sim_shower(shower_id)[shp.vertex])
        simulation.begin(det, station, use_channels, raytypesolution = 2)#[1, 2, 3] [direct, refracted, reflected]
        print("simulated zenith", np.rad2deg(self._simulated_zenith))
        print("simulatd azimuth", np.rad2deg(self._simulated_azimuth))        
        a, b, self._launch_vector_sim, c, d =  simulation.simulation(det, station, vertex[0],vertex[1], vertex[2], self._simulated_zenith, self._simulated_azimuth, simulated_energy, use_channels, first_iter = True)
        print("LAN VECTOR SIM", self._launch_vector_sim)
        print("viewing angle", np.rad2deg(c))
        #print(stop)
        self._simulation = simulation
        pass
    
    def run(self, event, shower_id, station, det,
            use_channels=[6, 14], filenumber = 1, debugplots_path = None, template = False):
        self._station = station
        self._use_channels = use_channels
        self._det = det
        
        
        sim_vertex = True
        if sim_vertex:
            vertex = event.get_sim_shower(shower_id)[shp.vertex]
        else:
            vertex = station[stnp.nu_vertex]
        simulation = propagated_analytic_pulse.simulation(template, vertex)#event.get_sim_shower(shower_id)[shp.vertex])
        simulation.begin(det, station, use_channels, raytypesolution = self._station[stnp.raytype])
        self._simulation = simulation
        simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]
        simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
        self._simulated_azimuth = simulated_azimuth
        simulated_energy = event.get_sim_shower(shower_id)[shp.energy]
### get raytype of maximum channel
        tracsim, timsim, lv_sim, vw_sim, raytypes_sim = simulation.simulation(det, station, event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], simulated_zenith, simulated_azimuth, simulated_energy, use_channels, first_iter = True) 
        max_trace = 0
       
    
        channl = station.get_channel(use_channels[0])
        n_samples = channl.get_number_of_samples()
        sampling_rate = channl.get_sampling_rate()
        self._sampling_rate = sampling_rate
        combined_fit = True #direction and energy are fitted simultaneously
        seperate_fit = False ##direction is first fitted and then values are used to fit energy
        if seperate_fit:
            fitprocedure = 'seperate'
        if combined_fit: ## Often a local minima is found
            fitprocedure = 'combined'            
        if combined_fit == seperate_fit:
            print(" WARNING: decision needs to be made what kind of fit is performed" )

        
    
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
    
        sigma = 0.0114
      
        
    
        station.set_is_neutrino()
        if station.has_sim_station():
            simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]   
            simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
            simulated_energy = event.get_sim_shower(shower_id)[shp.energy]
           
            print("energy", event.get_sim_shower(shower_id)[shp.energy])
            simulated_vertex = event.get_sim_shower(shower_id)[shp.vertex]
            if sim_vertex:
                reconstructed_vertex = simulated_vertex#station[stnp.nu_vertex]
            else: 
                reconstructed_vertex = station[stnp.nu_vertex]#event.get_sim_shower(shower_id)[shp.vertex]
 	
            print("simulated vertex position is", simulated_vertex)
            print("reconstructed vertex", reconstructed_vertex)
            SNR = []
            for ich, channel in enumerate(station.iter_channels()): ## checks SNR of channels
                Vrms = sigma#1.6257*10**(-5)
                print("channel {}, SNR {}".format(channel.get_id(),(abs(min(channel.get_trace())) + max(channel.get_trace())) / (2*Vrms) ))
                if channel.get_id() in use_channels:
                    Vrms = sigma#1.6257*10**(-5) # 16 micro volt
                    SNR.append((abs(abs(min(channel.get_trace()))) + max(channel.get_trace())) / (2*Vrms))
                    print("SNR", SNR)
					
                    
            global launch_vector_sim
            global traces_sim
            global timing_sim
            
           

        if 1:#min(SNR) > 4:
            uncertainties = 1 ## check influence of vertex position
            if uncertainties:
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
            new_vertex = reconstructed_vertex
        
            print("simulated vertex", simulated_vertex)
            print('new vertex', new_vertex)
           
            traces_sim, timing_sim, self._launch_vector_sim, viewingangles_sim, rayptypes = simulation.simulation( det, station, reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], simulated_zenith, simulated_azimuth, simulated_energy, use_channels, fit = 'combined', first_iter = True)
            options = {'maxiter':500, 'disp':True}
            print("Launch", self._launch_vector_sim)

            traces_sim, timing_sim, launch_vector_sim, viewingangles_sim, raytype= simulation.simulation( det, station, reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], simulated_zenith, simulated_azimuth, simulated_energy, use_channels, fit = 'combined', first_iter = True)
            
            tracsim = self.minimizer([simulated_zenith,simulated_azimuth, simulated_energy], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = fitprocedure, first_iter = True)[0]
          
            fsim = self.minimizer([simulated_zenith,simulated_azimuth, simulated_energy], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  True, fit = fitprocedure, first_iter = True)
            print("FSIM", fsim)
            tot_N = 80 * 4 * 2 * 2 # number of datapoints # samples * ray solutions * channels
            probability_sim = -1* (tot_N /2)* ( np.log(2*np.pi) +  np.log(sigma**2)) -fsim
            print("FMIN", fsim)
        
            #print(stop)
            
            
            trac = self.minimizer([simulated_zenith,simulated_azimuth, simulated_energy], reconstructed_vertex[0],reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, fit = fitprocedure, first_iter = True) ## this is to store the correct launch vector for the new vertex position 
           # print(stop)

          
          
            
            print("SIMULATED INPUT VALUES")
          
            print("fsim", fsim)
            
            
            if combined_fit:
              
                delta = np.deg2rad(6)
                zen_start = simulated_zenith - delta
                zen_end = simulated_zenith + delta
                az_start = simulated_azimuth - delta
                az_end = simulated_azimuth + delta
                
                
                banana = True ## takes like 15 min for 1 energy scan (for 2 channels!). If there is no good polarization measurement (low snr in Vpol or Hpol), then we can use the signal arrival direction as starting input. 
                if banana: ### fitting
                    
                    
                    #### define simulated viewing angl e
                    signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*launch_vector_sim)
                    print("signal zenith", np.rad2deg(signal_zenith))
                    print("signal azimuth", np.rad2deg(signal_azimuth))
                    sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
                   # rotation_matrix = hp.get_rotation(np.array([0,0,1]), sig_dir)
                    rotation_matrix = hp.get_rotation(np.array([0,0,1]), sig_dir)
                    z = simulated_zenith
                    a = simulated_azimuth
                    p2 = hp.spherical_to_cartesian(z, a)
                    rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))
                    p2 = rotation_matrix.dot(p2)
                    vector = hp.cartesian_to_spherical(p2[0], p2[1], p2[2])
                    print("delta", 180 - (np.rad2deg(vector[0])))
                    sim_view = 180 - (np.rad2deg(vector[0]))
                    
                    signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*launch_vector_sim)
                    print("signal zenith", np.rad2deg(signal_zenith))
                    print("signal azimuth", np.rad2deg(signal_azimuth))
                    print("SIM VIEW", sim_view)
                    sim_view = 56        
#            print(stop)
                    #sim_view = (75.15)
                    #theta = np.deg2rad(-98.4)
                    viewing_start = np.deg2rad(sim_view) - np.deg2rad(15) 
                    viewing_end = np.deg2rad(sim_view) + np.deg2rad(15)
                    theta_start = np.deg2rad(-180)
                    theta_end =  np.deg2rad(180)
                    print('launch vector sim', launch_vector_sim)
                    ### write vertex as function of azimuth and also change the azimuth angle
                    import datetime
                    cop = datetime.datetime.now()
                    print("SIMULATED DIRECTION {} {}".format(np.rad2deg(simulated_zenith), np.rad2deg(simulated_azimuth)))
                    
                    
                    
                    
                    """#### Test
                    
                    signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*launch_vector_sim)
                    sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
                    rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))
                    zn = []
                    az = []
                    for angle in np.arange(theta_start, theta_end, np.deg2rad(1)):
                        for cherenkov_angle in np.arange(np.deg2rad(sim_view - 5), np.deg2rad(sim_view + 5), np.deg2rad(1)):
                            p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                            p3 = rotation_matrix.dot(p3)
                            azimuth = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                            zenith = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                            zenith = np.deg2rad(180) - zenith
                            
                                #azimuth = np.deg2rad(360) - azimuth
                            zn.append(zenith)
                            az.append(azimuth)
                            print("zenith {} and azimuth {}".format(np.rad2deg(zenith), np.rad2deg(azimuth)))

                        
                    fit = plt.figure()
                    ax = fit.add_subplot(111)
                    ax.plot(np.rad2deg(az), np.rad2deg(zn), 'o')
                    ax.plot(np.rad2deg(simulated_azimuth), np.rad2deg(simulated_zenith), 'o')
                    fit.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/fig.pdf")
                    #print('banana')
                    print(stop)
                    """
                    
                    
                    results = opt.brute(self.minimizer, ranges=(slice(viewing_start, viewing_end, np.deg2rad(1)), slice(theta_start, theta_end, np.deg2rad(1)), slice(simulated_energy - simulated_energy/5 ,simulated_energy+2*simulated_energy/5, simulated_energy/10)), full_output = True, finish = opt.fmin , args = (new_vertex[0], new_vertex[1], new_vertex[2], True,fitprocedure, False, False, True, False))
                    print('start datetime', cop)
                    print("end datetime", datetime.datetime.now() - cop)
					
                else:
                    import datetime
                    cop = datetime.datetime.now()
                    results = opt.brute(minimizer, ranges=(slice(zen_start, zen_end, np.deg2rad(.5)), slice(az_start, az_end, np.deg2rad(.5)), slice(9*10**18, 11*10**18, 1e18)), args = (new_vertex[0], new_vertex[1], new_vertex[2], True,fitprocedure, False, False, False), full_output = False, finish = opt.fmin)
                    print('start datetime', cop)
                    print("DELTAdatetime", datetime.datetime.now() - cop)
                    print("no banana")
                
                if banana: ## convert reconstructed viewing angle and R to azimuth and zenith
                
           
                    if 1:
                        rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))
                        cherenkov_angle = results[0][0]
                        angle = results[0][1]
                  
                        p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                        p3 = rotation_matrix.dot(p3)
                        global_az = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                        global_zen = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]

                        global_zen = np.deg2rad(180) - global_zen
                        if np.rad2deg(simulated_azimuth) > 180:
                            if np.rad2deg(global_az) < 0:
                                global_az += np.deg2rad(180)
                            global_az += np.deg2rad(180)

                    rec_zenith = global_zen
                    rec_azimuth = global_az
                    rec_energy = results[0][2]
                    
                    #rec_zenith = simulated_zenith
                    #rec_azimuth = simulated_azimuth
                    #rec_energy = simulated_energy
                    print("reconstructed energy {}".format(rec_energy))
                    print("reconstructed zenith {} and reconstructed azimuth {}".format(np.rad2deg(rec_zenith), np.rad2deg(rec_azimuth)))
                    print("         simualted zenith {}".format(np.rad2deg(simulated_zenith)))
                    print("         simualted azimuth {}".format(np.rad2deg(simulated_azimuth)))
                    
                
                
                else:
                    global_zen = results[0][0]
                    global_az = results[0][1]
                    rec_zenith = global_zen
                    rec_azimuth = global_az
                    rec_energy = results[0][2]
                    #rec_zenith = simulated_zenith
                    #rec_azimuth = simulated_azimuth
                    #rec_energy = simulated_energy
                


                zvalues = []
                az = np.arange(az_start, az_end, np.deg2rad(.5))
                zen = np.arange(zen_start, zen_end,  np.deg2rad(.5))
               
                   
                
               
                azimuth = []
                zenith = []
                zplot = []
                zvalue_lowest = np.inf
                print("plotting output for reconstructecd energy  ...")
                ## determine zplot values for zenith and azimuths
                if banana:
                   
                    print('banana')
                else:
                    for a in az:
                        for z in zen:
                            zvalue = self.minimizer([z, a, rec_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize =  True, fit = fitprocedure)
                            zplot.append(zvalue)
                            azimuth.append(a)
                            zenith.append(z)
                            print("zenith {}, azimuth {}, zmin {}".format(np.rad2deg(z), np.rad2deg(a), np.rad2deg(zvalue)))
                            if zvalue < zvalue_lowest:
                                global_az = a
                                global_zen = z
                                zvalue_lowest = zvalue
                    


                
                """
                ### plot the likelihood landscape 
                fig = plt.figure()
                matplotlib.rc('xtick', labelsize = 10)
                matplotlib.rc('ytick', labelsize = 10)
                ###### PLOT SKYPLOT DIRECTION
                if banana: 
                    start = datetime.datetime.now()
                    ax =fig.add_subplot(111, projection = 'mollweide')

                
                    zenith = np.arange(np.deg2rad(-1), np.deg2rad(181), np.deg2rad(10))
                    azimuth = np.arange(np.deg2rad(-181), np.deg2rad(181), np.deg2rad(10))#.5
                    XX, YY = np.meshgrid(azimuth, zenith)
                    zplot = np.full(len(zenith)*len(azimuth), np.inf)
                    zplot_probability = np.full(len(zenith)*len(azimuth), np.inf)
                    k = 0
					
                   

					
                    zvalue_lowest = np.inf
                    l = 0 
                    for i in range(YY.shape[0]):
                        for j in range(YY.shape[1]):
                            z = YY[i][j]
                            a = XX[i][j]
                            
                         
                            p2 = hp.spherical_to_cartesian(z, a)
                            rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))

                            p2 = rotation_matrix.dot(p2)
                            vector = hp.cartesian_to_spherical(p2[0], p2[1], p2[2])
                            #print("delta", abs( (180 - (np.rad2deg(vector[0]))) - sim_view))
                            if (abs( (180 - (np.rad2deg(vector[0]))) - sim_view) < 7): ## 
                                zvalue = self.minimizer([z, a, rec_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize =  True, fit = fitprocedure, banana = False)
                               
                                tot_N = 80 * 4 * 2 * 2 # number of datapoints # samples * ray solutions * channels
                                probability = -1* (tot_N /2)* ( np.log(2*np.pi) +  np.log(sigma**2)) - zvalue
                               
                                zplot[k] = zvalue 
                                zplot_probability[k] = float(probability)
                            
                                if zvalue < zvalue_lowest:
                                    global_az = a
                                    global_zen = z
                                    zvalue_lowest = zvalue               

                           
                           
                            k += 1
                    print('start datetime', cop)
                    print("end datetime", datetime.datetime.now() -  start)
                    print("rec az", np.rad2deg(global_az))
                    print("rec zen", np.rad2deg(global_zen))
                                
                    ZZ = zplot.reshape(XX.shape)
                    ZZ_probability = zplot_probability.reshape(XX.shape)
                    
              
                
                    vmin = min(zplot)
                    vmax = max(np.array(zplot)[np.array(zplot) != np.inf])
                    mask = [np.array(ZZ) != np.inf]
                    
                    print("ZVALUE lowest", zvalue_lowest)
                    fsim_recenergy = self.minimizer([simulated_zenith,simulated_azimuth, rec_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = fitprocedure, first_iter = False)
                    print("fsim recenergy", fsim_recenergy)
                    
                    tot = 0
                    for i in ZZ[mask]:

                        if i < (fsim_recenergy+ 1):
                            tot += 1
                            print("i smaller", i)
                    print("Number of bins", tot)
                    Total_area = tot *.5*.5
                    print("TOTAL AREA LARGER THAN SIM VALUE", Total_area)
                            
               
                    XX -= (simulated_azimuth - np.deg2rad(180)) ## put simulated azimuth at 180 degrees
                    XX += np.deg2rad(180) ## shift such that they correspond to mollweide coordination system
                    
                    
                    
                    XX = get_normalized_angle(XX, interval = np.deg2rad([-180, 180]))
                    YY -= (simulated_zenith  - np.deg2rad(90)) ## put simulated azimuth at 90 degrees
                    
                   
                    YY = np.remainder(YY, np.deg2rad(180)) ## rotate values such that they are between 0 and 180
                   
                    YY -= np.deg2rad(90) ## hisft to mollweide projection
                   
                 
                    cs = ax.pcolor(XX, YY, ZZ, cmap = 'viridis', vmax = vmax, vmin = vmin) ## set vmax to max of zplot ## mollweide projection so put in radians 
                   

                    ZZ_new = np.empty(ZZ.shape)
                    XX_new = np.empty(ZZ.shape)
                    YY_new = np.empty(ZZ.shape)
                    XX_new = XX
                    YY_new = YY
                    for i in range(YY.shape[0]):
                        for j in range(YY.shape[1]):
                            if ZZ[i][j] < (fsim_recenergy +.1):
                                ZZ_new[i][j] = 1
                            else:
                                ZZ_new[i][j] = np.nan
                    
                  
                    mask1 = [np.array(ZZ_new) != 0]
                  
                    try:
                        ax.pcolor(XX_new, YY_new, ZZ_new, cmap = 'jet', vmin = .5, vmax = 3) ## mark all bins for which fit is better than simulated vlue  
                    except:
                        print("no values larger than the simulated value")
                        
                    xtick_labels = np.deg2rad([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]) 
                    xtick_labels += np.deg2rad(180) ## such that range is 0, 360
                    xtick_labels += (simulated_azimuth - np.deg2rad(180)) ## such that simulated azimuth is at the middle
                    
                    xtick_labels = np.remainder(xtick_labels, np.deg2rad(360)) ## such that we only have values between zero and 360
                    xtick_labels = [int(i) for i in np.rad2deg(xtick_labels)]

                    
                    ytick_labels = np.deg2rad([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]) 
                    ytick_labels += np.deg2rad(90)
                    ytick_labels += (simulated_zenith - np.deg2rad(90))
                    ytick_labels = np.remainder(ytick_labels, np.deg2rad(180))
                    ytick_labels = [int(i) for i in np.rad2deg(ytick_labels)]
               
                    
                    ### Plot zoom in
                    axins = ax.inset_axes([0, .8, .8, .8]) ## location of zoom in on plot
                    axins.pcolor(XX, YY, ZZ, cmap = 'viridis', vmax = vmax, vmin = vmin)
                  
                    try:
                        axins.pcolor(XX_new, YY_new, ZZ_new, cmap = 'jet', vmin = .5, vmax = 3)
                    except:
                        print("simualted is best value ")
                   
                    axins.set_xlim(-np.deg2rad(5), np.deg2rad(5)) ## in coordinate system of mollweide
                    axins.set_ylim(-np.deg2rad(5), np.deg2rad(5))
                   
                    axins.tick_params(labelsize = 5)
                    axins.grid(alpha = .5)

                    
                 
                    
                
                else:
                    ax = fig.add_subplot(111)
                    cs = ax.pcolor(np.rad2deg(np.array(azimuth)).reshape(len(az), len(zen)), np.rad2deg(np.array(zenith)).reshape(len(az), len(zen)), np.array(zplot).reshape(len(az), len(zen)), cmap = 'Greys_r')

                    levels = [min(zplot) + 0.5, min(zplot) + 2, min(zplot) + 4.5]

                    cs_contour = ax.contour(np.rad2deg(np.array(azimuth)).reshape(len(az), len(zen)), np.rad2deg(np.array(zenith)).reshape(len(az), len(zen)), np.array(zplot).reshape(len(az), len(zen)), [min(zplot) + 0.5, min(zplot) + 2, min(zplot) + 4.5], colors = ['green', 'yellow', 'red'] )

                
               
               
             
                ## the integral is given by https://stackoverflow.com/questions/22678990/how-can-i-calculate-the-area-within-a-contour-in-python-using-the-matplotlib
                
            
                
                try:
                    def area(vs):
                        a = 0 
                        x0, y0 = vs[0]
                        for [x1, y1] in vs[1:]:
                            dx = x1 - x0
                            dy = y1 - y0
                            print(" 0.5*(y0*dx - x0*dy)",  0.5*(y0*dx - x0*dy))
                            a += 0.5*(y0*dx - x0*dy)
                            x0 = x1
                            y0 = y1
                        return a 
                    area = []
                    for i in range(len(levels)):
                        contour = cs_contour.collections[i]
                        vs = contour.get_paths()[0].vertices
                        x = vs[:, 0]
                        y = vs[:, 1]
                        x = np.rad2deg(x)
                        y = np.rad2deg(y)
                        area1=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
                        area.append(np.abs(area1))
                        print("sigma = " + str(i) + ": area in degrees = " + str(area[i]))
                        


                    for i in range(len(labels)):
                        cs_contour.collections[i].set_label(labels[i])
                except: 
                    print("Area of sigma contours could not be determined")
                area.append(Total_area)
                

               
                
                if banana:
                    
                    
                    ax.axvline(mollweide_azimuth(rec_azimuth), label = 'reconstructed fit', color = 'darkblue',linewidth = 2)
                    ax.axhline(mollweide_zenith(rec_zenith),  color = 'darkblue',linewidth = 2)
                    mollweide_az = mollweide_azimuth(simulated_azimuth) ## this should turn out to be zero
                    mollweide_zen = mollweide_zenith(simulated_zenith) ## this should turn out to be zero
                    print("mollweide zen simulated", np.rad2deg(mollweide_zen))
                    print("mollweide az simulated", np.rad2deg(mollweide_az))
                    ax.axvline(mollweide_az, label = 'simulated direction', color = 'orange', linewidth = 1)
                    ax.axhline(mollweide_zen, color = 'orange',linewidth = 2)
                    ax.axvline(mollweide_azimuth(global_az), color = 'lightblue', ls = '--', linewidth = 2)
                    ax.axhline(mollweide_zenith(global_zen), label = 'reconstructed global', color = 'lightblue', ls = '--', linewidth = 2)
                    ax.set_xlabel("azimuth [degrees]")
                    ax.set_ylabel("zenith [degrees]")
                   #ax.set_title("reconstructed energy and varying azimuth and zenith")
                
                    axins.axvline(mollweide_azimuth(rec_azimuth), label = 'reconstructed fit', color = 'darkblue',linewidth = 2)
                    axins.axhline(mollweide_zenith(rec_zenith),  color = 'darkblue',linewidth = 2)
                    mollweide_az = mollweide_azimuth(simulated_azimuth)
                    axins.axvline(mollweide_az, label = 'simulated direction', color = 'orange', linewidth = 2)
                    axins.axhline(mollweide_zen, color = 'orange',linewidth = 2)
                    axins.axvline(mollweide_azimuth(global_az), color = 'lightblue', ls = '--', linewidth = 2)
                    axins.axhline(mollweide_zenith(global_zen), label = 'reconstructed global', color = 'lightblue', ls = '--', linewidth = 2)
                
                    ax.indicate_inset_zoom(axins)
                else:
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
                if banana:
                   
                    ax.legend(loc = 9, ncol =2, bbox_to_anchor = (0.5, 0.45), fontsize = 'small')
                    ax.set_xticklabels(xtick_labels)
                    ax.set_yticklabels(ytick_labels)
                    ax.tick_params(labelsize = 4)
                    print('banana')
                else:
                    minor_ticksx = np.round(np.arange(min(np.rad2deg(azimuth)), max(np.rad2deg(azimuth)), .5), 1)
                    minor_ticksy = np.round(np.arange(min(np.rad2deg(zenith)), max(np.rad2deg(zenith)), .5), 1)
                    
                    ax.set_xticks(minor_ticksx)
                    ax.set_yticks(minor_ticksy)
                ax.grid(alpha = .5)
                ax.set_title("Area = {}".format(Total_area))

                cbar = fig.colorbar(cs, shrink = 0.5)
                cbar.set_label('-1 * log(L)', rotation=270)
                fig.tight_layout()
                
               # fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/Uncertainties/plots/Alvarez/direction_{}_{}.pdf".format(filenumber, shower_id))
                """ 
                
                print("     reconstructed zenith = {}".format(np.rad2deg(rec_zenith)))
                print("     reconstructed azimuth = {}".format(np.rad2deg(rec_azimuth)))
                
                print("###### seperate fit reconstructed valus")
                print("         zenith = {}".format(np.rad2deg(rec_zenith)))
                print("         azimuth = {}".format(np.rad2deg(rec_azimuth)))
                print("        energy = {}".format(rec_energy))
                print("         simualted zenith {}".format(np.rad2deg(simulated_zenith)))
                print("         simualted azimuth {}".format(np.rad2deg(simulated_azimuth)))
                print("         simulated energy {}".format(simulated_energy)) 
            
            
            
        
            
            
            
            
            print("RECONSTRUCTED DIRECTION ZENITH {} AZIMUTH {}".format(np.rad2deg(rec_zenith), np.rad2deg(rec_azimuth)))
            print("RECONSTRUCTED ENERGY", rec_energy)
            print("RECONSTRUCTED DIRECTION ZENITH {} AZIMUTH {} GLOBAL".format(np.rad2deg(global_zen), np.rad2deg(global_az)))

            
            ## get the traces for the reconstructed energy and direction
            tracrec = self.minimizer([rec_zenith, rec_azimuth, rec_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize = False, fit = 'combined')[0]
            
            ## get the min likelihood value for the simulated values
            fminsim = self.minimizer([simulated_zenith, simulated_azimuth, simulated_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = 'combined', first_iter = False)
           
            fminrec = self.minimizer([global_zen, global_az, rec_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize =  True, fit = 'combined')
            
            
            fminfit = self.minimizer([rec_zenith, rec_azimuth, rec_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize =  True, fit = 'combined')
            
            print("FMIN SIMULATED VALUE", fminsim)
            print("FMIN RECONSTRUCTED VALUE GLOBAL", fminrec)
            print("FMIN RECONSTRUCTED VALUE FIT", fminfit)
            print("FMIN SIMULATED DIreCtioN, RECONStrUCteD ENERGY", fsim_recenergy)
            
            
            station.set_parameter(stnp.nu_zenith, rec_zenith)
            station.set_parameter(stnp.nu_azimuth, rec_azimuth)
            station.set_parameter(stnp.nu_energy, rec_energy)
            station.set_parameter(stnp.chi2_efield_time_direction_fit, [fminsim, fminfit])
            
          
            debug_plot = 1
            if debug_plot:
                
                tracglobal = self.minimizer([global_zen, global_az, rec_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize = False, fit = 'combined')
                tracglobal = tracglobal[0]
                
                tracdata = self.minimizer([rec_zenith, rec_azimuth, rec_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize = False, fit = fitprocedure)[1]
             
                timingdata = self.minimizer([rec_zenith, rec_azimuth, rec_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize = False, fit = fitprocedure)[2]
                timingsim = self.minimizer([simulated_zenith, simulated_azimuth, simulated_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize = False, fit = fitprocedure)[2]
                tracsim = self.minimizer([simulated_zenith, simulated_azimuth, simulated_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize = False, fit = fitprocedure)[0]

                
                
                tracglobal_k = self.minimizer([rec_zenith, rec_azimuth, rec_energy], new_vertex[0], new_vertex[1], new_vertex[2], minimize = False, fit = fitprocedure, timing_k = True)
                
                fig, ax = plt.subplots(len(use_channels), 3, sharex=False, figsize=(40, 20))
                matplotlib.rc('xtick', labelsize = 30)
                matplotlib.rc('ytick', labelsize = 30)
                ich = 0
                TOT = 0
                sigma = 0.0114
                SNRs = np.zeros((len(use_channels), 2))
                for channel in station.iter_channels():
                    if channel.get_id() in use_channels: # use channels needs to be sorted
                        
                        if len(tracdata[channel.get_id()]) > 0:
                            SNR = abs(max(tracdata[channel.get_id()][0]) - min(tracdata[channel.get_id()][0])) / (2*sigma)
                            print("SNR", SNR)
                            #print("LEN DATA", len(tracdata[channel.get_id()][0]))
                            SNRs[ich, 0] = SNR
                            #ax[ich][0].plot(channel.get_trace(), label = 'data', color = 'black')
                            ax[ich][0].plot(timingdata[channel.get_id()][0], tracdata[channel.get_id()][0], label = 'data', color = 'black')
                            ax[ich][0].fill_between(timingsim[channel.get_id()][0],tracsim[channel.get_id()][0]- sigma, tracsim[channel.get_id()][0] + sigma, color = 'red', alpha = 0.2 )
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq( tracdata[channel.get_id()][0], sampling_rate)), color = 'black')
                            ax[ich][0].plot(timingsim[channel.get_id()][0], tracsim[channel.get_id()][0], label = 'simulation', color = 'orange')
                            if channel.get_id() in [6]:#,7,8,9,13,14]: 
                                 ax[ich][0].plot(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0], label = 'reconstruction', color = 'green')
                            elif SNR > 3.5:
                                 ax[ich][0].plot(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0], label = 'reconstruction', color = 'green')


                            ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][0], sampling_rate)), color = 'orange')
                            if channel.get_id() in [6]:
                                 ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][0], sampling_rate)), color = 'green')
                         
                            elif SNR > 3.5:
                                 ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][0], sampling_rate)), color = 'green')

                            ax[ich][0].legend(fontsize = 'x-large')
                           
                        if len(tracdata[channel.get_id()]) > 1:
                            SNR = abs(max(tracdata[channel.get_id()][1]) - min(tracdata[channel.get_id()][1])) / (2*sigma)
                            print("SNR", SNR)
                            SNRs[ich, 1] = SNR
                            ax[ich][1].plot(timingdata[channel.get_id()][1], tracdata[channel.get_id()][1], label = 'data', color = 'black')
                            ax[ich][1].fill_between(timingsim[channel.get_id()][1],tracsim[channel.get_id()][1]- sigma, tracsim[channel.get_id()][1] + sigma, color = 'red', alpha = 0.2 )

                            ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracdata[channel.get_id()][1], sampling_rate)), color = 'black')
                            ax[ich][1].plot(timingsim[channel.get_id()][1], tracsim[channel.get_id()][1], label = 'simulation', color = 'orange')
                            if channel.get_id() in [6]:#,7,8,9]: 
                                ax[ich][1].plot(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1], label = 'reconstruction', color = 'green')
                            elif SNR > 3.5:
                                ax[ich][1].plot(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1], label = 'reconstruction', color = 'green')


                            ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][1], sampling_rate)), color = 'orange')
                            if channel.get_id() in [6]:
                                 ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][1], sampling_rate)), color = 'green')
                            elif SNR > 3.5:
                                 ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][1], sampling_rate)), color = 'green')

                            
                        
                        
                        ich += 1
                fig.tight_layout()
                fig.savefig("{}/fit_{}_{}.pdf".format(debugplots_path, filenumber, shower_id))
                SNRsfit, viewinganglesfit = self.minimizer([rec_zenith, rec_azimuth, rec_energy], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = True, fit = fitprocedure, return_pulse_parameters = True)
                SNRssim, viewinganglessim = self.minimizer([simulated_zenith,simulated_azimuth, simulated_energy], simulated_vertex[0], simulated_vertex[1], simulated_vertex[2], minimize =  True, fit = fitprocedure, return_pulse_parameters = True, first_iter = True)
                #station.set_parameter(stnp.SNR, [SNRssim, SNRsfit])
                station.set_parameter(stnp.viewingangles, [viewinganglessim, viewinganglesfit])


    
                  
    def minimizer(self, params, vertex_x, vertex_y, vertex_z, minimize = True, fit = 'seperate', timing_k = False, first_iter = False, banana = False, return_pulse_parameters = False, direction = [0, 0]):
            import datetime
            start1 = datetime.datetime.now()
            
            sigma = 0.0114 #noise Rms for with amplifier #1.7*10**(-5) * 10000 
       
            if len(params) == 2: ##hadronic shower direction fit
                if fit == 'seperate':
                    zenith, azimuth = params
                
                if fit == 'combined':
                    zenith, azimuth = params
                    energy = rec_energy
            if len(params) == 1: ##hadronic shower energy fit with input fitted direction
                energy = params[0]
                zenith, azimuth = direction
            if fit == 'seperate':
                if len(params) == 3: ## hadronic or electromagnetic total fit
                    zenith, azimuth, energy = params
            if fit == 'combined':
                if len(params) == 3: ## hadronic or electromagnetic total fit
                    zenith, azimuth, energy = params
                    
            ###### Get channel with maximum amplitude to find reference timing
            maximum_trace1 = 0
            for ich, channel in enumerate(self._station.iter_channels()):
                if channel.get_id() in self._use_channels:
                    maximum_trace = max(abs(np.copy(channel.get_trace())))
                    if maximum_trace > maximum_trace1:
                        maxchannelid = channel.get_id()
                        maximum_trace1 = maximum_trace
            
            
      
            
            if banana: ## input values accoring to launch vector
                
                
                signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*self._launch_vector_sim)
                
                sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
                rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))

                cherenkov_angle = params[0]
                angle = params[1]
              
                
                
                p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                p3 = rotation_matrix.dot(p3)
                azimuth = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                zenith = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                zenith = np.deg2rad(180) - zenith
                if np.rad2deg(self._simulated_azimuth) > 180:
                        #print("zen", np.rad2deg(zenith))
                    if np.rad2deg(azimuth) < 0:
                        azimuth += np.deg2rad(180)
                    azimuth += np.deg2rad(180)
                #azimuth  = get_normalized_angle(azimuth)
                
                
            if np.rad2deg(zenith) > 100:
                return np.inf ## not in field of view
            
        
            traces, timing, launch_vector, viewingangles, raytypes = self._simulation.simulation(self._det, self._station, vertex_x, vertex_y, vertex_z, zenith, azimuth, energy, self._use_channels, fit, first_iter = first_iter)
            chi2 = 0
            dof = -3


            rec_traces = {}
            data_traces = {}
            data_timing = {}
            normalization_factors = {} ## dictionary to store the normalzation factors


           
            data_trace = np.copy(self._station.get_channel(maxchannelid).get_trace())
            max_trace = 0
            rec_trace = np.zeros(len(data_trace))## if for the simulated trace there is no solution
           
       
            #get timing for raytype of triggered pulse
            for iS in raytypes[6]:
                if raytypes[6][iS] == ['direct', 'refracted', 'reflected'].index(self._station[stnp.raytype]):
                    solution_number = iS
            T_ref = timing[6][solution_number]
           
            k_ref = self._station[stnp.pulse_position]# np.argmax(abs(data_trace))## assume here is the pulse
            
            ks = {}
            SNRs = np.zeros((len(self._use_channels), 2))

            ich = -1
           
            for channel in self._station.iter_channels():
                if channel.get_id() in self._use_channels: #iterate over channels
                   
                    ich += 1 ## number of channel
                    data_trace = np.copy(channel.get_trace())
                    rec_traces[channel.get_id()] = {}
                    data_traces[channel.get_id()] = {}
                    data_timing[channel.get_id()] = {}
                    
                    max_trace = 0
                    ### if no solution exist, than analytic voltage is zero
                    rec_trace = np.zeros(len(data_trace))# if there is no raytracing solution, the trace is only zeros

                    delta_k = [] ## if no solution type exist then channel is not included
                    num = 0
                    for i_trace, key in enumerate(traces[channel.get_id()]): ## iterate over ray type solutions
                        rec_trace_i = traces[channel.get_id()][key]
                        rec_trace = rec_trace_i
                        
                        max_trace = max(abs(rec_trace_i))
                        delta_T =  timing[channel.get_id()][key] - T_ref
                  
                        ## before correlating, set values around maximum voltage trace data to zero
                        delta_toffset = delta_T * self._sampling_rate
                   
                        ### figuring out the time offset for specfic trace
                        dk = int(k_ref + delta_toffset )
                        rec_trace1 = rec_trace
                     
                        ### now correlate rectrace1 with the cut data, and look for the offset
                        ## cut data:
                        ARZ =1
                        if 1:#channel.get_id() == 9:#ARZ:#(channel.get_id() == 10 and i_trace == 1):
                            data_trace_timing = np.copy(data_trace) ## cut data around timing
							## DETERMIINE PULSE REGION DUE TO REFERENCE TIMING 

                           
                                        
                            data_timing_timing = np.arange(0, len(data_trace_timing), 1)
                            data_timing_timing = data_timing_timing[dk - 300 : dk + 500]
                            data_trace_timing = data_trace_timing[dk -300 : dk + 500]
                            data_trace_timing_1 = np.copy(data_trace_timing)
                            ### cut data trace timing to make window to search for pulse smaller
                            data_trace_timing_1[data_timing_timing < (dk - 150)] = 0
                            data_trace_timing_1[data_timing_timing > (dk + 250)] = 0
                        if 1:
                          
                            try: 
                                corr = signal.hilbert(signal.correlate(rec_trace1, data_trace_timing_1))#signal.hilbert(signal.correlate(data_trace_timing, rec_trace_timing)) ## correlate ## they do not have to be the same length  
                            except: 
                                corr = signal.hilbert(signal.correlate(rec_trace1, np.zeros(len(rec_trace1))))### if data trace_timing does not exist because mamximum is wrongly defined
                                data_trace_timing = np.zeros(len(rec_trace1))
                            if channel.get_id() == 6:
                                if i_trace == 0:
                                    dt_1 = np.argmax(corr) - (len(corr)/2) +1 ## find offset
                                if i_trace ==1:
                                    dt_2 = np.argmax(corr) - (len(corr)/2) + 1
                        dt = np.argmax(corr) - (len(corr)/2) + 1
                        if 1:#channel.get_id() not in [6,7,8,9,13,14]:
                            
                            rec_trace1 = np.roll(rec_trace1, math.ceil(-1*dt))
                           
                            rec_trace1 = rec_trace1[150 : 550] 
                            data_trace_timing = data_trace_timing[150:  550]
                            data_timing_timing = data_timing_timing[150:  550]
			
    
                             
                        delta_k.append(int(k_ref + delta_toffset + dt )) ## for overlapping pulses this does not work
                        ks[channel.get_id()] = delta_k

                     #   print("rec traces",rec_trace1)
                        if fit == 'combined':
                          
                            rec_traces[channel.get_id()][i_trace] = rec_trace1
                            data_traces[channel.get_id()][i_trace] = data_trace_timing
                            data_timing[channel.get_id()][i_trace] = data_timing_timing
                         
                            SNR = abs(max(data_trace_timing) - min(data_trace_timing) ) / (2*sigma)
                          
                            SNRs[ich, i_trace] = SNR
                   #        
                            if (int(delta_T) == 0):#,7,8,9,13,14]:
                                dof += len(rec_trace1)
                        
                                chi2 += np.sum((rec_trace1 - data_trace_timing)**2 / (2*sigma**2))
                            elif (SNR > 3.5):
                                dof += len(rec_trace1)

                                #print("use channel", channel.get_id())
                                chi2 += np.sum((rec_trace1 - data_trace_timing)**2 / (2*sigma**2))
                             
            
            if timing_k:
                return ks
            if not minimize:
                return [rec_traces, data_traces, data_timing]
            if return_pulse_parameters:
                return SNRs, viewingangles 
            return chi2/ dof



    
        
        
    def end(self):
        pass
