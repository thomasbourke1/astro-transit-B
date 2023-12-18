#Necessary imports

from astropy.io import fits #used to open fits files in which data are stored
import glob

#SciPy functions:

from scipy.signal import find_peaks, lombscargle, medfilt
from scipy.optimize import curve_fit


#Math and Data Analysis:

import numpy as np
from numpy import pi
import pandas as pd
import math

#Visualisation:

import matplotlib.pyplot as plt 
from IPython.display import display, Math

#Transit finding:

import transit_finder as tf

#Physical constants and values

SUN_MASS = 1.989e30 # solar mass in kg
SUN_RADIUS = 6.957e8 #solar radius in metres
AU = 1.496e11 # earth - sun distance in metres
GRAVITATIONAL_CONSTANT = 6.672e-11
EARTH_MASS = 5.972e24 #Earth mass in kg
EARTH_RADIUS = 6378e3 #Earth radius in metres
G_EARTH = 9.81 #acceleration due to gravity at surface of earth
C_SOUND = 343 #speed of sound in air at 20 degrees celcius
T_EFF = 5756 #temperature of surface of sun

#All functions in this document were used for the purpose of discovering transit timing variations for each exoplanet planet

#####################################################################################################################
#Flux flipper
#####################################################################################################################

def flux_flipper(TRANSIT_FLUX): 
    
    """Function to invert transit flux
    Prepares for use of scipy.find_peaks

    Returns:
       FLIPPED_FLUX: Returns 1 / TRANSIT_FLUX
    """
       
    FLIPPED_FLUX = []
    for i in range(len(TRANSIT_FLUX)):
        TMP_FLIPPED = 1 / TRANSIT_FLUX[i]
        FLIPPED_FLUX.append(TMP_FLIPPED)
    
    
    return FLIPPED_FLUX

#####################################################################################################################
#Time shifter
#####################################################################################################################

def time_shifter(TIME):
    
    """Function to shift time data to start at zero
    - Useful in TTV analysis

    Returns:
        SHIFTED_TIME: Time data that is shifted to start at zero   
    """

    SHIFTED_TIME = []

    MIN_TIME = np.min(TIME)

    for i in range(len(TIME)):
        TMP_TIME = TIME[i] - MIN_TIME
        SHIFTED_TIME.append(TMP_TIME)
    
    return SHIFTED_TIME



#####################################################################################################################
#Peaks
#####################################################################################################################

def peaks(TIME, FLUX):
 
    """Function to analyse a given dataset and return the positions of peaks in the data
    
    Returns:
        PEAK_POSITIONS, PEAK_INDEXES: Time and index values for peaks in the data

    """
    THRESHOLD = 1.0005 #Transits above this threshold will be identified by scipy.find_peaks()
    
    PEAKS = find_peaks(FLUX, height=(THRESHOLD))
    
    TIME = np.linspace(np.min(TIME), np.max(TIME), len(TIME)) # generates linspace of time values
    
    HEIGHTS = PEAKS[1]['peak_heights']
    PEAK_INDEXES = PEAKS[0]
    PEAK_POSITIONS = TIME[PEAKS[0]]

    return PEAK_POSITIONS, PEAK_INDEXES, HEIGHTS, TIME

#####################################################################################################################
#Snapshot
#####################################################################################################################

def snapshot(PEAK_INDEXES, SNAPSHOT_SIZE, TIME, FLUX, FLUX_ERR):
    
    """Function takes snapshot of data either side of peak position

    Returns:
        SNAP_TIME, SNAP_FLUX, SNAP_FLUX_ERR: Sliced data values within given window
    """
    
    START_INDEX = max(0, PEAK_INDEXES - SNAPSHOT_SIZE)
    END_INDEX = min(len(TIME), PEAK_INDEXES + SNAPSHOT_SIZE + 1)
    
    SNAP_TIME = TIME[START_INDEX:END_INDEX]
    SNAP_FLUX = FLUX[START_INDEX:END_INDEX]
    SNAP_FLUX_ERR = FLUX_ERR[START_INDEX:END_INDEX]
    
    return SNAP_TIME, SNAP_FLUX, SNAP_FLUX_ERR

#####################################################################################################################
#Transit times -> Unsorted
#####################################################################################################################

def unsorted_transits(transit_time, transit_flux, transit_flux_err, PLOT=False):
    """Function to produce list of unsorted transit position from normalised Kepler data.

    Args:
        transit_time (_type_): Time data from Kepler
        transit_flux (_type_): Flux data from Kepler
        transit_flux_err (_type_): Flux error data from Kepler

    Returns:
        _type_: _description_
    """
    
    transit_flux_flipped = flux_flipper(transit_flux) #flips transit flux so it can be detected by find_peaks()
    #transit_time_shifted = time_shifter(transit_time) #shifts time to start from zero
    transit_time_shifted = transit_time
    
    THRESHOLD = 1.0005 # sets threshold for peak finding based on flux histogram
    
    PEAKS = find_peaks(transit_flux_flipped, height=(THRESHOLD))
    
    TIME = np.linspace(np.min(transit_time_shifted), np.max(transit_time_shifted), len(transit_time_shifted)) # generates linspace of time values
    
    HEIGHTS = PEAKS[1]['peak_heights']
    PEAK_INDEXES = PEAKS[0]
    PEAK_POSITIONS = TIME[PEAKS[0]]
    
    transit_positions = []
    transit_depths = []
    transit_durations = []

    for i in range(len(PEAK_INDEXES)):
        try:
            TMP_TIME, TMP_FLUX, TMP_FLUX_ERR = snapshot(PEAK_INDEXES[i], 20, TIME, transit_flux, transit_flux_err)
            
            xdata = np.linspace(min(TMP_TIME), max(TMP_TIME), len(TMP_TIME)) #x values for fitted function
            
            #initial guesss for parameters:
            mu = np.mean(TMP_TIME) #median of time value
            std = np.std(TMP_TIME) # standard deviation of flux distribution
            A = (std * np.sqrt(2 * pi)) * (1 - 0.9985)  #sets transit depth coefficient, modulated to estimate of flux value at bottom of transit
            
            #fit Gaussian model:
            popt, pcov = curve_fit(tf.lc_gaussian, TMP_TIME, TMP_FLUX, p0=[A, 1, mu, std], sigma=TMP_FLUX_ERR, maxfev=10000)
            parameters = popt
            parameters_error = np.sqrt(np.diag(pcov))
            
            #transit depth [relative flux]
            TMP_DEPTH = tf.transit_depth(parameters[0], parameters_error[0], parameters[3], parameters_error[3])[0]
            
            #transit duration [hours]
            TMP_DURATION = tf.theoretical_transit_time(13.024, 0, 0.1083, 0)* 24
            
            
            
            #filter out unphysical transits:
            if TMP_DEPTH < 0.0020 and TMP_DEPTH > 0:
                transit_positions.append(parameters[2])
                transit_depths.append(TMP_DEPTH)
                transit_durations.append
            else:
                PLOT=False
             
            #plot each individual transit   
            if PLOT:
                plt.figure(figsize=(10,7))
                plt.rcParams.update({'font.size': 12})
                plt.errorbar(TMP_TIME, TMP_FLUX, TMP_FLUX_ERR, ls='None', marker='o', label='Data')
                plt.xlabel("Time / days")
                plt.ylabel("Relative Flux")
                plt.title(f"Transit at peak position {PEAK_POSITIONS[i]}")
                
                #plot Gaussian:
                plt.plot(xdata, tf.lc_gaussian(xdata, popt[0], popt[1], popt[2], popt[3]), zorder=4, lw=2, color="r", label="Fitted Gaussian")
                
                plt.show()
            
        #exception in case curve_fit() can't optimise parameters
        except RuntimeError as e:
            print(f"Gaussian could not be fitted for transit {i+1} / {len(PEAK_INDEXES)}")
            
    #get rid of repeat measurements of same transit:
    new_transit_positions = []
    new_transit_depths = []
    tolerance = 1
    
    for i in range(1,len(transit_positions)):
        if abs(transit_positions[i] - transit_positions[i-1]) > tolerance :
            new_transit_positions.append(transit_positions[i])
            new_transit_depths.append(transit_depths[i])
            
    return new_transit_positions, transit_depths
            

#####################################################################################################################
#Transit depth histogram:
#####################################################################################################################

def depth_histogram(depths):

    #Mean depths calculated from folded lightcurve:
        MEAN_DEPTHS = [0.00077, 0.00097, 0.001482]
        MEAN_DEPTHS_ERR = [0.000022, 0.000032,  0.00004]
        colours = ['red', 'orange', 'blue']
        
        
        #Plot transit depth histogram:
        plt.figure(figsize=(10,7))
        plt.rcParams.update({'font.size': 12})
        plt.hist(depths, bins=75, color='blue', edgecolor='black')
    
        for i in range(len(MEAN_DEPTHS)):
            plt.vlines(MEAN_DEPTHS[i], 0, 40, colors=colours[i], label= f'Exoplanet {i+1} Mean Transit Depth')
            plt.vlines(MEAN_DEPTHS[i] - MEAN_DEPTHS_ERR[i], 0, 40, colors=colours[i], linestyles='dashed')
            plt.vlines(MEAN_DEPTHS[i] + MEAN_DEPTHS_ERR[i], 0, 40, colors=colours[i], linestyles='dashed')
            
        plt.xlabel('Transit Depth [relative flux]')
        plt.ylabel('Transit Depth count')
        plt.title("Transit Depth Histogram")
        plt.legend(loc= 2, fontsize = 9 )
        print()
        print(f"There are {len(depths)} transits detected.")
        
        return


#####################################################################################################################
#"Looking ahead"sorting
#####################################################################################################################

def looking_ahead(transit_positions, transit_depths, period):
    """Function to sort exoplanet transits by "looking ahead" in transit list by specific exoplanet period

    Args:
        transit_positions (_type_): Unsorted list of exoplanet transits
        transit_depths (_type_): Depths of unsorted exoplanet transits
        period (_type_): Ortbital period that function uses to look ahead

    Returns:
        ttvs, ttv_time, depths
    """

    #initialise arrays for transit timing variations, time at which they occur and their transit depths.
    ttvs = []
    depths = []
    ttv_time = []
    
    #assumes 
    PREVIOUS_TT = transit_positions[0]
    
    #tolerance window for "looking ahead" window
    tol = 1.5
    
    while PREVIOUS_TT < max(transit_positions):
        
        predicted_transit = PREVIOUS_TT + period #looks ahead
        
        #if in tolerance window -> appends transit 
        for i in range(len(transit_positions)):
            if predicted_transit - tol < transit_positions[i] < predicted_transit + tol:
                depths.append(transit_depths[i])
                ttvs.append(predicted_transit - transit_positions[i])
                ttv_time.append(transit_positions[i])
        
        PREVIOUS_TT = predicted_transit

    return ttvs, ttv_time, depths