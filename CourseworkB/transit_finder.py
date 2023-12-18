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
import matplotlib.patheffects as path_effects

#my files
import transit_timing_variations as ttv
import masses as ma

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

    
#####################################################################################################################
#Import telescope data from .lc files and clean, normalise, filter outliers
#####################################################################################################################


def exoplanet_data_cleaner(PLOT = False):
    
    """ exoplanet_data_cleaner() masks to remove NaNs, normalises with a median filter, and removes data above 3 standard deviations of the mean    
    - Output can be subsequently used to plot periodogram

    Returns:
        TRANSIT_TIMES, TRANSIT_FLUXES, TRANSIT_FLUXES_ERR: reduced data from Kepler telescope
    """
   
    #create empty arrays that temp arrays will append to
    
    TRANSIT_TIMES = []
    TRANSIT_FLUXES = []
    TRANSIT_FLUXES_ERR = []
    
    #open each .fits file, cleans data then appends to transit data set

    for lcfile in glob.glob('Data/Objectlc/kplr*.fits'):
        
        #create temporary arrays for Kepler data that will be overwitten for each file
        TEMP = fits.open(lcfile)
        TEMP_TIME = (TEMP[1].data['TIME'])
        TEMP_FLUX = (TEMP[1].data['PDCSAP_FLUX'])
        TEMP_FLUX_ERR = (TEMP[1].data['PDCSAP_FLUX_ERR'])
        
        #remove NaNs from data using Boolean indexing:
        indices = np.logical_not(np.isnan(TEMP_FLUX))
        TEMP_TIME = TEMP_TIME[indices]
        TEMP_FLUX = TEMP_FLUX[indices]
        TEMP_FLUX_ERR = TEMP_FLUX_ERR[indices]
        
        #normalise data by dividing each flux, flux error value by median -> uses medfilt
        MEDIAN_FLUX = medfilt(TEMP_FLUX, kernel_size=31) ##kernel_size is the size of the window over which the median is calculated 
        TEMP_FLUX = TEMP_FLUX / MEDIAN_FLUX
        TEMP_FLUX_ERR = TEMP_FLUX_ERR / MEDIAN_FLUX
        
        #filter upper range of flux to within 3 standard deviations of mean flux value
        
        FLUX_STANDEV = np.std(TEMP_FLUX) #finds standard deviation of flux
    
        i=0
        for i in range(len(TEMP_FLUX)):
            if TEMP_FLUX[i] < (1 + (3*FLUX_STANDEV)):
                TRANSIT_TIMES.append(TEMP_TIME[i])
                TRANSIT_FLUXES.append(TEMP_FLUX[i])
                TRANSIT_FLUXES_ERR.append(TEMP_FLUX_ERR[i]) 
                
    if PLOT:
        plt.figure(figsize=(100,10))
        plt.rcParams.update({'font.size': 60})
        plt.errorbar(TRANSIT_TIMES, TRANSIT_FLUXES, TRANSIT_FLUXES_ERR, ls='None', marker='o', label='Data')
        plt.xlabel("Time / days")
        plt.ylabel("Relative Flux")
        plt.title('Total Lightcurve for Kepler Space Telescope Data')

        return
    else:
        return TRANSIT_TIMES, TRANSIT_FLUXES, TRANSIT_FLUXES_ERR
                
#####################################################################################################################
#Vicky Scowcroft's light curve folding function
#####################################################################################################################

def fold_lightcurve(time, flux, error, period): 
    """
    Folds the lightcurve given a period.
    time: input time (same unit as period)
    flux: input flux
    error: input error
    period: period to be folded to, needs to same unit as time (i.e. days)
    
    Returns:
        phase, folded flux, folded error: Phase, flux and error for folded lightcurve
    """
    #Create a pandats dataframe from the 
    data = pd.DataFrame({'time': time, 'flux': flux, 'error': error})
    
    #create the phase 
    data['phase'] = data.apply(lambda x: ((x.time/ period) - np.floor(x.time / period)), axis=1)
    
    #Creates the out phase, flux and error
    phase_long = np.concatenate((data['phase'], data['phase'] + 1.0, data['phase'] + 2.0))
    flux_long = np.concatenate((flux, flux, flux))
    err_long = np.concatenate((error, error, error))
    
    return(phase_long, flux_long, err_long)

#####################################################################################################################
#Periodogram analysis
#####################################################################################################################

def periodogram(TRANSIT_TIMES, TRANSIT_FLUXES, PERIOD_RANGE):
    
    """creates periodogram for a reduced lightcurve
    
    Returns:
        POTENTIAL_EXOPLANET_PERIODS: Array of potential exoplanet periods. Contains data for repeated transits, needs further processing to extract "real" periods
    """
    
    print("Takes 1m 36s to run natively on my personal computer (M2 Macbook pro), may take longer on Noteable server.")
    print()
    
    #creates initial lombscargle of planetary periods
    TIME_DIFF = np.mean(np.diff(TRANSIT_TIMES)) # average difference between TRANSIT_TIMES values
    T = max(TRANSIT_TIMES) - min(TRANSIT_TIMES)
    FREQUENCIES = np.linspace(1/TIME_DIFF, 1/T, len(TRANSIT_TIMES))
    LOMB = lombscargle(TRANSIT_TIMES, TRANSIT_FLUXES, FREQUENCIES, precenter=True)
    
    #carries out second lombscargle to remove aliasing from periodogram, transforms back to time domain
    PERIODS = np.linspace(1, PERIOD_RANGE, 7000)
    LOMB_2 = lombscargle(FREQUENCIES, LOMB, PERIODS, precenter=True)
    
    #uses SciPy.find_peaks() to identify spikes on periodogram
    PEAKS = find_peaks(LOMB_2, height=(0.5e-12)) #Height chosen by looking at periodogram
    HEIGHTS = PEAKS[1]['peak_heights'] #list of the heights of the peaks
    POTENTIAL_EXOPLANET_PERIODS = PERIODS[PEAKS[0]] #list of the peaks positions
    
    #Plots periodogram
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 12})
    plt.plot(PERIODS, LOMB_2, zorder=0)
    plt.scatter(POTENTIAL_EXOPLANET_PERIODS, HEIGHTS, color = 'r', marker = 'D', label = 'Maxima')
    plt.xlabel('Period [d]')
    plt.ylabel('Power')
    plt.title('Periodogram for Kepler Space Telescope data')
    print("Potential exoplanet periods = ", POTENTIAL_EXOPLANET_PERIODS, "days")
    
    return(POTENTIAL_EXOPLANET_PERIODS)

#####################################################################################################################
#Extract real planetary periods from periodogram
#####################################################################################################################

"""
period_finder returns a list of real exoplanet periods ± uncertainty
If there is a exoplanet period of 5 days, a "period" will also be measured at 10 days, 15 days etc.
period_finder sums up all the periods in the periodogram, finds the average
The error is taken as the standard deviation in the list of planetary periods
"""

def gen_multiple_list(x, lim):
    """
    Function that generates integer multiples of x up to a limit 'lim'
    Used to compute multiples of the intial planetary period
    - Similar functionality to "enumertate"
    """
    n = 2
    while x * n < lim:
        yield x * n
        n += 1

def period_finder(POTENTIAL_EXOPLANET_PERIODS, TOLEREANCE=0.1):
    
    """Function that finds list of real exoplanet periods from output of periodogram()

    Returns:
        TRUE_PERIODS, TRUE_PERIODS_ERR: Returns exoplanet periods ± error
    """
    
    TEMP_PERIODS = list(POTENTIAL_EXOPLANET_PERIODS)
    lim = max(TEMP_PERIODS) #upper limit to which multiples of period are generated
    i = 0
    SORTED_PERIODS = {} 
    while i < len(TEMP_PERIODS): #for each element in POTENTIAL_EXOPLANET_PERIODS
        SORTED_PERIODS[TEMP_PERIODS[i]] = []
        j = i + 1
        while j < len(TEMP_PERIODS): #compares multiple to every other element in list above it
            increment = True
            for multiple in gen_multiple_list(TEMP_PERIODS[i], lim):
                if multiple + TOLEREANCE > TEMP_PERIODS[j] and multiple - TOLEREANCE < TEMP_PERIODS[j]:
                    SORTED_PERIODS[TEMP_PERIODS[i]].append(TEMP_PERIODS.pop(j)) #removes period element if in TOLEREANCE range, appends to SORTED_PERIODS{}
                    increment = False
                    break
            if increment:
                j += 1
        i += 1
        
    ACTUAL_PERIODS = []
    MULTIPLE_PERIODS = []

    for p1, p2 in SORTED_PERIODS.items(): #appends SORTED_PERIODS{} into seperate arrays
        ACTUAL_PERIODS.append(p1)
        MULTIPLE_PERIODS.append(p2)
    
    TRUE_PERIODS = [] #creates arrays in which true values are stored
    TRUE_PERIODS_ERR = []
    
    for k in range(len(ACTUAL_PERIODS)):
        TEMP_PERIODS = SORTED_PERIODS[ACTUAL_PERIODS[k]]
        dp = np.diff(TEMP_PERIODS) #interval between planetary transits
        TEMP_PERIOD = np.mean(dp) #takes mean of intervals between planetary transits
        TEMP_PERIOD_ERR = np.std(dp) #takes standard deviation of intervals between planetary transits
        if TEMP_PERIOD_ERR == 0:
            TEMP_PERIOD_ERR = TOLEREANCE # if error comes out as zero, sets it to TOLEREANCE value
        TRUE_PERIODS.append(TEMP_PERIOD)  
        TRUE_PERIODS_ERR.append(TEMP_PERIOD_ERR)  
        
        
    #mask to remove NaN periods
    TRUE_PERIODS = np.array(TRUE_PERIODS) 
    TRUE_PERIODS_ERR = np.array(TRUE_PERIODS_ERR)
    indices = np.logical_not(np.isnan(TRUE_PERIODS))
    TRUE_PERIODS = TRUE_PERIODS[indices]
    TRUE_PERIODS_ERR = TRUE_PERIODS_ERR[indices]

    return TRUE_PERIODS, TRUE_PERIODS_ERR

#####################################################################################################################
#multiple light curve plotting
#####################################################################################################################


def multiple_lc_plotter(TRUE_PERIODS, TRUE_PERIODS_ERR, STEPSIZE, transit_time, transit_flux, transit_fluxerr):
    
    """Function that plots multiple graphs per exoplanet period, iterating over uncertainty in period
    Takes outputs from period_finder as inputs
    Generated graphs will have to be eyeballed in order to judge which period value folds to the best light curve
    """
    
    print("Warning: Run just once to eyeball periods that fold to clearest lightcurves! ")
    print()
    print(f"Will plot 3 * {STEPSIZE} graphs")
    print()
    print(f"Takes 32s to create 30 graphs on personal computer (M2 Macbook pro), may take longer on Notable server.")
        
    for i in range(len(TRUE_PERIODS)):
        
        MINIMUM_PERIOD = TRUE_PERIODS[i] - TRUE_PERIODS_ERR[0]
        MAXIMUM_PERIOD = TRUE_PERIODS[i] + TRUE_PERIODS_ERR[i]


        PERIOD_RANGE = np.linspace(MINIMUM_PERIOD, MAXIMUM_PERIOD, STEPSIZE)

        for j in range(len(PERIOD_RANGE)):
            PHASE, FLUX, FLUXERR = fold_lightcurve(transit_time, transit_flux, transit_fluxerr, PERIOD_RANGE[j])

            #phase diagram of transit:
            
            #use scatter not errorbar to speed up computational time
            
            
            plt.figure(figsize=(10,5))
            plt.rcParams.update({'font.size': 12})
            plt.scatter(PHASE, FLUX, marker='o', ls='None', zorder=4, label='_nolegend_', s=0.1)
            plt.xlabel('Phase')
            plt.ylabel('Flux')
            plt.xlim(0, 1)
            print(f"Period = {PERIOD_RANGE[j]}")
            plt.show()
            
    return()

#####################################################################################################################
#slice arrays
#####################################################################################################################

def array_slicer(TIME, FLUX, FLUX_ERR, START_TIME, STOP_TIME):
    
    """Function that slices arrays from START_TIME to STOP_TIME
    
    - Used for fitting model to data in specific period

    Returns:
        NEW_TIME, NEW_FLUX, NEW_FLUX_ERR: Sliced arrays
    """
    
    NEW_TIME = []
    NEW_FLUX = []
    NEW_FLUX_ERR = []
    
    for i in range(len(TIME)):
        if TIME[i] > START_TIME and TIME[i] < STOP_TIME:
            NEW_TIME.append(TIME[i])
            NEW_FLUX.append(FLUX[i])
            NEW_FLUX_ERR.append(FLUX_ERR[i])
            
    return NEW_TIME, NEW_FLUX, NEW_FLUX_ERR

#####################################################################################################################
#equation for Gaussian curve modulated to depth of transit
#####################################################################################################################

def lc_gaussian(x, A, B, mu, std):
    
    """Modified Gaussian function that is fit to folded lightcurve data


    Returns:
        (B - (A / (std*np.sqrt(2 * pi)) * np.exp(- (((x - mu) / std)**2) / 2))): Modified Gaussian at every x value
    """
    
    return (B - (A / (std*np.sqrt(2 * pi)) * np.exp(- (((x - mu) / std)**2) / 2)))

#####################################################################################################################
#linear equation used in residual plotting
#####################################################################################################################

def linear_reg(x, m, c):
    
    """Linear regression line for use in residual analysis

    Returns:
        (m*x) +c: linear regression line
    """
    
    return  (m*x) +c


#####################################################################################################################
#Chi analysis
#####################################################################################################################

def chi_analysis(FLUX, FLUX_MODEL, FLUX_ERR, DOF):
    
    """Function that carries out chi analysis on data
    - Chi is a measure of how well a given model fits the data
    

    Returns:
        REDUCED_CHI_SQUARED: If close to 1 this indicates a good fit to data
    """
    
    CHI_SQUARED = 0
    TMP_CHI = 0
    
    for i in range(len(FLUX)):
        TMP_CHI = ((FLUX[i] - FLUX_MODEL[i])**2) / FLUX_ERR[i]**2
        CHI_SQUARED = CHI_SQUARED + TMP_CHI
    
    REDUCED_CHI_SQUARED = CHI_SQUARED / (len(FLUX) - DOF - 1)
    
    return REDUCED_CHI_SQUARED     
    
    
#####################################################################################################################
#Transit fitter
#####################################################################################################################

def transit_fitter(xmin,
                   xmax,
                   ymin,
                   PERIOD,
                   TIME,
                   FLUX,
                   FLUX_ERR,
                   EXOPLANET_NUMBER,
                   PLOT = False):
    
    """Function that fits a modified Gaussian to a planetary transit within the limits xmin and xmax
    
    - Uses curve_fit from SciPy to estimate parameters
    - Plots data alongside modified Gaussian
    
    Main parameters:
        xmin: Lower time or phase limit
        xmax: Upper time or phase limit
    Choose a sensible range for xmin and xmax to capture the full transit
        ymin: Initial estimate flux at bottom of transit
        PERIOD =  Orbital period for ith exoplanet
        TIME =  Time measurements from Kepler telescope
        FLUX = Normalised flux measurements from Kepler telescope
        FLUX_ERR = Error in flux measurements from Kepler telescope
        
    Fitting parameters:
        A : Transit depth coefficient, used to modulate depth of transit
        B : Flux without transit present (should equal 1 for normalised lc)
        mu : Mean time
        std: Standard deviation of the flux
        
    Returns:
        (parameters, parameters_error): List of optimised parameters for Gaussian fit
    """

    PHASE, FLUX, FLUX_ERR = fold_lightcurve(TIME, FLUX, FLUX_ERR, PERIOD)
    TIME = PHASE * PERIOD
    
    #Data sliced to relevant range for each transit:

    TMP_TIME = array_slicer(TIME, FLUX, FLUX_ERR, xmin, xmax)[0]
    TMP_FLUX = array_slicer(TIME, FLUX, FLUX_ERR, xmin, xmax)[1]
    TMP_FLUX_ERR = array_slicer(TIME, FLUX, FLUX_ERR, xmin, xmax)[2]
    TMP_PHASE = array_slicer(PHASE, FLUX, FLUX_ERR, xmin / PERIOD, xmax / PERIOD)[0]
    
    xdata = np.linspace(xmin, xmax, len(TMP_TIME)) #x values for fitted function
    
    #Fitting parameters:
    mu = np.mean(TMP_TIME) #median of time value
    std = np.std(TMP_TIME) # standard deviation of flux distribution
    A = (std * np.sqrt(2 * pi)) * (1 - ymin)  #sets transit depth coefficient, modulated to estimate of flux value at bottom of transit
    

    
    
    #Optimise parameters to data:
    popt, pcov = curve_fit(lc_gaussian, TMP_TIME, TMP_FLUX, p0=[A, 1, mu, std], sigma=TMP_FLUX_ERR, maxfev=10000)
    
    #print out fitted parameters with error
    parameters = popt
    parameters_error = np.sqrt(np.diag(pcov))
    parameter_list = ["A", "B", "mu", "std"]
    
    #print out mu in [phase]:
    mu_PHASE = parameters[2] / PERIOD
    mu_PHASE_ERR = parameters_error[2] / PERIOD
    
    mu_TIME = parameters[2]
    mu_TIME_ERR = parameters_error[2]
    
    
    
          
    if PLOT: # if true, transits will be plotted along with residuals and chi analysis. Else optimised parameters will be returned
          
    #Plot transit and lightcurve:
    
        print("------------------------------------------------------------------------------------------------")


        #Plot lightcurve:
        
        
        plt.rcParams.update({'font.size': 12})
        
        #create subplot
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.errorbar(TMP_TIME, TMP_FLUX, TMP_FLUX_ERR, marker='o', ls='None', zorder=4, label='Data', markersize=0.1, elinewidth=0.5) #label="_nolegend_")
        ax1.plot(xdata, lc_gaussian(xdata, popt[0], popt[1], popt[2], popt[3]), zorder=4, lw=2, color="r", label="Fitted Gaussian")
        ax1.set_xlabel('Time / days')
        ax1.set_ylabel('Relative flux')
        
        
        #setting title:
        title_text = f"Folded transit for Exoplanet {EXOPLANET_NUMBER}"
        title = ax1.set_title(title_text, fontsize=14)
        underline = path_effects.withStroke(linewidth=1, foreground='black')
        title.set_path_effects([underline])
        
        ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)

        
        #adding phase scale:
        ax2 = ax1.twiny()
        ax2.set_xlabel('Phase')
        ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax1.legend(loc= 3, fontsize = 12 )
        
        
        
        plt.show()
        
        #chi analysis   
        REDUCED_CHI_SQUARED = 0 
        REDUCED_CHI_SQUARED = chi_analysis(TMP_FLUX, lc_gaussian(xdata, popt[0], popt[1], popt[2], popt[3]), TMP_FLUX_ERR, 4)
        display(Math(r'\chi_r: {}'.format(REDUCED_CHI_SQUARED)))
        
        
        print(f"Time of first transit = {mu_TIME:.4f} ± {mu_TIME_ERR:.2g}")
            
        #residuals plotting:
        
        plt.figure(figsize=(7,4))
        
        fitted_values = lc_gaussian(xdata, popt[0], popt[1], popt[2], popt[3])
        residuals = TMP_FLUX - fitted_values
        residuals_mean = np.mean(residuals)
        
        plt.scatter(fitted_values, residuals, ls='None', marker='o', label='Residuals', s=4)
        plt.hlines(0, np.min(fitted_values), np.max(fitted_values), color='red', label='Zero line')
        plt.xlabel("Fitted values / Flux")
        plt.ylabel("Residuals")
        plt.title(f"Residuals for Exoplanet {EXOPLANET_NUMBER}")
        plt.legend(loc= 3, fontsize = 12 )
        plt.show()
        print(f"Mean of the residuals for Exoplanet {EXOPLANET_NUMBER} = {residuals_mean:.2g}")
        print("------------------------------------------------------------------------------------------------")
        
        return    
     
    else:
        return(parameters, parameters_error)

#####################################################################################################################
#error printer
#####################################################################################################################

def error_rounder(value, error):
    
    """Function that rounds 'value' to same order of magnitude as 'error' rounded to 2 significant figures
    
    Currently only works for error > 1
    
    Returns:
        (rounded_value, rounded_error): Correctly rounded values
    """
    
    rounded_error =  round(error, -int(math.floor(math.log10(abs(error)))) + 1)
    power = math.floor(math.log10(error))
    
    rounded_value = round(value, - ( power -1))
    
   # print(f"{name} = {rounded_value} ± {rounded_error} [{unit}]")

    return (rounded_value, rounded_error)


#####################################################################################################################
#Fitted function analysis
#####################################################################################################################

"""
Extract exoplanet properties from fitted function
"""

def semi_major_axis(PERIOD, PERIOD_ERR):
    
    """Function that finds semi-major axis of exoplanet from orbital period
    
    - Assumes Sun-like star
    - Uses Kepler's 3rd law
    - Enter PERIOD, PERIOD_ERR in [days]

    Returns:
        SEMI_MAJOR_AXIS, SMA_ERR: semi-major-axis ± error in metres
    """
    
    
    PERIOD_S = PERIOD * 24*3600 #convert period into seconds for calculation
    PERIOD_S_ERR = PERIOD_ERR * 24 * 3600
    
    
    SEMI_MAJOR_AXIS = np.cbrt(( (GRAVITATIONAL_CONSTANT*SUN_MASS) / (4*pi*pi) ) * PERIOD_S**2) #SMA in metres
    SMA_AU = SEMI_MAJOR_AXIS / AU #SMA in astronomical units
    
    #compute SMA error using error propagation formula:
    #added 5% error to compensate for assuming Sun-like star
    
    SMA_ERR = SEMI_MAJOR_AXIS * (2/3) * (PERIOD_ERR/PERIOD + 0.05)
    
    return SEMI_MAJOR_AXIS, SMA_ERR

def transit_depth(A, A_ERR, STD, STD_ERR):
    
    """Function that finds transit depth from coeffiecient A and standard deviation STD

    Returns:
        TRANSIT_DEPTH, TRANSIT_DEPTH_ERR: transit depth ± error in [relative flux]
    """
    TRANSIT_DEPTH = (1 / np.sqrt(2 * pi)) * (A / STD)
    TRANSIT_DEPTH_ERR = TRANSIT_DEPTH * np.sqrt((A_ERR / A)**2 + (STD_ERR / STD)**2 )
    
    return TRANSIT_DEPTH, TRANSIT_DEPTH_ERR

def exoplanet_radius(TRANSIT_DEPTH, TRANSIT_DEPTH_ERR, B, B_ERR, RSTAR = SUN_RADIUS):
    
    """Function that finds exoplanet radius from ratio of flux change to areas of planet and star
    - RSTAR is in [metres]
    - TRANSIT_DEPTH in [relative flux]
    
    Returns:
        EXOPLANET_RADIUS, EXOPLANET_RADIUS_ERR: in [metres]
    """
    EXOPLANET_RADIUS = RSTAR * np.sqrt(TRANSIT_DEPTH * B)
    
    #use partial deriviatives to find error in exoplanet radius
    #added 5% error to compensate for assuming sun-like star

    
    partial1 = np.sqrt(TRANSIT_DEPTH * B)
    partial2 = 0.5 * RSTAR * TRANSIT_DEPTH**(-0.5) * np.sqrt(B)
    partial3 = 0.5 * RSTAR * np.sqrt(TRANSIT_DEPTH) * B**(-0.5)
    
    EXOPLANET_RADIUS_ERR = np.sqrt((0.05)**2 + (partial2 * TRANSIT_DEPTH_ERR)**2 + (partial3 * B_ERR)**2)
    
    return EXOPLANET_RADIUS, EXOPLANET_RADIUS_ERR #metres units
    
def exoplanet_density(MASS, MASS_ERR, RADIUS, RADIUS_ERR): #enter masses in Earth masses
    
    """Function that finds exoplanet density, assuming uniform mass distribution and perfect sphere
    - Enter MASS in [kg]
    - Enter RADIUS in [metres]
    
    Returns:
        EXOPLANET_DENSITY, EXOPLANET_DENSITY_ERR: returns in kg
    """
        
    VOLUME= (4/3) * pi * (RADIUS)**3
    EXOPLANET_DENSITY = MASS / VOLUME
    
    #error analysis -> partial deriviatives used:
    
    partial1 = (3/(4*pi)) * RADIUS**(-3)
    partial2 = (-9 / (4*pi)) * MASS * RADIUS**(-4)
    
    EXOPLANET_DENSITY_ERR = np.sqrt((partial1 * MASS_ERR)**2 + (partial2 * RADIUS_ERR)**2)
    
    return EXOPLANET_DENSITY, EXOPLANET_DENSITY_ERR #returns mass in kg

def theoretical_transit_time(PERIOD, PERIOD_ERR, SMA, SMA_ERR):
    
    """Function that finds theoretical transit time from ratio of stellar radius to semi-major axis
    - SMA must be in [AU]
    - PERIOD in [days]

    Returns:
        TRANSIT_TIME, TRANSIT_TIME_ERR: returns transit times in [days]
    """
    
    TRANSIT_TIME = PERIOD * ( (SUN_RADIUS/AU) / (pi * SMA) )
    
    #added 5% error to compensate for assuming sun-like star
    TRANSIT_TIME_ERR = TRANSIT_TIME * np.sqrt( (PERIOD_ERR /PERIOD)**2 + (SMA_ERR/SMA)**2 + 0.05**2)
    
    return TRANSIT_TIME, TRANSIT_TIME_ERR

def graphical_transit_time(std, std_err):
    
    """Function that finds graphical transit times using standard deviation

    - Graphically estimates size of transit time from standard deviation of modified Gaussian
    - Standard deviation can be linked to spread of data
    - 95% data is contained within ±2 std -> 4 std contains 95% of transit
    - Error of 5% added
    
    Returns:
        TRANSIT_TIME, TRANSIT_TIME_ERR: in [days]
    """
    TRANSIT_TIME = 4*std
    TRANSIT_TIME_ERR = 4*std_err + (0.05 * (std))
    
    return TRANSIT_TIME, TRANSIT_TIME_ERR

def surface_gravity(EXOPLANET_DENSITY, EXOPLANET_DENSITY_ERR, EXOPLANET_RADIUS, EXOPLANET_RADIUS_ERR):
    
    """Function finds surface gravity of planet 
    - All inputs must be in SI units

    Returns:
        GRAVITY, GRAVITY_ERR: in [ms^-2]
    """
    
    GRAVITY = ( (4*pi) /3 ) * GRAVITATIONAL_CONSTANT * EXOPLANET_DENSITY * EXOPLANET_RADIUS / G_EARTH
    GRAVITY_ERR = GRAVITY * np.sqrt( (EXOPLANET_DENSITY_ERR / EXOPLANET_DENSITY)**2 + (EXOPLANET_RADIUS_ERR / EXOPLANET_RADIUS)**2 ) 
    
    return GRAVITY, GRAVITY_ERR

def bondi(MASS, MASS_ERR):
    
    """Function finds Bondi radius for a given exoplanet assuming Earth like atmosphere
    - Enter MASS in [Earth masses]
    - Enter radius [m]

    Returns:
        BONDI_RADIUS, BONDI_RADIUS_ERR: returns in metres
    """
    
    TMP_MASS = MASS * EARTH_MASS
    TMP_MASS_ERR = MASS_ERR * EARTH_MASS
    
    BONDI_RADIUS = (GRAVITATIONAL_CONSTANT * TMP_MASS) / (C_SOUND**2)
    BONDI_RADIUS_ERR = (GRAVITATIONAL_CONSTANT * TMP_MASS_ERR) / (C_SOUND**2)
    
    return BONDI_RADIUS, BONDI_RADIUS_ERR

def temperature(SMA, SMA_ERR):
    
    """Function estimates surface temperature of a given exoplanet
    - Assumes Sun like star

    Args:
        SMA: semi-major axis [m]
        SMA_ERR: semi-major axis error [m]

    Returns:
        TEMPERATURE: estimate of temperature in Kelvin
        TEMPERATURE_ERR: temperature error
        
    """
    
    #added 5% error to compensate for assuming sun-like star
    TEMPERATURE = T_EFF * np.sqrt( (0.9 * SUN_RADIUS) / (2*SMA) )
    TEMPERATURE_ERR = 0.5* (SMA_ERR/SMA + 0.05) * TEMPERATURE
    
    
    return TEMPERATURE, TEMPERATURE_ERR
    

#####################################################################################################################
#Exoplanet analyser
#####################################################################################################################

def exoplanet_analyser(EXOPLANET_NUMBER, #give planet an ID number
                       EXOPLANET_PERIOD, EXOPLANET_PERIOD_ERR,
                       xmin, xmax, ymin, TIME, FLUX, FLUX_ERR,
                       EXOPLANET_MASS, UPPER_EXOPLANET_MASS, LOWER_EXOPLANET_MASS): #enter masses in Earth masses
                       

    """Idea is to create dictionary that stores all relevant physical properties for each exoplanet
    - Pass exoplanet_analyser values to calculate exoplanet physical properties
    - Calls upon functions in #fitted function analysis
    - Rounds to appropriate significant figures
    

    Returns:
        planet_dict: Can be cast to pandas DataFrame
    """
    
    planet_dict = {"Exoplanet Number": [],
                   "Orbital Period [days]": [],
                   "Orbital Period uncertainty [days]": [],
                   "Semi-major Axis [AU]": [],
                   "Semi-major Axis [AU] uncertainty": [],
                   "Transit depth [relative flux]": [],
                   "Transit depth uncertainty [relative flux]": [],
                   "Exoplanet radius [Earth radii]": [],
                   "Exoplanet radius uncertainty [Earth radii]": [],
                   "Bondi Radius [Earth radii]": [],
                   "Upper Bondi radius uncertainty [Earth radii]": [],
                   "Lower Bondi radius uncertainty[ Earth radii]": [],
                   "Mass [Earth masses]": [],
                   "Upper Mass [Earth masses]": [],
                   "Lower Mass [Earth masses]": [],
                   "Density [grams per cubic cm]": [],
                   "Upper density uncertainty [grams per cubic cm]": [],
                   "Lower density uncertainty [grams per cubic cm]": [],
                   "Surface Gravity [g]": [],
                   "Upper gravity uncertainty [g]": [],
                   "Lower gravity uncertainty [g]": [],
                   "Transit duration [hours][theoretical]": [],
                   "Transit duration uncertainty [hours][theoretical]": [],
                   "Transit duration [hours][graphical]": [],
                   "Transit duration uncertainty [hours][graphical]": [],
                   "Surface Temperature [K]": [],
                   "Surface Temperature uncertainty [K]": []                   
                   }
    
    #ID number
    planet_dict['Exoplanet Number'].append(EXOPLANET_NUMBER)
    
    #period
    planet_dict["Orbital Period [days]"].append(round(EXOPLANET_PERIOD, 3))
    planet_dict["Orbital Period uncertainty [days]"].append(round(EXOPLANET_PERIOD_ERR, 3))
    
    #semi major axis
    EXOPLANET_SMA = semi_major_axis(EXOPLANET_PERIOD, EXOPLANET_PERIOD_ERR)[0]
    EXOPLANET_SMA_ERR = semi_major_axis(EXOPLANET_PERIOD, EXOPLANET_PERIOD_ERR)[1]
    
    planet_dict["Semi-major Axis [AU]"].append(round(EXOPLANET_SMA / AU, 4))
    planet_dict['Semi-major Axis [AU] uncertainty'].append(round(EXOPLANET_SMA_ERR / AU, 4))
    
    #surface temperature
    TEMPERATURE = temperature(EXOPLANET_SMA, EXOPLANET_SMA_ERR)[0]
    TEMPERATURE_ERR = temperature(EXOPLANET_SMA, EXOPLANET_SMA_ERR)[1]
    
    planet_dict['Surface Temperature [K]'].append(error_rounder(TEMPERATURE, TEMPERATURE_ERR)[0])
    planet_dict['Surface Temperature uncertainty [K]'].append(error_rounder(TEMPERATURE, TEMPERATURE_ERR)[1])
    
    #theoretical transit time:
    EXOPLANET_SMA_AU = EXOPLANET_SMA / AU
    EXOPLANET_SMA_AU_ERR = EXOPLANET_SMA_ERR / AU
    
    THEO_TRANSIT_TIME = theoretical_transit_time(EXOPLANET_PERIOD, EXOPLANET_PERIOD_ERR, EXOPLANET_SMA_AU, EXOPLANET_SMA_AU_ERR)[0] * 24
    THEO_TRANSIT_TIME_ERR = theoretical_transit_time(EXOPLANET_PERIOD, EXOPLANET_PERIOD_ERR, EXOPLANET_SMA_AU, EXOPLANET_SMA_AU_ERR)[1] * 24
    
    planet_dict["Transit duration [hours][theoretical]"].append(round(THEO_TRANSIT_TIME, 2))
    planet_dict["Transit duration uncertainty [hours][theoretical]"].append(round(THEO_TRANSIT_TIME_ERR, 2))
    
    #radius
    CHI = False
    A = transit_fitter(xmin, xmax, ymin, EXOPLANET_PERIOD, TIME, FLUX, FLUX_ERR, EXOPLANET_NUMBER, CHI)[0][0]
    A_ERR = transit_fitter(xmin, xmax, ymin, EXOPLANET_PERIOD, TIME, FLUX, FLUX_ERR, EXOPLANET_NUMBER, CHI)[1][0]
    
    B = transit_fitter(xmin, xmax, ymin, EXOPLANET_PERIOD, TIME, FLUX, FLUX_ERR, EXOPLANET_NUMBER, CHI)[0][1]
    B_ERR = transit_fitter(xmin, xmax, ymin, EXOPLANET_PERIOD, TIME, FLUX, FLUX_ERR, EXOPLANET_NUMBER, CHI)[1][1]
    
    STD = transit_fitter(xmin, xmax, ymin, EXOPLANET_PERIOD, TIME, FLUX, FLUX_ERR, EXOPLANET_NUMBER, CHI)[0][3]
    STD_ERR = transit_fitter(xmin, xmax, ymin, EXOPLANET_PERIOD, TIME, FLUX, FLUX_ERR, EXOPLANET_NUMBER, CHI)[1][3]
    
    TRANSIT_DEPTH = transit_depth(A, A_ERR, STD, STD_ERR)[0]
    TRANSIT_DEPTH_ERR = transit_depth(A, A_ERR, STD, STD_ERR)[1]
    
    planet_dict["Transit depth [relative flux]"].append(round(TRANSIT_DEPTH, 6))
    planet_dict["Transit depth uncertainty [relative flux]"].append(round(TRANSIT_DEPTH_ERR,6))
    
    EXOPLANET_RADIUS = exoplanet_radius(TRANSIT_DEPTH, TRANSIT_DEPTH_ERR, B, B_ERR, RSTAR = SUN_RADIUS)[0] / EARTH_RADIUS
    EXOPLANET_RADIUS_ERR = exoplanet_radius(TRANSIT_DEPTH, TRANSIT_DEPTH_ERR, B, B_ERR, RSTAR = SUN_RADIUS)[1] / EARTH_RADIUS
    
    planet_dict["Exoplanet radius [Earth radii]"].append(round(EXOPLANET_RADIUS, 3))
    planet_dict["Exoplanet radius uncertainty [Earth radii]"].append(round(EXOPLANET_RADIUS_ERR, 3))
    
    #graphical transit times:
    GRAP_TRANSIT_TIME = graphical_transit_time(STD, STD_ERR)[0] * 24
    GRAP_TRANSIT_TIME_ERR = graphical_transit_time(STD, STD_ERR)[1] * 24
    
    planet_dict["Transit duration [hours][graphical]"].append(round(GRAP_TRANSIT_TIME, 2))
    planet_dict["Transit duration uncertainty [hours][graphical]"].append(round(GRAP_TRANSIT_TIME_ERR, 2))
    
    #masses:
    planet_dict["Mass [Earth masses]"].append(round(EXOPLANET_MASS, 2))
    planet_dict["Upper Mass [Earth masses]"].append(round(UPPER_EXOPLANET_MASS, 2))
    planet_dict["Lower Mass [Earth masses]"].append(round(LOWER_EXOPLANET_MASS, 2))
    
    #density:
    #convert to metric units for calculation
    EXOPLANET_MASS = EXOPLANET_MASS* EARTH_MASS
    UPPER_EXOPLANET_MASS = UPPER_EXOPLANET_MASS * EARTH_MASS
    LOWER_EXOPLANET_MASS = LOWER_EXOPLANET_MASS * EARTH_MASS
    
    EXOPLANET_RADIUS_M = EXOPLANET_RADIUS * EARTH_RADIUS
    EXOPLANET_RADIUS_M_ERR = EXOPLANET_RADIUS_ERR * EARTH_RADIUS
    
    EXOPLANET_DENSITY, EXOPLANET_DENSITY_UPPER, EXOPLANET_DENSITY_LOWER = ma.exoplanet_density(EXOPLANET_MASS, UPPER_EXOPLANET_MASS, LOWER_EXOPLANET_MASS, EXOPLANET_RADIUS_M, EXOPLANET_RADIUS_M_ERR)
    
    planet_dict["Density [grams per cubic cm]"].append(error_rounder(EXOPLANET_DENSITY, EXOPLANET_DENSITY_UPPER)[0] / 1000)
    planet_dict["Upper density uncertainty [grams per cubic cm]"].append(error_rounder(EXOPLANET_DENSITY_UPPER, EXOPLANET_DENSITY_UPPER)[0] / 1000)
    planet_dict["Lower density uncertainty [grams per cubic cm]"].append(error_rounder(EXOPLANET_DENSITY_LOWER, EXOPLANET_DENSITY_LOWER)[0] / 1000)
    
    #surface gravity:
    
    GRAVITY, UPPER_GRAVITY_UNCERTAINTY, LOWER_GRAVITY_UNCERTAINTY = ma.surface_gravity(EXOPLANET_DENSITY, EXOPLANET_DENSITY_UPPER, EXOPLANET_DENSITY_LOWER, EXOPLANET_RADIUS_M, EXOPLANET_RADIUS_M_ERR)
    
    planet_dict["Surface Gravity [g]"].append(round(GRAVITY, 2))
    planet_dict["Upper gravity uncertainty [g]"].append(round(UPPER_GRAVITY_UNCERTAINTY, 2))
    planet_dict["Lower gravity uncertainty [g]"].append(round(LOWER_GRAVITY_UNCERTAINTY, 2))

    
    #Bondi radius:
    BONDI_RADIUS = ma.bondi(EXOPLANET_MASS, UPPER_EXOPLANET_MASS, LOWER_EXOPLANET_MASS)[0] / EARTH_RADIUS 
    UPPER_BONDI_UNCERTAINTY = ma.bondi(EXOPLANET_MASS, UPPER_EXOPLANET_MASS, LOWER_EXOPLANET_MASS)[1] / EARTH_RADIUS
    LOWER_BONDI_UNCERTAINTY = ma.bondi(EXOPLANET_MASS, UPPER_EXOPLANET_MASS, LOWER_EXOPLANET_MASS)[2] / EARTH_RADIUS
    
    planet_dict["Bondi Radius [Earth radii]"].append(error_rounder(BONDI_RADIUS, UPPER_BONDI_UNCERTAINTY)[0])
    planet_dict["Upper Bondi radius uncertainty [Earth radii]"].append(error_rounder(UPPER_BONDI_UNCERTAINTY, UPPER_BONDI_UNCERTAINTY)[0])
    planet_dict["Lower Bondi radius uncertainty[ Earth radii]"].append(error_rounder(LOWER_BONDI_UNCERTAINTY, UPPER_BONDI_UNCERTAINTY)[0])
    
    
        
    return planet_dict

#####################################################################################################################
#Archive plotters
#####################################################################################################################
    
def archive_plotter():
    
    """Function plots data from NASA exoplanet archive along with planets detected in this investigation
    
    """
    
    exo_data = pd.read_csv('PS_2023.11.28_07.47.32.csv')
    #column 1 is semi major axis / AU
    #column 2 is planetary mass / Earth masses

    semi_major_axes = exo_data['pl_orbsmax']
    masses = exo_data['pl_masse']

    #mask masses to remove NaNs
    indices = np.logical_not(np.isnan(masses))
    semi_major_axes = semi_major_axes[indices]
    masses = masses[indices]

    #my data
    my_sma = [0.108332, 0.156834, 0.197235] #my sma values in AU
    my_masses = [2.9, 7.3, 8.8]

    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 12})
    plt.scatter(semi_major_axes, masses, ls='None', marker='o', label='Exoplanet archive exoplanets', s=4, alpha=.2)
    plt.scatter(my_sma, my_masses, ls='None', marker='o', label='Kepler Telescope Exoplanets', s=30, color='red')
    plt.xlabel('Semi-major axes / AU')
    plt.ylabel('Exoplanet mass / Earth masses')
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Exoplanets from NASA exoplanet archive")
    plt.legend()
    
