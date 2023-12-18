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
#Asymmetric mass calculations
#####################################################################################################################

def exoplanet_density(MASS, UPPERMASS, LOWERMASS, RADIUS, RADIUS_ERR):
    
    """Function to find exoplanet density -> Uses asymmetric mass errors
    - Enter MASS, UPPERMASS, LOWERMASS, in [kg]
    - Enter RADIUS in [metres]
    
    Returns:
        DENSITY, UPPER_UNCERTAINTY, LOWER_UNCERTAINTY: mass density + upper uncertainty - lower uncertainty
    """
    
    VOLUME = (4/3) * pi * (RADIUS)**3
    UPPER_VOLUME = (4/3) * pi * (RADIUS + RADIUS_ERR)**3
    LOWER_VOLUME = (4/3) * pi * (RADIUS - RADIUS_ERR)**3
    
    DENSITY = MASS / VOLUME
    
    UPPER_DENSITY = UPPERMASS / LOWER_VOLUME
    LOWER_DENSITY = LOWERMASS / UPPER_VOLUME
    
    UPPER_UNCERTAINTY = UPPER_DENSITY - DENSITY
    LOWER_UNCERTAINTY = DENSITY - LOWER_DENSITY
    
    return DENSITY, UPPER_UNCERTAINTY, LOWER_UNCERTAINTY
    
    
def surface_gravity(DENSITY, UPPER_UNCERTAINTY, LOWER_UNCERTAINTY, RADIUS, RADIUS_ERR):
    
    """Function finds surface gravity of planet with asymmetric errors
    - All inputs must be in SI units
    - Output is in [g]

    Returns:
        GRAVITY, UPPER_GRAVITY_UNCERTAINTY, LOWER_GRAVITY_UNCERTAINTY: upper and lower gravity uncertainty in [g]
    """
    
    GRAVITY = ( (4*pi) /3 ) * GRAVITATIONAL_CONSTANT * DENSITY * RADIUS / G_EARTH

    #upper limit on gravity
    UPPER_GRAVITY = ( (4*pi) /3 ) * GRAVITATIONAL_CONSTANT * (DENSITY + UPPER_UNCERTAINTY) * (RADIUS + RADIUS_ERR) / G_EARTH
    
    #lower limit on gravity
    LOWER_GRAVITY = ( (4*pi) /3 ) * GRAVITATIONAL_CONSTANT * (DENSITY - LOWER_UNCERTAINTY) * (RADIUS - RADIUS_ERR) / G_EARTH
    
    UPPER_GRAVITY_UNCERTAINTY = UPPER_GRAVITY - GRAVITY
    LOWER_GRAVITY_UNCERTAINTY = GRAVITY - LOWER_GRAVITY
    
    
    return GRAVITY, UPPER_GRAVITY_UNCERTAINTY, LOWER_GRAVITY_UNCERTAINTY

def bondi(MASS, UPPER_MASS, LOWER_MASS):
    """Function finds Bondi radius for a given exoplanet assuming Earth like atmosphere and asymmetric mass errors.

    Args:
        MASS (_type_): Enter in [kg]
        UPPERMASS (_type_): Upper mass limit
        LOWER_MASS (_type_): Lower mass limit
        
    Returns:
        BONDI, UPPER_BONDI_UNCERTAINTY, LOWER_BONDI_UNCERTAINTY: Bondi radius with upper and lower uncertainty
    """
    
    TMP_MASS = MASS # * EARTH_MASS
    TMP_LOWER_MASS = LOWER_MASS #*  EARTH_MASS
    TMP_UPPER_MASS = UPPER_MASS #* EARTH_MASS
    
    BONDI = (GRAVITATIONAL_CONSTANT * TMP_MASS) / (C_SOUND**2)
    LOWER_BONDI = (GRAVITATIONAL_CONSTANT * TMP_LOWER_MASS) / (C_SOUND**2)   
    UPPER_BONDI = (GRAVITATIONAL_CONSTANT * TMP_UPPER_MASS) / (C_SOUND**2)  
    
    UPPER_BONDI_UNCERTAINTY = UPPER_BONDI - BONDI
    LOWER_BONDI_UNCERTAINTY = BONDI - LOWER_BONDI
    
    return BONDI, UPPER_BONDI_UNCERTAINTY, LOWER_BONDI_UNCERTAINTY
