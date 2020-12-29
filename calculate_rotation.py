import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sun_rotation = 1997 #avg, m/s, not including differential rotation

def w(phi):
  """returns angular velocity in arcseconds per second dependent on the latitude
  
  Parameters
  ----------
  phi: float, solar latitude in degrees
  
  Output
  ------
  omega_arcsecond_per_second: float, the angular velocity in degrees per second

  """

  A = 14.713 #± 0.0491 
  B = -2.396 #± 0.188 
  C = 1.787 #± 0.253

  omega = A + B*(np.sin(phi))**2 + C*(np.sin(phi))**4
  omega_arcsecond_per_second = omega*3600*(24*60*60)
  return  omega_arcsecond_per_second #angular velocity in radians per second


#phi has pixel size of 0.5 arcseconds
#need to know distance from phi
#also need to correct for solar orbiter's rotation

def get_solar_orbiter_rotation():
  #need to get the solar orbiter rotation, from spice kernels?

  return omega


def convert_w_to_pixels(omega_arcsecond_per_second, time):

  """returns amount the pixels shifted
  
  Parameters
  ----------
  omega_arcsecond_per_second: float, the angular velocity of a point on the Sun in degrees per second
  time: float, time the images moved (or cadence)
  
  Output
  ------
  total_pixel_shift: float, the amount of pixels the scene moved across the detector, deal with float converting to int in other program

  """
  solar_orbiter = get_solar_orbiter_rotation()
  relative_omega = omega_arcsecond_per_second - solar_orbiter

  total_pixel_shift = relative_omega*time/0.5

  return total_pixel_shift
