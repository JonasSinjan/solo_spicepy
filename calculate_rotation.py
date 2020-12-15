import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sun_rotation = 1997 #avg, m/s, not including differential rotation

def w(phi):
  """returns angular velocity in degrees per day dependent on the latitude
  
  Parameters
  ----------
  phi float, solar latitude in degrees
  
  Output
  ------
  w: the angular velocity in degrees per day

  """

  A = 14.713 #± 0.0491 
  B = -2.396
  C = 1.787

  omega = A + B*(np.sin(phi))**2 + C*(np.sin(phi))**4
  omega_rad_per_second = omega*(np.pi/180)*(24*60*60)
  return  omega_rad_per_second #angular velocity in radians per second


#phi has pixel size of 0.5 arcseconds
#need to know distance from phi
#also need to correct for solar orbiter's rotation