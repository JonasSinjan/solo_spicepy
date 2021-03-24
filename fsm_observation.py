# -*- coding: utf-8 -*-

import numpy as np
import os
import spiceypy as sp
from pathlib import Path
import spiceypy.utils.support_types as stypes
import json
import xml.etree.ElementTree as xmlet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import more_itertools as mit

import json, requests, math, drms, sunpy.map
import matplotlib.colors as mcol
from astropy.io import fits

from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from IPython.core.debugger import set_trace

def loadkernel(kpath, kname):
    "This function loads a SPICE kernel (which could be a metakernel) then returns to the current working directory."
    cur_wd = os.getcwd()
    os.chdir(kpath)
    sp.furnsh(kname)
    os.chdir(cur_wd)
    nloaded = sp.ktotal("ALL")
    return(nloaded)
    
def unloadkernel(kpath,kname):
    "This function unloads a SPICE kernel (which could be a metakernel) then returns to the current working directory."
    cur_wd = os.getcwd()
    os.chdir(kpath)
    sp.unload(kname)
    os.chdir(cur_wd)
    nloaded = sp.ktotal("ALL")
    return(nloaded)
    
def get_solo_coverage(mkpath):
    "This function simply returns the coverage of the loaded Solar Orbiter orbit kernel."
    for kernel in range(0,sp.ktotal("ALL")-1):
        kernel_data=sp.kdata(kernel,"ALL")
        if "solo_ANC_soc-orbit" in kernel_data[0]:
            solo_coverage = stypes.SPICEDOUBLE_CELL(2)
            kernel_path = os.path.join(mkpath,Path(kernel_data[0]))
            sp.spkcov(kernel_path,-144,solo_coverage)
            coverage_out=sp.wnfetd(solo_coverage,0)
    return(coverage_out)

def get_orbit_coverage(kernel_path, kernel_name, obj_id):
    "This function simply returns the coverage of the given orbit kernel."
    orbit_coverage = stypes.SPICEDOUBLE_CELL(200)
    kernel_path = os.path.join(kernel_path,kernel_name)
    sp.spkcov(kernel_path,obj_id,orbit_coverage)
    coverage_out=sp.wnfetd(orbit_coverage,0)
    return(coverage_out)



def get_NAV_windows(spacecraft,planet,times,threshold):
    """Calculates Closest approach between SPACECRAFT and PLANET, based on first getting intervals
    where planetocentric distance is below THRESHOLD, then finding the minimum in each interval.
    
    Output is an n x 3 array where each row is a nav window specification containing CA-28 days, CA, CA + 7 days."""
    interval = stypes.SPICEDOUBLE_CELL(2)
    sp.wninsd(times[0],times[1],interval)
    gam_times = stypes.SPICEDOUBLE_CELL(2000)
    sp.gfdist(planet,"NONE",spacecraft,"<",threshold,0.0,86400.0,1000,interval,gam_times)
    ngams = sp.wncard(gam_times)
    closest_approach = stypes.SPICEDOUBLE_CELL(2*ngams)
    sp.gfdist(planet,"NONE",spacecraft,"LOCMIN",threshold,0.0,86400.0,1000,gam_times, closest_approach)
    n_ca = sp.wncard(closest_approach)
    nav_windows = np.zeros([n_ca,3],dtype="float64")
    for i in range(n_ca):
        ca=sp.wnfetd(closest_approach,i)
        nav_windows[i,:] = [ca[0]-(28*86400.0),ca[0],ca[0]+(7*86400.0)]
    return(nav_windows)

def get_RSWs(times):
    """Calculates times of default RSWs. In the case there are overlapping windows, 
    the second window is shifted later to be contiguous with the first.
    Inputs are simply the coverage of the orbit file as a 2 element variable, in
    SPICE ephemeris time.
    Uses the SPICE geometric event finder."""
    
    interval = stypes.SPICEDOUBLE_CELL(2)
    sp.wninsd(times[0],times[1],interval)
    per_times = stypes.SPICEDOUBLE_CELL(100)
    win_start = []
    win_end = []
    win_type = []
    sp.gfdist("SUN","NONE","SOLO","LOCMIN",1.5e8,0.0,86400.0,1000,interval,per_times)
    for i in range(sp.wncard(per_times)):
        win_start.append(sp.wnfetd(per_times,i)[0]-(5*86400))
        win_end.append(sp.wnfetd(per_times,i)[0]+(5*86400))
        win_type.append("Perihelion Window")
    nor_times = stypes.SPICEDOUBLE_CELL(100)
    sp.gfposc("SOLO", "SUN_INERTIAL", "NONE", "SUN", "LATITUDINAL", "LATITUDE",  "LOCMAX", 0.0, 0.0, 86400.0, 1000, interval, nor_times )
    for i in range(sp.wncard(nor_times)):
        win_start.append(sp.wnfetd(nor_times,i)[0]-(5*86400))
        win_end.append(sp.wnfetd(nor_times,i)[0]+(5*86400))
        win_type.append("North Window")
    sou_times = stypes.SPICEDOUBLE_CELL(100)
    sp.gfposc("SOLO", "SUN_INERTIAL", "NONE", "SUN", "LATITUDINAL", "LATITUDE",  "LOCMIN", 0.0, 0.0, 86400.0, 1000, interval, sou_times )
    for i in range(sp.wncard(sou_times)):
        win_start.append(sp.wnfetd(sou_times,i)[0]-(5*86400))
        win_end.append(sp.wnfetd(sou_times,i)[0]+(5*86400))
        win_type.append("South Window")
    win_start=np.array(win_start)
    win_end = np.array(win_end)
    win_type = np.array(win_type)
    indices = win_start.argsort()
    win_start=win_start[indices]
    win_end = win_end[indices]
    win_type=win_type[indices]
    for i in range(len(win_start)-2):
        if win_end[i] > win_start[i+1]:
            win_start[i+1] = win_end[i]+1
            win_end[i+1] = win_start[i+1]+(86400.0*10)
    win_start_2 = et2datetime64(win_start)
    win_start_3 = []
    for ws in win_start_2:
        buffer = np.datetime64(ws,"D")
        win_start_3.append(datetime642et(buffer)[0])
    win_end_2 = et2datetime64(win_end)
    win_end_3 = []
    for we in win_end_2:
        buffer = np.datetime64(we,"D")
        win_end_3.append(datetime642et(buffer)[0])
    return((list(win_start_3),list(win_end_3),list(win_type)))
    
def write_rsws(rsws, outfile):
    startstrs = np.datetime_as_string(et2datetime64(rsws[0]),unit="ms")
    endstrs = np.datetime_as_string(et2datetime64(rsws[1]),unit="ms")
    with open(outfile, 'w') as savehere:
        savehere.write("{\n")
        for index in enumerate(startstrs):
            bigkey = '    "RSW'+str(index[0]+1).zfill(3)+'":' 
            content=json.dumps({"start":str(startstrs[index[0]]), "end":str(endstrs[index[0]]), "type":str(rsws[2][index[0]])})
            if index[0]+1 < len(startstrs):
                content=content+','
            savehere.write(bigkey+content+"\n")
        savehere.write('}')
        savehere.close
    return(outfile)
    
def et2datetime64(ets):
    outtime = []
    try:
        iterator = iter(ets)
    except TypeError:
        ets2=[ets]
    else:
        ets2=ets
    for et in ets2:
        utc_string=sp.et2utc(et,"ISOC",3)
        outtime.append(np.datetime64(utc_string,"ms"))
    return(outtime)

def datetime642et(dts):
    outets = []
    timstr = np.datetime_as_string(dts,unit="ms")
    if isinstance(timstr,str):
        timstr=[timstr]
    for time in timstr:
        outet=sp.utc2et(time)
        outets.append(outet)
    return(outets)

def get_planning_periods(ets,rsws):
    datetime_bounds = et2datetime64(ets)
    startyear = np.datetime64(datetime_bounds[0],"Y")
    firstjuly = startyear+np.timedelta64(6,"M")
    if datetime_bounds[0] < firstjuly:
        pp_start = startyear
    else:
        pp_start = firstjuly
    endyear = np.datetime64(datetime_bounds[1],"Y")
    lastjuly = endyear+np.timedelta64(6,"M")
    if datetime_bounds[1] < lastjuly:
        pp_end = lastjuly
    else:
        pp_end = endyear+np.timedelta64(1,"Y")
    LTP_start = np.arange(pp_start,pp_end,6,dtype="datetime64[M]")
    LTP_end = np.arange(pp_start,pp_end,6,dtype="datetime64[M]")+np.timedelta64(6,"M")
    LTP_start2 = []
    LTP_end2 = []
    for (LTP_s, LTP_e) in zip(LTP_start,LTP_end):
        LTP_start2.append(np.datetime64(LTP_s,"D"))
        LTP_end2.append(np.datetime64(LTP_e,"D"))
    [rsw_start,rsw_end,rsw_type]=rsws
    rsw_start2 = [x - (86400.0*4.0) for x in rsw_start]
    rsw_start3 = et2datetime64(rsw_start2) # subtract 4 days to account for RSW extensions.
    rsw_end = et2datetime64(rsw_end)
    for i, (LTP_s, LTP_e) in enumerate(zip(LTP_start,LTP_end)):
        indices = np.where(np.logical_and(np.logical_and(np.greater_equal(rsw_start3,LTP_s),np.less(rsw_start3,LTP_e)),np.greater_equal(rsw_end,LTP_e)))
        if indices[0].size > 0:
            LTP_end2[i] = np.datetime64(rsw_start3[indices[0][0]],"D")
            LTP_start2[i+1] = np.datetime64(rsw_start3[indices[0][0]],"D")           
    LTP_start_et = datetime642et(LTP_start2)
    LTP_end_et = datetime642et(LTP_end2)
    return((LTP_start_et,LTP_end_et))
    
def get_conjunctions(times, angle1, angle2):
    #longest combination of intervals where  sun-earth-spacecraft angle less than "angle" degrees and/or sun-sc-earth angle less than angle degrees.
    #cant handle multiple overlaps between the different angle conditions in a single interval
    threshold1 = np.deg2rad(angle1)
    threshold2 = np.deg2rad(angle2)
    interval = stypes.SPICEDOUBLE_CELL(2)
    sp.wninsd(times[0],times[1],interval)
    sesc_times = stypes.SPICEDOUBLE_CELL(100)
    ssce_times = stypes.SPICEDOUBLE_CELL(100)
    sesc_start=[]
    sesc_end=[]
    ssce_start=[]
    ssce_end=[]
    sp.gfsep("SUN","POINT","NULL","SOLO","POINT","NULL","NONE","EARTH", "<", threshold1,0.0,86400.0,1000,interval,sesc_times)    
    sp.gfsep("SUN","POINT","NULL","EARTH","POINT","NULL","NONE","SOLO", "<", threshold2,0.0,86400.0,1000,interval,ssce_times)
    for i in range(sp.wncard(sesc_times)):
        sesc_start.append(sp.wnfetd(sesc_times,i)[0])
        sesc_end.append(sp.wnfetd(sesc_times,i)[1])
    for i in range(sp.wncard(ssce_times)):
        ssce_start.append(sp.wnfetd(ssce_times,i)[0])
        ssce_end.append(sp.wnfetd(ssce_times,i)[1])
    conj_start = []
    conj_end = []
    conj_source = []
    if len(sesc_start) >= len(ssce_start):
        long_start = sesc_start
        long_end = sesc_end
        short_start = ssce_start
        short_end = ssce_end
        long = "sesc"
        short = "ssce"
    else:
        long_start = ssce_start
        long_end = ssce_end
        short_start = sesc_start
        short_end = sesc_end
        long = "ssce"
        short = "sesc"
    overlap = np.full(len(short_start),np.inf)
    for i, (l_start,l_end) in enumerate(zip(long_start,long_end)):
        for j, (s_start,s_end) in enumerate(zip(short_start,short_end)):
            latest_start = max(l_start, s_start)
            earliest_end = min(l_end, s_end)
            delta = earliest_end - latest_start
            if delta >= 0:
                conj_start.append(min(l_start,s_start))
                conj_end.append(max(l_end,s_end))
                conj_source.append("both")
                overlap[j]=i
        if not i in overlap:
            conj_start.append(l_start)
            conj_end.append(l_end)
            conj_source.append(long)
    indices = np.where(overlap == np.inf)
    if len(indices[0]) > 0:
        for index in indices[0]:
            conj_start.append(short_start[index])
            conj_end.append(short_end[index])
            conj_source.append(short)
    output = (conj_start,conj_end,conj_source)
    return(output)

def read_json(filename):
    if filename:
        with open(filename, 'r') as f:
            config = json.load(f)
    return(config)

def read_rscws(filename):
    rscws = read_json(filename)
    rscw_start = []
    rscw_end = []
    rscw_type = []
    for rscw in rscws:
        rscw_start.append(sp.utc2et(rscws[rscw]["start"]))
        rscw_end.append(sp.utc2et(rscws[rscw]["end"]))
        rscw_type.append(rscws[rscw]["type"])
    return((rscw_start,rscw_end,rscw_type))

def read_rsws(filename):
    rsws = read_json(filename)
    rsw_start = []
    rsw_end = []
    rsw_type = []
    for rsw in rsws:
        rsw_start.append(sp.utc2et(rsws[rsw]["start"]))
        rsw_end.append(sp.utc2et(rsws[rsw]["end"]))
        rsw_type.append(rsws[rsw]["type"])
    return((rsw_start,rsw_end,rsw_type))
    
def calc_relative_rotation(hci_state):
    
    omega_a = 14.713*sp.rpd()/86400.0
    omega_b = -2.316*sp.rpd()/86400.0
    omega_c = -1.787*sp.rpd()/86400.0

    delta_omega = []

    for (index, item) in enumerate(hci_state[:,0]):
        [void1,void2,lat] = sp.reclat(hci_state[index,0:3])
        omega_sun = omega_a + omega_b*(np.sin(lat)**2) + omega_c*(np.sin(lat)**4)
        omega_sun_d_per_d = omega_sun*sp.dpr()*86400.0
        (r_norm, r_mag) = sp.unorm(hci_state[index,0:3])
        omega_solo = np.cross(hci_state[index,0:3],hci_state[index,3:6])/(r_mag**2)
        omega_solo_d_per_d = omega_solo[2]*sp.dpr()*86400.0
        delta_omega.append(omega_sun_d_per_d-omega_solo_d_per_d)
    return(delta_omega)

def simple_carrington(hci_lon, ets):
    
    rotation_period = 25.38
    base_time = "20 December 2018 11:47 (UTC)"

    omega = (2*np.pi)/(rotation_period*86400.0)
    t0 = sp.str2et(base_time)
    (basepos, ltime) = sp.spkpos("EARTH", t0, "SUN_INERTIAL","NONE","SUN")
    (baserad,baselon,baselat) = sp.reclat(basepos)
    baselon2 = (baselon*sp.dpr()+360.0) % 360
    
    clon = []
    for (index, et) in enumerate(ets):
        delta_t = et - t0
        delta_phi = (delta_t*omega*sp.dpr()) % 360
        zero_lon = (baselon2+delta_phi) % 360
        solo_lon = (hci_lon[index]*sp.dpr()+360) % 360
        clon_buffer = (360+(solo_lon-zero_lon)) % 360
        if clon_buffer > 180:
            clon_buffer = clon_buffer - 360
        clon.append(clon_buffer)
    return(clon)

def calc_true_anomaly(hci_dist, gams, ets):

    et_index  = []         # et index for gams
    ra = np.array([]) # distance aphelion
    rp = np.array([]) # distance perihelion
    e  = np.array([]) # ellipticity before gam
    T  = np.array([]) # true anomaly

    # find ets indices for each gam to access hci_dist data
    for gam in gams:
        et_index.append(np.ravel(np.argwhere(gam[1] > ets))[0])

    for i, eti in enumerate(et_index):
        if i == 0:
            ra = np.append(ra, np.max(hci_dist[:eti]))
            rp = np.append(rp, np.min(hci_dist[:eti]))
        else:
            ra = np.append(ra, np.max(hci_dist[et_index[i-1]:eti]))
            rp = np.append(rp, np.min(hci_dist[et_index[i-1]:eti]))

    a = (ra+rp)/2
    e = (ra-rp)/(ra+rp)
    T = np.degrees(np.arccos(((a/(e*solo.r))*(1-e*e)) - (1/e)))

    #orb_bounds = []
    #orb_end    = np.ravel(np.argwhere(gams > et))[0]
    #orb_start  = np.ravel(np.argwhere(gams > et))[0]-1
    #orb_bounds.append(orb_start, orb_end)
    #print(orb_bounds)
    return T

def carrington_observation_times(clon, hdis, ets, start_time, interval):
    # IMPORTANT NOTE:
    # Calculating the observation times in in 4h intervals on earth
    # requires the synodic carrington rotation period as we don't iterate
    # over the (hopefully) correctly calculated carrington longitudes!

    # clon      ....... carrington longitude for et timesteps
    # hdis      ....... heliospheric distance
    # ets       ....... timestamps for clon/hdis in ephermeris time (spice)
    # start_time....... start time for observations
    # interval  ....... interval between observations in hours

    obs_utc    = np.array([])
    obs_et     = np.array([])
    obs_dist   = np.array([])
    obs_clon   = np.array([])

    rotation_period = 27.2753#25.38 # sidereal period 
    n_obs = int(np.ceil(rotation_period*24)/interval)

    t0 = sp.str2et(start_time)                  # convert start_time to et
    t0_index = np.ravel(np.argwhere(ets == t0)) # find the et index of start_time
 
    if t0_index.size > 0:                       # check if exact timestamp was found
        t0_index = t0_index[0]                  
    else:                                       # first timestamp after time t0
        t0_index = np.ravel(np.argwhere(ets > t0))[0]
    
    clon_t0 = np.interp(t0, ets, clon)          # interpolate clon at time t0
    delta_clon = clon[t0_index] - clon_t0       # clon offset between t0 and nearest entry

    # numpy interp needs the data in increasing order
    # transform alternating clon to linear increment

    # find sign transitions  
    signs = np.sign(clon)
    trans = np.ravel(np.argwhere(np.diff(signs) == 2) + 1)
    
    # linearise between transition points
    clon_lin = np.copy(clon)                 # tracks the clon angle that has elapsed since t0 (positive, increasing)
    for i, pos in enumerate(trans):
        if i == len(trans)-1: 
            clon_lin[trans[i]:] -= np.zeros(len(clon_lin)-trans[i]) + 360 * (i+1)  
        else:
            clon_lin[trans[i] : trans[i+1]] -= np.zeros(trans[i+1]-trans[i]) + 360 * (i+1)            

    clon_lin = clon_lin * -1
    
    # initialise first element of output arrays
    if clon_t0 < 0:
        obs_clon = np.append(obs_clon, clon_t0+360)
    else:
        obs_clon = np.append(obs_clon, clon_t0)

    obs_et   = np.append(obs_et, t0)
    obs_dist = np.append(obs_dist, hdis[t0_index])

    for i in range(1,n_obs):
        
        t = t0 + i*interval*3600 #s
        obs_et = np.append(obs_et, t)
        
        clon_lin_crnt = np.interp(t, ets, clon_lin)                     # current linear clon
        clon_crnt = ((abs(clon_lin_crnt) % 360) -360)*-1 # currenct clon transformed from linear

        #if clon_crnt < 0: clon_crnt += 360   

        obs_clon = np.append(obs_clon, clon_crnt)
        obs_dist = np.append(obs_dist, np.interp(t, ets, hdis))
    
 
    for entry in obs_et:
        obs_utc = np.append(obs_utc, sp.et2utc(entry, 'c', 2))
    
    if False:
        print('###### RESULTS ######')
        output = np.column_stack((obs_utc, obs_clon, obs_dist))
        print(output)

    return [obs_utc, obs_et, obs_clon, obs_dist]



def interp360(clon, clons, ets):

    print((clons[0], clon, np.interp(clon, clons, ets, period=360)))
    # interpolate within one cycle of carrington longitues [0,360]

    # numpy interp needs the data in increasing order
    # transform decreasing uncontiguous clons to linear increment
 
    # find sign transitions  
    signs = np.sign(clons)
    #trans = np.ravel(np.argwhere(np.diff(signs) == 2) + 1) # for -180 to 180
    trans = np.ravel(np.argwhere(np.diff(clons)>0) + 1) # for 0 to 360
    #print(clon)
    #print(trans)
    #print(clons)
    # linearise between transition points
    clons_lin = np.copy(clons)                 # tracks the clon angle that has elapsed since t0 (positive, increasing)
    for i, pos in enumerate(trans):
        if i == len(trans)-1: 
            #clon_lin[:trans[i]] += np.zeros(len(clon_lin)-trans[i]) + 360 * (i+1)  
            clons_lin[trans[i]:] -= np.zeros(len(clons_lin)-trans[i]) + 360 * (i+1)  

    # correct clon if within range that was mapped to negative numbers
    if clon > clons[0]:
        #clon -= 360 
        # eg 321 -> (321-(170-360)) % 180 * -1 = -151
        clon = (clon-(clons[0]-360)) % 180 * -1

    #print(clons_lin)
    print(clons[0], clon, np.interp(clon, clons_lin[::-1], ets[::-1]))

    #set_trace()
    return np.interp(clon, clons_lin[::-1], ets[::-1])





def carrington_observation_times_backup(clon, hdis, ets, start_time, interval):
    # IMPORTANT NOTE:
    # Calculating the observation times in in 4h intervals on earth
    # requires the synodic carrington rotation period as we don't iterate
    # over the (hopefully) correctly calculated carrington longitudes!

    # clon      ....... carrington longitude for et timesteps
    # hdis      ....... heliospheric distance
    # ets       ....... timestamps for clon/hdis in ephermeris time (spice)
    # start_time....... start time for observations
    # interval  ....... interval between observations in hours

    obs_utc    = np.array([])
    obs_et     = np.array([])
    obs_dist   = np.array([])
    obs_clon   = np.array([])

    rotation_period = 27.2753#25.38 # sidereal period 
    n_obs = int(np.ceil(rotation_period*24)/interval)

    t0 = sp.str2et(start_time)                  # convert start_time to et
    t0_index = np.ravel(np.argwhere(ets == t0)) # find the et index of start_time
 
    if t0_index.size > 0:                       # check if exact timestamp was found
        t0_index = t0_index[0]                  
    else:                                       # first timestamp after time t0
        t0_index = np.ravel(np.argwhere(ets > t0))[0]
    
    clon = np.array(clon)
    clon[clon<0] += 360                         # fix negative clon entries for np.interp()
                                                # not enough since the jump from 1 to 359 will be wrong

    clon_t0 = np.interp(t0, ets, clon)          # interpolate clon at time t0
   
    obs_et   = np.append(obs_et, t0)
    obs_dist = np.append(obs_dist, hdis[t0_index])
    obs_clon = np.append(obs_clon, clon_t0)

    for i in range(1,n_obs):

        t = t0 + i*interval*3600 #s
        obs_et = np.append(obs_et, t)
        
        obs_clon = np.append(obs_clon, np.interp(t, ets, clon))
        obs_dist = np.append(obs_dist, np.interp(t, ets, hdis))
    
    for i, val in enumerate(obs_clon):
        if val < 0:
            obs_clon[i] = obs_clon[i] + 360

    for entry in obs_et:
        obs_utc = np.append(obs_utc, sp.et2utc(entry, 'c', 2))
    
    if False:
        print('###### RESULTS ######')
        output = np.column_stack((obs_utc, obs_clon, obs_dist))
        print(output)

    return [obs_utc, obs_et, obs_clon, obs_dist]

def carrington_observation_deg_old1(clon, hdis, ets, start_time, delta_phi):

    # clon      ....... carrington longitude for et timesteps
    # hdis      ....... heliospheric distance
    # ets       ....... timestamps for clon/hdis in ephermeris time (spice)
    # start_time....... start time for observations
    # delta_phi ....... interval between observations in hours

    obs_utc    = np.array([])
    obs_et     = np.array([])
    obs_dist   = np.array([])
    obs_clon   = np.array([])

    n_obs = int(np.ceil(360. / delta_phi))
    
    t0 = sp.str2et(start_time)                  # convert start_time to et
    t0_index = np.ravel(np.argwhere(ets == t0)) # find the et index of start_time
 
    if t0_index.size > 0:                       # check if exact timestamp was found
        t0_index = t0_index[0]                  
    else:                                       # first timestamp after time t0
        t0_index = np.ravel(np.argwhere(ets > t0))[0]
        
    clon_t0 = np.interp(t0, ets, clon)          # interpolate clon at time t0
    delta_clon = clon[t0_index] - clon_t0       # clon offset between t0 and nearest entry
    
    # numpy interp needs the data in increasing order
    # transform alternating clon to linear increment

    # find sign transitions  
    signs = np.sign(clon)
    trans = np.ravel(np.argwhere(np.diff(signs) == 2) + 1)
    
    # why did I want this in a linear representation?
    # linearise between transition points
    clon_linear = np.copy(clon)                 # tracks the clon angle that has elapsed since t0 (positive, increasing)
    for i, pos in enumerate(trans):
        if i == len(trans)-1: 
            clon_linear[trans[i]:] -= np.zeros(len(clon_linear)-trans[i]) + 360 * (i+1)  
        else:
            clon_linear[trans[i] : trans[i+1]] -= np.zeros(trans[i+1]-trans[i]) + 360 * (i+1)            

    clon_linear = clon_linear * -1

    obs_et   = np.append(obs_et, t0)
    obs_dist = np.append(obs_dist, hdis[t0_index])
    
    if clon_t0 < 0:
        obs_clon = np.append(obs_clon, clon_t0+360)
    else:
        obs_clon = np.append(obs_clon, clon_t0)
    
    for i in range(1,n_obs):

        clon_obs_lin = clon_linear[t0_index] + delta_clon + delta_phi * i
        obs_et   = np.append(obs_et,   np.interp(clon_obs_lin, clon_linear, ets))
        obs_dist = np.append(obs_dist, np.interp(clon_obs_lin, clon_linear, hdis))
        
        clon_obs = clon[t0_index] - delta_clon - delta_phi * i
        
        # make sure clon_obs does not exceed 360 deg. Sign adjusted later
        clon_obs = (abs(clon_obs) % 360) * np.sign(clon_obs)

        if clon_obs < 0: clon_obs += 360    
        obs_clon = np.append(obs_clon, clon_obs)
        #obs_clon = np.append(obs_clon, clon_obs-clon_linear[t0_index]+clon_t0) # observed clon in linear increasing longitude (>360°)

    for entry in obs_et:
        obs_utc = np.append(obs_utc, sp.et2utc(entry, 'c', 2))
    
    if False:
        print('###### RESULTS ######')
        output = np.column_stack((obs_utc, obs_clon, obs_dist))
        print(output)

    return [obs_utc, obs_et, obs_clon, obs_dist]


def carrington_observation_deg(clon, hdis, ets, delta_phi, lld_start, lld_end=None, rsw_start=None, rsw_end=None):

    # clon      ....... carrington longitude for et timesteps
    # hdis      ....... heliospheric distance
    # ets       ....... timestamps for clon/hdis in ephermeris time (spice)
    # lld_start ....... start time for synoptic maps observations (can be within RSW)
    # rsw_start ....... start time for RSW observations
    # rsw_end   ....... end time for RSW observations
    # delta_phi ....... interval between observations in hours

    obs_utc    = np.array([])
    obs_et     = np.array([])
    obs_dist   = np.array([])
    obs_clon   = np.array([])

    #rotation_period = 25.38 # sidereal period 
    #delta_phi = 360/(rotation_period * 24) * interval
    n_obs = int(np.ceil(360. / delta_phi))

    if rsw_start:
        t0 = max([sp.str2et(rsw_start), sp.str2et(lld_start)]) # select later time (eg lld_start within RSW)
        t1 = sp.str2et(rsw_end)                    # convert end_time to et
    else:
        # no RSW specified, use LLD start time
        t0 = sp.str2et(lld_start)   
        t1 = sp.str2et(lld_end)

    t0_index = np.ravel(np.argwhere(ets == t0)) # find the et index of start_time
 
    if t0_index.size > 0:                       # check if exact timestamp was found
        t0_index = t0_index[0]                  
    else:                                       # first timestamp after time t0
        t0_index = np.ravel(np.argwhere(ets > t0))[0]
        
    clon_t0 = np.interp(t0, ets, clon)          # interpolate clon at time t0
    delta_clon = clon[t0_index] - clon_t0       # clon offset between t0 and nearest entry

    # numpy interp needs the data in increasing contiguous order
    # transform alternating clon to linear increment

    # find sign transitions  
    signs = np.sign(clon)
    trans = np.ravel(np.argwhere(np.diff(signs) == 2) + 1)
    
    # linearise between transition points
    clon_linear = np.copy(clon)                 # tracks the clon angle that has elapsed since t0 (positive, increasing)
    for i, pos in enumerate(trans):
        if i == len(trans)-1: 
            clon_linear[trans[i]:] -= np.zeros(len(clon_linear)-trans[i]) + 360 * (i+1)  
        else:
            clon_linear[trans[i] : trans[i+1]] -= np.zeros(trans[i+1]-trans[i]) + 360 * (i+1)            

    clon_linear = clon_linear * -1

    obs_et   = np.append(obs_et, t0)
    obs_dist = np.append(obs_dist, hdis[t0_index])
    
    if clon_t0 < 0:
        obs_clon = np.append(obs_clon, clon_t0+360)
    else:
        obs_clon = np.append(obs_clon, clon_t0)

    i = 1    
    current_et = obs_et[0]
    
    while current_et < t1:
        # must be +delta_clon since clon_linear is always increasing
        # the original delta was measured on a decreasing slope
        clon_obs_lin = clon_linear[t0_index] + delta_clon + delta_phi * i

        current_et   = np.interp(clon_obs_lin, clon_linear, ets)
        current_hdis = np.interp(clon_obs_lin, clon_linear, hdis)

        obs_et   = np.append(obs_et,   current_et)
        obs_dist = np.append(obs_dist, current_hdis)
        
        clon_obs = clon[t0_index] - delta_clon - delta_phi * i
        
        # make sure clon_obs does not exceed 360 deg. Sign adjusted later
        clon_obs = (abs(clon_obs) % 360) * np.sign(clon_obs)

        if clon_obs < 0: clon_obs += 360    
        obs_clon = np.append(obs_clon, clon_obs)
        #obs_clon = np.append(obs_clon, clon_obs-clon_linear[t0_index]+clon_t0) # observed clon in linear increasing longitude (>360°)

        i += 1


    for entry in obs_et:
        obs_utc = np.append(obs_utc, sp.et2utc(entry, 'c', 2))
    

    if False:
        print('###### RESULTS ######')
        print(np.column_stack((obs_utc, obs_clon, obs_dist)))

    return [obs_utc, obs_et, obs_clon, obs_dist]


def merge_lld_rsw_observation_times(obs_lld, obs_rsw):
    
    # unpack obs_lld and obs_rsw
    [lld_utc, lld_et, lld_clon, lld_hdis] = obs_lld
    [rsw_utc, rsw_et, rsw_clon, rsw_hdis] = obs_rsw

    # define output arrays
    utc  = np.array([])
    et   = np.array([])
    clon = np.array([])
    hdis = np.array([])


    lower = np.where(rsw_et[0]  > lld_et)[0]    # find LLD indices before RSW
    upper = np.where(rsw_et[-1] < lld_et)[0]    # find LLD indices after RSW

    print(lower)
    print(upper)

    
    if len(lower) == 0:     
        # case lld_et[0] == rsw_et[0]: start in RSW 
        #   discard LLD entries before RSW end time lld_et < rsw_et[-1]
        #   append reduced LLD to RSW set
        utc  = np.append(rsw_utc,  lld_utc[upper])
        et   = np.append(rsw_et,   lld_et[upper])
        clon = np.append(rsw_clon, lld_clon[upper])
        hdis = np.append(rsw_hdis, lld_hdis[upper])

        # src: 0 RSW, 1 LLD, 2 HMI
        src  = np.zeros(len(utc))
        n_rsw = len(rsw_utc)
        src[n_rsw:] = 1

    else: 
        # case lld_et[0] < rsw_et[0]: start before RSW 
        #   discard LLD entries rsw_et[0] < lld_et[:] < rsw_et[-1]
        #   replace with RSW set
        
        #center = np.arange(lower[-1], upper[0])[1:] # lld indices to be replaced by rsw

        utc = np.append(utc, lld_utc[lower])
        utc = np.append(utc, rsw_utc)
        utc = np.append(utc, lld_utc[upper])

        et = np.append(et, lld_et[lower])
        et = np.append(et, rsw_et)
        et = np.append(et, lld_et[upper])

        clon = np.append(clon, lld_clon[lower])
        clon = np.append(clon, rsw_clon)
        clon = np.append(clon, lld_clon[upper])

        hdis = np.append(hdis, lld_hdis[lower])
        hdis = np.append(hdis, rsw_hdis)
        hdis = np.append(hdis, lld_hdis[upper])

        # src: 0 RSW, 1 LLD, 2 HMI
        src  = np.zeros(len(utc))
        n_lld1 = len(lower)
        n_rsw  = len(utc) - len(lower) - len(upper)
        n_lld2 = len(upper)

        src[:n_lld1] = 1
        src[n_lld1+n_rsw:] = 1

    # case lld_et[0] > rsw_et[0]: not possible (rsw_et cropped to start times)

    return [utc, et, clon, hdis, src]


def carrington_observation_deg_old2(clon, solo_hdis, ets):

    rotation_period = 25.38
    #start_time = "20 December 2020 11:47 (UTC)" # start of observation for carrington rotation
    #start_time = "15 March 2022 11:47 (UTC)" # LTP05 during high omega
    start_time = "3 April 2022 19:24 (UTC)"   # LTP05 during high omega CR2256 start time
    
    obs_utc  = np.array([])
    obs_et   = np.array([])
    obs_dist = np.array([])
    obs_clon = np.array([])
    delta_t  = 86400                  # 24h, temporal spacing between observations on carrington grid
    n_obs    = 30
    
    t0 = sp.str2et(start_time)
    t0_index = np.ravel(np.argwhere(ets == t0))

    if t0_index.size > 0:                       # check if exact timestamp was found
        t0_index = t0_index[0]                  
    else:                                       # first timestamp after time t0
        t0_index = np.ravel(np.argwhere(ets > t0))[0]
        
    clon_t0 = np.interp(t0, ets, clon)          # interpolate clon at time t0

    obs_et   = np.append(obs_et, t0)
    obs_clon = np.append(obs_clon, clon_t0)
    
    for i in range(1, n_obs):
        t = t0 + i*delta_t
        obs_et = np.append(obs_et, t)
        
        obs_clon = np.append(obs_clon, np.interp(t, ets, clon))
        obs_dist = np.append(obs_dist, np.interp(t, ets, solo_hdis))
    
    for i, val in enumerate(obs_clon):
        if val < 0:
            obs_clon[i] = obs_clon[i] + 360
            
    for et in obs_et:
        obs_utc = np.append(obs_utc, sp.et2utc(et, 'c', 2))
    
    print()
    print('###### CONSTANT TIME #####')
    print('CLON OBS')
    print(obs_clon)
    print('OBS DIST')
    print(obs_dist)
    print('OBS UTC')
    print(obs_utc)
    print()
    return obs_utc, obs_et, obs_clon


def carrington_observation_duration(solo_clon, earth_clon, solo_hdis, ets):

    start_time = "1 January 2022 00:00 (UTC)"   # LTP05 during high omega CR2256 start time
    dt = np.array([])
    dt_date = np.array([])
    
    t0_index = 0
    ii = 0
    
    for i, time in enumerate(ets):

        t0 = sp.str2et(start_time) + i*21600    # loop over every 1/4 day in mission duration
        t0_index = np.ravel(np.argwhere(ets == t0))
        t = t0
        
        if t0_index.size > 0:                       # check if exact timestamp was found
            t0_index = t0_index[0]                  
        else:                                       # first timestamp after time t0
            try:
                t0_index = np.ravel(np.argwhere(ets > t0))[0]
            except:
               break

        if i == 0:
            ii = t0_index

        coverage = np.zeros(360)
        

        for i, et in enumerate(ets[t0_index:]):

            eclon = int(earth_clon[t0_index+i])
            sclon = int(solo_clon[t0_index+i])

            # eclon and sclon are always decreasing after this operation
            if eclon < 0:
                eclon = eclon + 360

            if sclon < 0:
                sclon = sclon + 360
            
            if i > 0:
                if prev_eclon - eclon < 0:
                    # if we jump over zero fill in both sides of the coverage array
                    # 3xx:360 and 0:xx
                    coverage[0:prev_eclon+1] = 1
                    coverage[eclon:361] = 1
                    prev_eclon = eclon
                    
                elif prev_eclon == 0:
                    coverage[eclon:361] = 1
                    prev_eclon = eclon

                else:
                    coverage[eclon:prev_eclon+1] = 1
                    prev_eclon = eclon
                
                if prev_sclon - sclon < 0:
                    coverage[0:prev_sclon+1] = 1
                    coverage[sclon:361] = 1
                    prev_sclon = sclon

                elif prev_sclon == 0:
                    coverage[sclon:361] = 1
                    prev_sclon = sclon

                else:
                    coverage[sclon:prev_sclon+1] = 1
                    prev_sclon = sclon
                
                #print(i, et, eclon, prev_eclon, sclon, prev_sclon, np.sum(coverage))
            else:
                coverage[eclon] = 1
                coverage[sclon] = 1
                #print(i, et, eclon, eclon, sclon, sclon, np.sum(coverage))
                prev_eclon = eclon
                prev_sclon = sclon

            if np.sum(coverage) == 360:
                t = ets[t0_index+i]
                dt = np.append(dt, (t - t0)/86400) # days
                dt_date = np.append(dt_date, sp.et2utc(t0, 'C', 3))
                break

    print('min creation time: ', np.min(dt))
    print('max creation time: ', np.max(dt))
    print('avg creation time: ', np.average(dt))
    
    fsm_dates = np.where(dt < 16.5)[0]

    #fsm_dates = np.where(dt[fsm_dates] >14)[0]
    
    #for i in fsm_dates:
        #print(dt_date[i], dt[i])
    # 5+ to start at ltp 5 / *2 since ltp are half yearly
    ltp = 5+(ets[ii:ii+len(dt)]-ets[ii])/86400/365 * 2

    fig, ax = plt.subplots(figsize=(16,9), linewidth=20, edgecolor='#930534')
    ax.plot(ltp, dt, color='#003247', linewidth=4)  #003247 930534

    text_style = dict(fontsize=14)

    ax.set_title('Synoptic Map Observation Duration', y=1.05, fontsize=20)
    ax.set_xlabel('LTP Period',**text_style)
    ax.set_ylabel('Completion Time [Days]',**text_style)

    ax.tick_params(labelsize=12)

    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10

    fig.subplots_adjust(left=0.1,right=0.9,top=0.85,bottom=0.15)
    plt.savefig('./plots/fsm_observation_duration.png', dpi=120, edgecolor=fig.get_edgecolor())
    

    return dt


def carrington_observation_coverage(solo_obs, earth_obs, plot=False):

    
    # unpack obs_lld and obs_rsw
    [solo_utc,  solo_et,  solo_clon,  solo_hdis, solo_src]  = solo_obs
    [earth_utc, earth_et, earth_clon, earth_hdis] = earth_obs
    

    earth_src = np.zeros(len(earth_utc)) + 2    # src: 0 RSW, 1 LLD, 2 HMI

    utc   = np.concatenate([solo_utc,  earth_utc])
    ets   = np.concatenate([solo_et,   earth_et])
    clons = np.concatenate([solo_clon, earth_clon])
    hdis  = np.concatenate([solo_hdis, earth_hdis])
    src   = np.concatenate([solo_src,  earth_src])


    # The resulting array contains the et ordered solo values before the earth values
    # This has to be ordered for increasing et again. The new order is determined with
    # the np.argsort function and applied to all 5 arrays

    order = np.argsort(ets)
 
    utc   = utc[order]
    ets   = ets[order]
    clons = clons[order]
    hdis  = hdis[order]
    src   = src[order]

    #return [utc, ets, src, order]

    #start_time = "1 January 2022 00:00 (UTC)"   # LTP05 during high omega CR2256 start time
    dt = np.array([])

    coverage       = np.zeros(360)
    coverage_earth = np.zeros(360)
    coverage_solo  = np.zeros(360)
    coverage_src   = np.zeros(360)
    coverage_track = []

    # New version cannot iterate over general ets since there are entries
    # for different INTERPOLATED ets for solo and hmi. Therefore, 
    # merge solo_obs and earth_obs while tracking the obsevation source
    # (add src array = 2 to earth_obs) and then iterate over all ets
    # until the map is full. Discard remaining data once the map is full.
    # Maybe save single spacecraft coverage history for separate use?
    # This also provides a temporal evolution of the synoptic map creation
    # and can be used for future animations and the clon source plot
    # Then create a nice clon source plot and a script for the temporal
    # evolution with single frames for each observation
    
    eclon = 0
    sclon = 0
    prev_eclon = None
    prev_sclon = None

    for i, et in enumerate(ets):
        #print(i, clons[i], src[i])
        coverage_src   = np.zeros(360)-1

        if src[i] < 2:
            clon = int(clons[i])
            prev_clon = prev_sclon
        else:
            clon = int(clons[i])
            prev_clon = prev_eclon

        #print(i, et, coverage)

        if prev_clon is None:
            coverage[clon] = 1

            coverage_src[clon] = src[i]
            coverage_track.append(coverage_src)

            #print(i, et, eclon, eclon, sclon, sclon, np.sum(coverage))
            if src[i] < 2:
                prev_sclon = clon
            else:
                prev_eclon = clon

        else:
            if prev_clon - clon < 0:
                coverage[0:prev_clon+1] = 1
                coverage[clon:361] = 1

                coverage_src[0:prev_clon+1] = src[i]
                coverage_src[clon:361] = src[i]
                coverage_track.append(coverage_src)

                if src[i] < 2:
                    prev_sclon = clon
                else:
                    prev_eclon = clon
            
            elif prev_clon == 0:
                coverage[clon:361] = 1

                coverage_src[clon:361] = src[i]
                coverage_track.append(coverage_src)

                if src[i] < 2:
                    prev_sclon = clon
                else:
                    prev_eclon = clon

            else:
                coverage[clon:prev_clon+1] = 1

                coverage_src[clon:prev_clon+1] = src[i]
                coverage_track.append(coverage_src)

                if src[i] < 2:
                    prev_sclon = clon
                else:
                    prev_eclon = clon
     
            
            #print(i, et, eclon, prev_eclon, sclon, prev_sclon, np.sum(coverage))


        if np.sum(coverage) == 360:
            dt = np.append(dt, (et - ets[0])/86400) # days
            break

    print('FSM Creation Time: %s days'% np.round(dt[0],2))
    nobs =len(coverage_track)
    if plot:
        carrington_coverage_plot(coverage_track, utc[:nobs], ets[:nobs])

    if False:
        fig, ax = plt.subplots()
        ax.scatter(range(len(coverage_src)), coverage_src)#, label='HMI')
        #ax.scatter(range(len(coverage_solo)),  coverage_solo,  label='PHI')
        ax.set_xlabel('Carrington Longitude')
        ax.set_ylabel('Coverage')
        
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles, labels)
        plt.show()
    
    return [coverage, coverage_track, utc, ets, clons, hdis, src, order, nobs]


def carrington_coverage_plot(coverage_track, utc, ets):
    #ffmpeg -r 5 -f image2 -i coverage_%03d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 5 fsm_observation_hd.mp4

    texture, keys = get_synoptic_map_data()

    n = len(coverage_track[0])
    nn= len(coverage_track)
    coverage = np.zeros(len(coverage_track[0]))-1

    # load the color map
    colors = np.loadtxt('HMI_MagColor_256.txt')
    colormap_customized = mcol.ListedColormap(colors)

    # setup ticks
    ytick_latitude = []
    ytick_normalize = []
    for i in range(19):
        calculation = math.sin((np.pi/18)*(i-9.0))
        ytick_latitude.append(calculation)
        ytick_normalize.append((calculation+1)*720.)


 

    for i, src in enumerate(coverage_track):
        ii = np.where(src>=0)
        coverage[ii] = src[ii]
        
        x = np.arange(0, n*10)
        y = np.zeros([n])
        y_hmi = np.zeros(n)
        y_lld = np.zeros(n)
        y_rsw = np.zeros(n)

        ii = np.where(coverage == 0)
        jj = np.where(coverage == 1)
        kk = np.where(coverage == 2)
        
        y_hmi[kk] = 1
        y_lld[jj] = 1
        y_rsw[ii] = 1

        y[kk] = 1
        y[jj] = 1
        y[ii] = 1

         # upscale to 3600 texture resolution
        yy     = sample_coverage2texture_size(y)
        yy_hmi = sample_coverage2texture_size(y_hmi)
        yy_lld = sample_coverage2texture_size(y_lld)
        yy_rsw = sample_coverage2texture_size(y_rsw)

        ll = np.ravel(np.argwhere(yy!=1))
        yy[ll]= np.nan

        #print('###############################')
        #print(i)
        #print(y)
        
        # plot synoptic map
        #fig, ax = plt.subplots(figsize=(16,9), nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [6, 1]})
        fig = plt.figure(figsize=(16,9))
        widths = [94.3, 5.7]
        heights = [6, 1]
        spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
        spec.update(wspace=0.0, hspace=-0.15) # set the spacing between axes. 

        ax1  = fig.add_subplot(spec[0,:])
        ax2  = fig.add_subplot(spec[1,0])

        fig.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.125)

        fig.suptitle('Fast Synoptic Map Observation', size=20, y=0.925)

        text_style = dict(fontsize=16)
        ax1.tick_params(labelsize=14)
        #ax1.set_title('HMI $B_{r}$ Synoptic Chart for Carrington Rotation '+str(keys.CAR_ROT[0]),y=1.1,fontsize=20)
        ax1.tick_params(axis='both', which='both', bottom=True, labelbottom=False, labeltop=False, labelleft=True, labelright=False)
        sm1=ax1.imshow(texture*np.nan,cmap=colormap_customized,vmin=-1500,vmax=1500,origin='lower',extent=[0,3600,0,1440],interpolation='nearest')
                    
        # Create the latitude labels on the right-hand side of the plot
        ylabels_r = [' ','-80',' ','-60',' ','-40',' ','-20',' ','0',' ',' 20',' ',' 40',' ',' 60',' ',' 80',' ']
        ylocations_r = ytick_normalize
        ax1.set_yticklabels(ylabels_r)
        ax1.set_yticks(ylocations_r)
        ax1.set_ylabel('Latitude',**text_style)

        ax1.tick_params(labelsize=14, pad=5)
        #ax1.yaxis.set_label_position("left")
        #ax1.yaxis.tick_right()


        
        #ax2.set_xlabel('Observed Carrington Longitude')
        ax2.set_xlim(0,3600)
        ax2.set_ylim(0,1)
            
        # label the x-axis 
        xlabels = [0,60,120,180,240,300,360]
        xlocations = [0,600,1200,1800,2400,3000,3600]
        ax2.set_xticklabels(xlabels)
        ax2.set_xticks(xlocations)
        ax2.tick_params(labelsize=14)
        ax2.set_xlabel('Carrington Longitude',**text_style)
        ax2.xaxis.labelpad=10
        ax2.tick_params(axis='y', colors='white')
        #ax2.yaxis.labelpad=10
        #ax2.yaxis.set_visible(False)
        ax2.set_ylabel('Source',**text_style)

        x = np.arange(0, n*10)
        yy_dummy = np.zeros(3600)
        ax2.bar(x, yy_dummy, facecolor='dodgerblue', width=1.0, label='HMI')
        ax2.bar(x, yy_dummy, facecolor='firebrick', width=1.0, label='PHI-LLD')
        ax2.bar(x, yy_dummy, facecolor='gold', width=1.0, label='PHI-RSW')


        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, labels, bbox_to_anchor=(1.04, 0.792), loc='upper left', borderaxespad=0.)

        ax2.annotate('Loeschl et al.', xy=(0,0), xytext=(0.9075, 0.025), annotation_clip=False, size=12, textcoords='figure fraction',)

        # plot the colorbar
        divider_cbar = make_axes_locatable(ax1)
        width_cbar = axes_size.AxesY(ax1, aspect=0.05)
        pad_cbar = axes_size.Fraction(2, width_cbar)
        cax = divider_cbar.append_axes("right", size=width_cbar, pad=pad_cbar)
        cax.tick_params(labelsize=14)
        cbar = fig.colorbar(sm1,cax=cax)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.labelpad=-25
        cbar.set_label('$B_{r}$ [Gauss]', size=16)


        time = np.round((ets[i]-ets[0])/86400., 2)
        ax1.set_title('Observation %s/%s: %s - Elapsed Time %s days' %(i+1, nn-1, utc[i], format(time, '.2f')), size=14)
        sm1=ax1.imshow(texture*yy,cmap=colormap_customized,vmin=-1500,vmax=1500,origin='lower',extent=[0,3600,0,1440],interpolation='nearest')
        
                
        #plot synoptic map coverage
        ax2.bar(x, yy_hmi, facecolor='dodgerblue', width=1.0, label='HMI')
        ax2.bar(x, yy_lld, facecolor='firebrick',  width=1.0, label='PHI-LLD')
        ax2.bar(x, yy_rsw, facecolor='gold',       width=1.0, label='PHI-RSW')

        print('Saving Coverage Step %s...' % i)

        if i < 10:
            istr = '00'+str(i)
        elif i < 100: 
            istr = '0'+str(i)
        else: 
            istr = str(i)
        fname = './plots/coverage_'+istr+'.png'
        #plt.show()
        plt.savefig(fname, dpi=240)
        #plt.show()

        plt.close()
        

def sample_coverage2texture_size(y):
    
    # y arrives with 360 entries
    # sample to 3600
    yy = np.zeros(3600)
    for i in range(len(y)):
        yy[10*i:10*(i+1)] = y[i]

    return yy



def carrington_coverage_plot_no_texture(coverage_track, utc, ets):
    #ffmpeg -r 5 -i coverage_%03d.png fsm_observation.mp4     
    
    n = len(coverage_track[0])
    nn= len(coverage_track)
    coverage = np.zeros(len(coverage_track[0]))-1

    for i, src in enumerate(coverage_track):

        ii = np.where(src>=0)
        coverage[ii] = src[ii]
        
        x = np.arange(0, n)
        y_hmi = np.zeros(n)
        y_lld = np.zeros(n)
        y_rsw = np.zeros(n)

        ii = np.where(coverage == 0)
        jj = np.where(coverage == 1)
        kk = np.where(coverage == 2)
        
        y_hmi[kk] = 1
        y_lld[jj] = 1
        y_rsw[ii] = 1

        #print(y_hmi)

        #print('###############################')
        
        time = np.round((ets[i]-ets[0])/86400., 2)
        print('Saving Coverage Step %s...' % i)
        fig, ax = plt.subplots(figsize=(20,5))
        ax.bar(x, y_hmi, facecolor='dodgerblue', width=1.0, label='HMI')
        ax.bar(x, y_lld, facecolor='firebrick', width=1.0, label='LLD')
        ax.bar(x, y_rsw, facecolor='gold', width=1.0, label='RSW')
        
        fig.suptitle('Fast Synoptic Map Coverage')
        ax.set_title('Observation %s/%s: %s - Elapsed Time %s days' %(i+1, nn-1, utc[i], format(time, '.2f')))
        ax.set_xlabel('Observed Carrington Longitude')
        #ax.set_ylabel('Observed')
        ax.set_xlim(0,360)
        ax.set_ylim(0,1)
            
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=1)

        if i < 10:
            istr = '00'+str(i)
        elif i < 100: 
            istr = '0'+str(i)
        else: 
            istr = str(i)
        fname = './plots/coverage_'+istr+'.png'
        #plt.show()
        plt.savefig(fname)
        plt.close()

def carrington_observation_coverage_old(solo_clon, earth_clon, solo_hdis, ets):

    start_time = "1 January 2022 00:00 (UTC)"   # LTP05 during high omega CR2256 start time
    dt = np.array([])

    t0 = sp.str2et(start_time)  + 5060*21600    # loop over every 1/4 day in mission duration
    t0_index = np.ravel(np.argwhere(ets == t0))
    t = t0
    
    if t0_index.size > 0:                       # check if exact timestamp was found
        t0_index = t0_index[0]                  
    else:                                       # first timestamp after time t0
        t0_index = np.ravel(np.argwhere(ets > t0))[0]

    coverage = np.zeros(360)
    coverage_earth = np.zeros(360)
    coverage_solo  = np.zeros(360)

    for i, et in enumerate(ets[t0_index:]):

        eclon = int(earth_clon[t0_index+i])
        sclon = int(solo_clon[t0_index+i])

        if eclon < 0:
            eclon = eclon + 360

        if sclon < 0:
            sclon = sclon + 360

        if i > 0:
            
            if prev_eclon - eclon < 0:
                coverage[0:prev_eclon+1] = 1
                coverage[eclon:361] = 1
                coverage_earth[0:prev_eclon+1] = 1
                coverage_earth[eclon:361] = 1
                prev_eclon = eclon
            
            elif prev_eclon == 0:
                coverage[eclon:361] = 1
                coverage_earth[eclon:361] = 1
                prev_eclon = eclon

            else:
                coverage[eclon:prev_eclon+1] = 1
                coverage_earth[eclon:prev_eclon+1] = 1
                prev_eclon = eclon
            
            if prev_sclon - sclon < 0:
                coverage[0:prev_sclon+1] = 1
                coverage[sclon:361] = 1                
                coverage_solo[0:prev_sclon+1] = 1
                coverage_solo[sclon:361] = 1
                prev_sclon = sclon

            elif prev_sclon == 0:
                coverage[sclon:361] = 1
                coverage_solo[sclon:361] = 1
                prev_sclon = sclon

            else:
                coverage[sclon:prev_sclon+1] = 1
                coverage_solo[sclon:prev_sclon+1] = 1                                
                prev_sclon = sclon
            
            #print(i, et, eclon, prev_eclon, sclon, prev_sclon, np.sum(coverage))
        else:
            coverage[eclon] = 1
            coverage[sclon] = 1
            coverage_earth[eclon] = 1
            coverage_solo[sclon] = 1
            #print(i, et, eclon, eclon, sclon, sclon, np.sum(coverage))
            prev_eclon = eclon
            prev_sclon = sclon

        if np.sum(coverage) == 360:
            t = ets[t0_index+i]
            dt = np.append(dt, (t - t0)/86400) # days
            break

    print('min creation time: ', np.min(dt))
    print('max creation time: ', np.max(dt))
    print('avg creation time: ', np.average(dt))
    """
    fig, ax = plt.subplots()
    ax.scatter(range(len(coverage_earth)), coverage_earth, label='HMI')
    ax.scatter(range(len(coverage_solo)), coverage_solo, label='PHI')
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Coverage')
    ax.set_ylim(0.9, 1.1)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    #plt.show()
    """
    return coverage_solo, coverage_earth


def daily_avg_downlink(filename, eventname, zones):
    tree = xmlet.parse(filename)
    root = tree.getroot()
    ns={"namespace":"http://soc.solarorbiter.org/sifecs"}
    passes = root.findall("namespace:events/namespace:"+eventname,namespaces=ns)
    pass_start = []
    pass_duration = []
    pass_rate = []
    pass_end = []

    for txon in passes:
        pass_rate.append(int(txon.get("tm_rate")))
        pass_duration.append(int(txon.get("duration")))
        time = txon.get("time")
        time_trunc = time[:len(time)-1]
        pass_start.append(sp.str2et(time_trunc))
        pass_end.append(sp.str2et(time_trunc)+float(txon.get("duration")))
    effective_start = []
    effective_rate = []
    pass_zone = []
    first_start = datetime642et(np.datetime64(et2datetime64(pass_start[0])[0],"D"))[0]    
    effective_start.append(first_start)
    for index, rate in enumerate(pass_rate):
        if index > 0:
            effective_start.append(pass_end[index-1])
        pass_vol = rate*pass_duration[index]
        effective_duration = pass_end[index]-effective_start[index]
        effective_rate.append(pass_vol/effective_duration)
        if rate >= int(zones["0"]): pass_zone.append(0)
        elif effective_rate[index] < int(zones["0"]) and effective_rate[index] >= int(zones["1"]): pass_zone.append(1)
        elif effective_rate[index] < int(zones["1"]) and effective_rate[index] >= int(zones["2"]): pass_zone.append(2)
        elif effective_rate[index] < int(zones["2"]): pass_zone.append(3)
    
    output = {"pass_start":pass_start, "pass_end":pass_end, "pass_rate":pass_rate, "effective_start":effective_start, "effective_rate":effective_rate, "pass_zone":pass_zone}            
    return output


def solo2earth_times(clons, clons_earth, ets, ets_earth , src):
    """ Locate Carrington longitudes observed by PHI and find the timeslot of the
        corresponding HMI observation. This function will extrend the fast synoptic 
        observation time a standard HMI period. """

    utc = np.array([])
    ip = np.ravel(np.argwhere(src < 2)) # indices of phi observations

    for i in ip:

        et_earth = np.interp(clons[i], clons_earth, ets_earth, period=360)
        #et_earth = interp360(clons[i], clons_earth, ets_earth)
        ets[i] = et_earth

    for et in ets:
        utc = np.append(utc, sp.et2utc(et, 'c', 2))
        
    return ets, utc


def obs2drms_times(start_time, target_time, utc):

    "Converts input UTC to TAI and Spice compatible time string into a DRMS time string"

    start_time =  "18 April 2022 03:00 (UTC)"
    target_time = "04 May 2014 14:42 (UTC)"        #"2014.05.14_14:46:00_TAI-2014.05.16_02:46:00_TAI"
    #target_time = "21 April 2014 16:23:23 (UTC)"    #2014.04.21_16:24:00_TAI	170.937988 
                                                    #start at -37s for correct TAI conversion

    months = dict([('JAN', '01'), ('FEB', '02'), ('MAR', '03'), ('APR', '04'), ('MAY', '05'), ('JUN', '06'), 
                   ('JUL', '07'), ('AUG', '08'), ('SEP', '09'), ('OCT', '10'), ('NOV', '11'), ('DEC', '12')]) 

    ets = np.array([])

    for t in utc:
        ets = np.append(ets, sp.utc2et(t))
    
    start_et = sp.utc2et(start_time)
    target_et = sp.utc2et(target_time)

    shift = start_et - target_et

    # TAI is exactly 37 seconds ahead of UTC. The 37 seconds results from the initial difference 
    # of 10 seconds at the start of 1972, plus 27 leap seconds in UTC since 1972. 
    tai_et  = ets - shift + 37

    tai_str = np.array([])
    fmt = "%s.%s.%s_%s:%s:%s_TAI"

    for tai in tai_et:

        # convert "14 May 2014 14:46 (UTC)" to "2014.05.14_14:46:00_TAI"
        t = sp.et2utc(tai,"C",3)
        tai = fmt % (t[:4], months[t[5:8]], t[9:11], t[12:14], t[15:17], t[18:20])
        tai_str = np.append(tai_str, tai)

    # src: 0 RSW, 1 LLD, 2 HMI

    return tai_et, tai_str


def drms_timestring(tai_str, tai_ets, src, clons):

    """ Shortens the list of DRMS compatible TAIs to a single time period string"""

    order = np.argsort(tai_ets)
    tai_str = tai_str[order]
    tai_ets = tai_ets[order]
    clons   = clons[order]
    src     = src[order]

    trans = np.ravel(np.argwhere(np.diff(src)!=0)+1) # count from 0 to trans[0] for first set
    trans = np.append(trans, len(src))               # cover entries between last transition and end

    drms_str = ''
    months = dict([('JAN', '01'), ('FEB', '02'), ('MAR', '03'), ('APR', '04'), ('MAY', '05'), ('JUN', '06'), 
                   ('JUL', '07'), ('AUG', '08'), ('SEP', '09'), ('OCT', '10'), ('NOV', '11'), ('DEC', '12')]) 

    fmt = "%s.%s.%s_%s:%s:%s_TAI"
    

    # src: 0 RSW, 1 LLD, 2 HMI
    # hmi: time1-time2
    # rsw: time1,time2,time3,...
    # lld: time1,time2,time3,...
     
    low = 0

    for up in trans:
        #print(up-low)
        if src[low] == 2:
            drms_str += tai_str[low] + '-' + tai_str[up-1] + ','

        else:
            # this if skips overlapping single LLD obesrvations that crawl
            # into earlier HMI observations
            # TODO might be useful wiht real images!
            if up-low == 1: 
                # only single observation. Check if between HMI observation
                if src[up]== 2 and src[low-1] == 2:
                    low = up
                    continue
            
            for i in range(low, up):
                
                # convert "14 May 2014 14:46 (UTC)" to "2014.05.14_14:46:00_TAI"
                #t12 = sp.et2utc(tai_ets[i]+720,"C",3) # tai_str after 12minutes
                #tai12_str = fmt % (t12[:4], months[t12[5:8]], t12[9:11], t12[12:14], t12[15:17], t12[18:20])
                 
                drms_str += tai_str[i] + ','
                #drms_str += tai_str[i] + '-' + tai12_str + ','
                
        low = up


    
    #from last transition to end
    #for i in range(low, len(src)):
    #   drms_str += tai_str[i] + ','
    #   print(drms_str)

    drms_str = drms_str[:-1] # truncate the last ','
    print('DRMS TIME STRING:')
    print(drms_str)

    return tai_str, tai_ets, src, clons, order


def get_synoptic_map_data():
    c = drms.Client()
    keys, segments = c.query('hmi.B_synoptic[2150]', key='HISTORY,CAR_ROT,CUNIT1,CUNIT2,LON_FRST,LON_LAST,T_START,T_STOP', seg='Br')
    url = 'http://jsoc.stanford.edu' + segments.Br[0]
    br = fits.open(url)
    return br[0].data, keys

def plot_synoptic_map(br, visibilty):
    

    xx = np.zeros([1440,3600])+1
    xx[:,:360] = np.nan

    brmask = br[0].data*xx

    # make the plot
    fig, ax = plt.subplots(figsize=(20,15), nrows=2, ncols=1)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    text_style = dict(fontsize=16)
    ax[0].tick_params(labelsize=14)
    sm1=ax[0].imshow(br[0].data,cmap=colormap_customized,vmin=-1500,vmax=1500,origin='lower',extent=[0,3600,0,1440],interpolation='nearest')
    ax[0].set_title('HMI $B_{r}$ Synoptic Chart for Carrington Rotation '+str(keys.CAR_ROT[0]),y=1.1,fontsize=20)
    ax[0].tick_params(axis='both', which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=True)

    # label the x-axis 
    xlabels = [0,60,120,180,240,300,360]
    xlocations = [0,600,1200,1800,2400,3000,3600]
    ax[0].set_xticklabels(xlabels)
    ax[0].set_xticks(xlocations)
    ax[0].set_xlabel('Carrington Longitude',**text_style)

    # Create the latitude labels on the right-hand side of the plot
    ylabels_r = [' ','-80 ',' ','-60',' ','-40',' ','-20',' ','0',' ',' 20',' ',' 40',' ',' 60',' ',' 80',' ']
    ylocations_r = ytick_normalize
    ax[0].set_yticklabels(ylabels_r)
    ax[0].set_yticks(ylocations_r)
    ax[0].set_ylabel('Latitude',**text_style)
    ax[0].yaxis.labelpad=-5
    ax[0].tick_params(labelsize=14)
    ax[0].yaxis.set_label_position("right")
    ax[0].yaxis.tick_right()


    ax[1].tick_params(labelsize=14)
    sm2=ax[1].imshow(brmask,cmap=colormap_customized,vmin=-1500,vmax=1500,origin='lower',extent=[0,3600,0,1440],interpolation='nearest')
    ax[1].set_title('HMI $B_{r}$ Synoptic Chart for Carrington Rotation '+str(keys.CAR_ROT[0]),y=1.1,fontsize=20)
    ax[1].tick_params(axis='both', which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=True)

    # label the x-axis 
    xlabels = [0,60,120,180,240,300,360]
    xlocations = [0,600,1200,1800,2400,3000,3600]
    ax[1].set_xticklabels(xlabels)
    ax[1].set_xticks(xlocations)
    ax[1].set_xlabel('Carrington Longitude',**text_style)

    # Create the latitude labels on the right-hand side of the plot
    ax[1].set_yticklabels(ylabels_r)
    ax[1].set_yticks(ylocations_r)
    ax[1].set_ylabel('Latitude',**text_style)
    ax[1].yaxis.labelpad=-5
    ax[1].tick_params(labelsize=14)
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()

    # plot the colorbar
    divider_cbar = make_axes_locatable(ax[0])
    width_cbar = axes_size.AxesY(ax[0], aspect=0.05)
    pad_cbar = axes_size.Fraction(2.6, width_cbar)
    cax = divider_cbar.append_axes("right", size=width_cbar, pad=pad_cbar)
    cax.tick_params(labelsize=14)
    cbar = fig.colorbar(sm1,cax=cax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('$B_{r}$ (Gauss)', size=16)

    # append empty colobar for layout
    divider_cbar = make_axes_locatable(ax[1])
    width_cbar = axes_size.AxesY(ax[1], aspect=0.05)
    pad_cbar = axes_size.Fraction(2.6, width_cbar)
    cax = divider_cbar.append_axes("right", size=width_cbar, pad=pad_cbar)

    # hide colorbar labels
    cax.axes.get_xaxis().set_visible(False)
    cax.axes.get_yaxis().set_visible(False)

    # hide colorbar frame
    cax.spines['bottom'].set_color('white')
    cax.spines['top'].set_color('white') 
    cax.spines['right'].set_color('white')
    cax.spines['left'].set_color('white')


def main():
    rscwdict = {"Extra Cold Checkout":"blue", "Cold Checkout":"orange", "Warm Checkout":"firebrick"}
    rswdict = {"Perihelion Window":"firebrick", "North Window":"blue", "South Window":"orange"}
    zonedict = {"Max Downlink":"white","> RSW Instantaneous":"lightgrey","> IS Instantaneous":"silver","< IS Instantaneous":"darkgrey"}
    AU = 149598000.0
    "Does the Stuff"
    config = read_json("orbit_config.json")
    mk_path = Path(config["metakernel"]["path"])
    mk_name = config["metakernel"]["name"]
    loaded=loadkernel(mk_path, mk_name)

    et_bounds=get_solo_coverage(mk_path)

    et_cp = sp.str2et(config["phases"]["cp"])
    et_np = sp.str2et(config["phases"]["np"])
    et_ep = sp.str2et(config["phases"]["ep"])    
    vgams = get_NAV_windows("SOLO","VENUS",et_bounds,10000.0)
    egams = get_NAV_windows("SOLO","EARTH",et_bounds,10000.0)
    gams = np.sort(np.concatenate((vgams,egams)),axis=None)
    
    downlink = daily_avg_downlink(config["downlink"]["sifecs"],"PASS",config["downlink"]["zones"])
    
    if config["windows"]["rsws"] == "default":
        print("Calculating default RSWs...")
        rsws = get_RSWs((et_np,et_bounds[1]))
        print("Writing default RSWs to "+config["windows"]["rsw_out"]+"...")
        outrsw = write_rsws(rsws,config["windows"]["rsw_out"])
    else:
        print("Reading RSWs from "+config["windows"]["rsws"]+"...")
        rsws = read_rsws(config["windows"]["rsws"])    
    print("Reading RSCWs from "+config["windows"]["rscws"]+"...")
    rscws = read_rscws(config["windows"]["rscws"])

    ets = np.arange(et_bounds[0],et_bounds[1],21600)
    ets[len(ets)-1]=et_bounds[1]

   
    [solo_GSE_pos, ltime] = sp.spkpos("SOLO",ets,"SOLO_GSE","NONE","EARTH")   
    [solo_HCI_state, ltime] = sp.spkezr("SOLO",ets,"SUN_INERTIAL","NONE","SUN") 
    #[earth_HCI_pos, ltime] = sp.spkpos("EARTH",ets,"SUN_INERTIAL","NONE","SUN")
    [earth_HCI_state, ltime] = sp.spkezr("EARTH",ets,"SUN_INERTIAL","NONE","SUN")

    solo_HCI_state = np.array(solo_HCI_state)
    solo_GSE_pos = np.array(solo_GSE_pos)/AU 
    solo_HCI_pos = solo_HCI_state[:,0:3]
    solo_HCI_vel = solo_HCI_state[:,3:6]    
    solo_hdis = np.zeros(len(ets))
    solo_hlon = np.zeros(len(ets))
    solo_hlat = np.zeros(len(ets))    
    #earth_HCI_pos = np.array(earth_HCI_pos)
    earth_HCI_state = np.array(earth_HCI_state)
    earth_HCI_pos = earth_HCI_state[:,0:3]
    earth_HCI_vel = earth_HCI_state[:,3:6]
    earth_hdis = np.zeros(len(ets))
    earth_hlon = np.zeros(len(ets))
    earth_hlat = np.zeros(len(ets))
    for i, void in enumerate(ets):
        [earth_hdis[i],earth_hlon[i],earth_hlat[i]] = sp.reclat(earth_HCI_pos[i,:])
        [solo_hdis[i],solo_hlon[i],solo_hlat[i]] = sp.reclat(solo_HCI_pos[i,:])
    solo_hdis = solo_hdis/AU
    solo_hlat = solo_hlat*sp.dpr()
    earth_hdis = earth_hdis/AU
    #earth_hlat = earth_hlat*sp.dpr()
    delta_omega_solo  = calc_relative_rotation(solo_HCI_state)
    delta_omega_earth = calc_relative_rotation(earth_HCI_state)
    delta_omega = delta_omega_solo
    solo_clon = simple_carrington(solo_hlon,ets)
    
# FSM CORE FUNCTIONS    

    # TODO Functionality for multiple RSWs
    fsm_start = "18 April 2022 03:00 (UTC)"
    fsm_end   = "18 May 2022 03:00 (UTC)"

    rsw_start = "11 April 2022 00:00 (UTC)"
    rsw_end   = "21 April 2022 00:00 (UTC)"

    deg4h = 2.2020520140929114  # average HMI rotation/4h 
    deg24h = 6*deg4h

    earth_clon = simple_carrington(earth_hlon,ets)


    solo_lld   = carrington_observation_deg(solo_clon, solo_hdis, ets, deg24h, fsm_start, fsm_end, None, None)
    solo_rsw   = carrington_observation_deg(solo_clon, solo_hdis, ets, deg4h, fsm_start, fsm_end, rsw_start, rsw_end)
    # solo_lld/rsw = [obs_utc, obs_et, obs_clon, obs_dist]    
    
    solo_obs   = merge_lld_rsw_observation_times(solo_lld, solo_rsw)
    # solo_obs = [utc, et, clon, hdis, src]

    earth_obs  = carrington_observation_times(earth_clon, earth_hdis, ets, fsm_start, 4)   
    #earth_obs = [obs_utc, obs_et, obs_clon, obs_dist]

    #earth_obs2 = carrington_observation_deg(earth_clon, earth_hdis, ets, deg4h, fsm_start, fsm_end, None, None)
    #[solo_lld_utc,  solo_lld_et,  solo_lld_clon,  solo_lld_hdis ] = solo_lld
    #[solo_rsw_utc,  solo_rsw_et,  solo_rsw_clon,  solo_rsw_hdis ] = solo_rsw
    #[earth_obs_utc, earth_obs_et, earth_obs_clon, earth_obs_hdis] = earth_obs
 
    [coverage, track, utc, et, clons, hdis, src, order, n] = carrington_observation_coverage(solo_obs, earth_obs, plot=False)

    #print(np.column_stack([utc[:n], np.round(clons[:n],2), np.round(hdis[:n],5), src[:n]]))

    # Map Solar Orbiter observation times to the corresponding Earth frame observation times
    earth_ets, earth_utcs = solo2earth_times(clons[:n], earth_obs[2], et[:n], earth_obs[1], src[:n])

    print('############')
    tai_et, tai_str = obs2drms_times("", "", earth_utcs)

    

    #print('\n\n TAI \n')
    #print(len(tai_str), n)
    #print(np.column_stack([earth_utcs[:n], tai_str[:n], src[:n]])) 
    
    tai_str, tai_et, src, clons, order = drms_timestring(tai_str[:n], tai_et[:n], src[:n], clons[:n])
    #print('############')
    print(np.column_stack([tai_str[:n], np.round(clons[:n],2), np.round(hdis[:n][order], 5), src[:n]])) 

   
    #print('#####')
    #print(np.column_stack([tai_str, clons, src])) 
    
    #TODO: If the latest run also didn't work
    # try adding 12 min intervals to all single 4h timestamps
    # maybe DRMS doesn't access the closest entry if the requested timestamp
    # is not available


    #print(np.column_stack([earth_obs[0], earth_obs[2]]))
    # ets earth_obs[1] clon earth_obs[2]



    #print(np.column_stack([utc[:n], test_utcs, np.round(clons[:n],2), src[:n]]))

    #print(sp.et2utc(np.interp(169.5, earth_obs[2][::-1][-10:], earth_obs[1][::-1][-10:]), 'c', 2))
    #print(sp.et2utc(interp360(169.5, earth_obs[2][:n], earth_obs[1][:n]), 'c', 2))

    #print(np.interp(169.5, earth_obs[2][::-1][-10:], earth_obs[1][::-1][-10:]))
    #print(interp360(169.5, earth_obs[2][:n], earth_obs[1][:n]))
    #set_trace()
    #coverage_solo, coverage_earth = carrington_observation_coverage(solo_obs, earth_obs)


    #set_trace()
    
    #order = np.argsort(clons[:n])
    #print(np.column_stack([utc[order], np.round(clons[order],2), src[order]]))
    # 2:hmi, replace 0 and 1 with hmi timestamps for that clon
    carrington_observation_duration(solo_clon, earth_clon, solo_hdis, ets)
    #carrington_observation_coverage_old(solo_clon, earth_clon, solo_hdis, ets)
# FSM CORE FUNCTIONS   



    sp.kclear()
    

main()
