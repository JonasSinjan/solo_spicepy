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
    sp.gfposc("SOLO", "SOLO_HCI", "NONE", "SUN", "LATITUDINAL", "LATITUDE",  "LOCMAX", 0.0, 0.0, 86400.0, 1000, interval, nor_times )
    for i in range(sp.wncard(nor_times)):
        win_start.append(sp.wnfetd(nor_times,i)[0]-(5*86400))
        win_end.append(sp.wnfetd(nor_times,i)[0]+(5*86400))
        win_type.append("North Window")
    sou_times = stypes.SPICEDOUBLE_CELL(100)
    sp.gfposc("SOLO", "SOLO_HCI", "NONE", "SUN", "LATITUDINAL", "LATITUDE",  "LOCMIN", 0.0, 0.0, 86400.0, 1000, interval, sou_times )
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
    (basepos, ltime) = sp.spkpos("EARTH", t0, "SOLO_HCI","NONE","SUN")
    (baserad,baselon,baselat) = sp.reclat(basepos)
    baselon2 = (baselon*sp.dpr()+360.0) % 360
    
    clon = []
    for (index, et) in enumerate(ets):
        delta_t = et - t0
        delta_phi = (delta_t*omega*sp.dpr()) % 360
        zero_lon = (baselon2+delta_phi) % 360
        solo_lon = (hci_lon[index]+360) % 360
        clon_buffer = (360+(solo_lon-zero_lon)) % 360
        if clon_buffer > 180:
            clon_buffer = clon_buffer - 360
        clon.append(clon_buffer)
    return(clon)
    
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
    return(output)

def setup_canvas():
    
    fig = plt.figure(figsize=(19.2, 10.8), dpi=150)
    

    orbit = fig.add_axes((0.05,0.07,0.50625,0.9))
    hdist = fig.add_axes((0.58,0.07,0.36,0.15))
    hlats = fig.add_axes((0.58,0.235,0.36,0.15))
    omega = fig.add_axes((0.58,0.40,0.36,0.15))
    rolls = fig.add_axes((0.58,0.565,0.36,0.15))

    hdist.yaxis.tick_right()    
    hlats.yaxis.tick_right()
    omega.yaxis.tick_right()
    rolls.yaxis.tick_right()

    hdist.yaxis.set_label_position("right")
    hlats.yaxis.set_label_position("right")
    omega.yaxis.set_label_position("right")
    rolls.yaxis.set_label_position("right")
    
    hlats.set_xticklabels([])
    omega.set_xticklabels([])
    rolls.set_xticklabels([])
        


    for axis in fig.get_axes():
        axis.tick_params(which='major', axis='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=12, length=7)
        axis.tick_params(which='minor', axis='both', direction="in", bottom=True, top=True, left=True, right=True, length=5)
    
    orbit.minorticks_on()
    orbit.xaxis.set_major_locator(plt.MaxNLocator(5))
    orbit.set_xlabel("GSE X (AU)", fontsize = 14)
    orbit.set_ylabel("GSE Y (AU)", fontsize = 14)  
    orbit.set_xlim(-0.2,2.0)
    orbit.set_ylim(-1.1,1.1)
    orbit.axvline(x=0, color = "grey")
    orbit.axvline(x=1, color = "grey")
    orbit.axhline(y=0, color = "grey")
    plot_concentric_circles(orbit, np.arange(0.1,2.1,0.1), (0,0), edgecolor='lightgray',facecolor='none', linestyle='dotted')
    plot_concentric_circles(orbit, np.arange(0.3,2.1,0.1), (1.0,0), edgecolor='lightgray',facecolor='none', linestyle='dashed')
    orbit.add_patch(mpatches.Circle((0,0),0.03,facecolor="lightblue",edgecolor = "black"))
    orbit.add_patch(mpatches.Circle((1,0),0.06,facecolor="yellow",edgecolor="black"))
    
    hdist.set_xlabel("Universal Time", fontsize = 14)
    hdist.set_ylabel(r"Hcentric" "\n" "Dist." "(AU)", fontsize = 14)
    hdist.set_ylim(0.2,1.2)
    hdist.yaxis.set_major_locator(plt.MaxNLocator(6))    

    hlats.set_ylabel(r"HCI Lat (deg)" "\n" "(Earth Green)", fontsize = 14)
    hlats.set_ylim(-35,35)
    hlats.yaxis.set_major_locator(plt.MaxNLocator(7))
    
    omega.set_ylabel(r"Delta Omega" "\n" "(deg/day)", fontsize = 14)
    omega.set_ylim(5,15)
    omega.yaxis.set_major_locator(plt.MaxNLocator(5))


    rolls.set_ylabel(r"Roll Angle (Green)" "\n" "Carr. Lon. (Black)", fontsize = 14)
    rolls.set_ylim(-180,180)
    rolls.yaxis.set_major_locator(plt.MaxNLocator(5))

    return(fig,orbit,hdist,hlats,omega,rolls)
    
def plot_concentric_circles(panel, radii, offset, **kwargs):
    for radius in radii:
        panel.add_patch(mpatches.Circle(offset,radius,**kwargs))
    return()
        
def add_legend(fig, LTPstr, datestr, windict, zonedict):

    legend = fig.add_axes((0.58,0.72,0.4,0.25))
    legend.axis('off')
    legend.add_patch(mpatches.Rectangle((0.0,0.0),0.4,1.0,fill = False, edgecolor='black', linewidth=2.0))
    legend.annotate(LTPstr,(0.42,0.89), fontsize=28)
    legend.annotate(datestr,(0.42,0.7), fontsize=24)
    zonenames = list(zonedict)
    zonenames.reverse()
    winnames = list(windict)
    for index, zonename in enumerate(zonenames):
        x = 0.42
        y = (index+1)*0.025+(index*0.125)
        legend.add_patch(mpatches.Rectangle((x,y),0.1,0.125,facecolor = zonedict[zonename], edgecolor='black', linewidth=2.0))
        legend.annotate(zonename,(x+0.125,y+0.025),fontsize=20)
    legend.add_patch(mpatches.Rectangle((0.05,0.9),0.025,0.07, color='black'))
    legend.add_patch(mpatches.Ellipse((0.0625,0.8),0.025,0.07, color='black'))
    legend.add_patch(mpatches.Rectangle((0.0425,0.6),0.04,0.07, color='magenta'))
    legend.add_patch(mpatches.Rectangle((0.0425,0.4465),0.04,0.07, color='cyan'))
    legend.add_patch(mpatches.Rectangle((0.0325,0.324),0.06,0.02, color=windict[winnames[0]]))
    legend.add_patch(mpatches.Rectangle((0.0325,0.199),0.06,0.02, color=windict[winnames[1]]))
    legend.add_patch(mpatches.Rectangle((0.0325,0.074),0.06,0.02, color=windict[winnames[2]]))
    
    legend.annotate("Period Start",(0.125,0.91),fontsize=14)
    legend.annotate("GAM Restrictions",(0.125,0.775),fontsize=14)
    legend.annotate("Solar Conjunction",(0.125,0.61),fontsize=14, color = "magenta")
    legend.annotate("Safe Mode Blackout",(0.125,0.455),fontsize=14, color = "cyan")
    legend.annotate(winnames[0],(0.125,0.31),fontsize=14, color = windict[winnames[0]])
    legend.annotate(winnames[1],(0.125,0.185),fontsize=14, color = windict[winnames[1]])
    legend.annotate(winnames[2],(0.125,0.06),fontsize=14, color = windict[winnames[2]])
  
    return(legend)
    
def find_ranges(iterable):
    lo = []
    hi = []
    for group in mit.consecutive_groups(iterable):
        group = np.array(list(group))
        lo.append(np.min(group))
        hi.append(np.max(group))
    return(lo,hi)            
def overplot_zones(panel,timetags_start,timetags_end,zone,zonedict):
    zonenames = list(zonedict)
    for zonenum, zonename in enumerate(zonenames):    
        indices = np.where(np.array(zone) == zonenum)
        low,high = find_ranges(indices[0])
        for lo, hi in zip(low,high):
            panel.axvspan(timetags_start[lo],timetags_end[hi],color = zonedict[zonename])
#            print(sp.et2utc(timetags[lo],"ISOC",3)+" - "+sp.et2utc(timetags[lo],"ISOC",3))
    return()


def setup_time_axis(panel,plot_start, plot_end, labels = True):
    timenudge = 86400.0*15.0
    panel.set_xlim(plot_start,plot_end)
    start_datetime = et2datetime64(plot_start)
    end_datetime = (et2datetime64(plot_end+timenudge))
    tick_datetimes = np.arange(start_datetime[0], end_datetime[0], dtype='datetime64[M]')
    tick_ets = datetime642et(tick_datetimes[1:])
    tick_strings = sp.timout(tick_ets,"DD/Mon ::UTC")
    panel.set_xticks(tick_ets)
    if labels == True: panel.set_xticklabels(tick_strings)
    return()

def overplot_conj(panel,timetags_start, timetags_end, conj_type):
    if conj_type == "bout":
        box_colour = 'cyan'
        minmax = (0.88,1.0)
    elif conj_type == "conj":
        box_colour = "magenta"
        minmax = (0.75,0.87)
    for t_start, t_end in zip(timetags_start,timetags_end):
        panel.axvspan(t_start,t_end,color=box_colour,ymin=minmax[0],ymax=minmax[1])
    return()


#def overplot_conj_orb(panel, timetags, conj_start, conj_end, conj_type):
#    if conj_type == "bout": plot_colour = "cyan"
#    if conj_type == "conj": plot_colour = "magenta"
#    lines = panel.get_lines()
#    for line in lines:
#        if line.get_label() == "forwin":
#            line_x = line.get_xdata()
#            line_y = line.get_ydata()
#            conj_startdex = np.where(np.logical_and(np.greater_equal(conj_start,np.min(timetags)), np.less_equal(conj_start, np.max(timetags))))
#            conj_enddex = np.where(np.logical_and(np.greater_equal(conj_end,np.min(timetags)), np.less_equal(conj_end, np.max(timetags))))
#            if len(conj_startdex[0]) > 0:
#                for cs in conj_startdex[0]:
#                    x = np.interp(conj_start[cs],timetags,line_x)
#                    y = np.interp(conj_start[cs],timetags,line_y)
#                    panel.plot([1,x],[0,y],linewidth=2,color=plot_colour)
#            if len(conj_enddex[0]) > 0:
#                for ce in conj_enddex[0]:
#                    x = np.interp(conj_end[ce],timetags,line_x)
#                    y = np.interp(conj_end[ce],timetags,line_y)
#                    panel.plot([1,x],[0,y],linewidth=2,color=plot_colour)
#    return()
    
def overplot_wins_ts(panel,windows,windict):
    lines = panel.lines
    for line in lines:
        if line.get_label() == "solo":
            x = line.get_xdata()
            y = line.get_ydata()
            for winstart, winend, wintype in zip(windows[0],windows[1],windows[2]):
                startdex = np.min(np.where(x >= winstart))
                enddex = np.max(np.where(x <= winend))
                panel.plot(x[startdex:enddex],y[startdex:enddex],' .', markersize=4,color=windict[wintype])
    return()     

def overplot_wins_orb(panel,timetags,windows,windict):
    lines = panel.lines
    for line in lines:
        if line.get_label() == "solo":
            x = line.get_xdata()
            y = line.get_ydata()
            for winstart, winend, wintype in zip(windows[0],windows[1],windows[2]):
                startdices = np.where(timetags >= winstart)
                enddices = np.where(timetags <= winend)
                print(wintype)
                if len(startdices[0]) > 0 and len(enddices[0]) > 0:
                    startdex = np.min(startdices)
                    enddex = np.max(enddices)
                    print("Plotting "+wintype)
                    panel.plot(x[startdex:enddex],y[startdex:enddex],linewidth=4,color=windict[wintype])
    return()     

def overplot_gam_ts(panel,gamx):
    lines = panel.lines
    for line in lines:
        if line.get_label() == "solo":
            x = line.get_xdata()
            y = line.get_ydata()
            gamy = np.interp(gamx,x,y)
            panel.plot(gamx, gamy, " ok", markersize=10)
    return()

def overplot_gam_orb(panel,gams,timetags):
    startdices = np.where(gams >= np.min(timetags))
    enddices = np.where(gams <= np.max(timetags))
    if len(startdices[0]) > 0 and len(enddices[0]) > 0:
        startdex = np.min(startdices)
        enddex = np.max(enddices)
        if startdex < enddex:
            lines= panel.lines    
            for line in lines:
                if line.get_label() == "solo":
                    x = line.get_xdata()
                    y = line.get_ydata()
                    gamt = gams[startdex:enddex+1]
                    gamx = np.interp(gamt,timetags,x)
                    gamy = np.interp(gamt,timetags,y)
                    panel.plot(gamx, gamy, ' ok',markersize=15)
    return()
    
    
def orbit_ticks(panel, timetags, line_label):
    timenudge = 86400.0*15.0
    start_datetime = et2datetime64(np.min(timetags))
    end_datetime = (et2datetime64(np.max(timetags)+timenudge))
    tick_datetimes = np.arange(start_datetime[0], end_datetime[0], dtype='datetime64[M]')
    tick_ets = datetime642et(tick_datetimes[1:])
    lines = panel.lines
    for line in lines:
        if line.get_label() == line_label:
            x = line.get_xdata()
            y = line.get_ydata()
            line_col = line.get_color()
            tick_x = np.interp(tick_ets,timetags,x)
            tick_y = np.interp(tick_ets,timetags,y)
            vec_x = []
            vec_y = []
            for t_et, tx, ty in zip(tick_ets,tick_x,tick_y):
                index = np.min(np.where(timetags > t_et))
                vec_x.append(x[index]-tx)
                vec_y.append(y[index]-ty)
            vec_z = np.zeros(len(vec_x),dtype="float64")
            vec_y = np.array(vec_y)
            vec_x = np.array(vec_x)
            vec = np.transpose(np.array([vec_x,vec_y,vec_z]))
            vec_norm = vec/np.linalg.norm(vec, axis=1, keepdims=True)
            z_arr = np.transpose(np.array([np.zeros(len(vec_x)),np.zeros(len(vec_x)),np.ones(len(vec_x))]))
            vec_plus = np.cross(vec_norm,z_arr)
            vec_minus = np.cross(z_arr,vec_norm)
            for tickdex, (tx, ty) in enumerate(zip(tick_x,tick_y)):
                newx = [tx+vec_minus[tickdex,0]*0.025,tx,tx+vec_plus[tickdex,0]*0.025]
                newy = [ty+vec_minus[tickdex,1]*0.025,ty,ty+vec_plus[tickdex,1]*0.025]
                panel.plot(newx,newy,color=line_col,linewidth=2.0)
    return()            

def orbit_key(orbit, plot_sta, plot_psp, plot_bc):
    solo = orbit.text(0.01,1.04,"Solar Orbiter", color = "black", size=18)
    if plot_psp: psp = orbit.text(0.44,1.04,"Parker Solar Probe", color = "darkorchid", size=18)
    if plot_sta: sta = orbit.text(1.03,1.04,"STEREO Ahead", color = "olivedrab", size=18)
    if plot_bc: bc = orbit.text(1.52,1.04,"Bepi Colombo", color = "darkgoldenrod", size=18)
                

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
    
    probe_path = Path(config["probekernel"]["path"])
    probe_name = config["probekernel"]["name"]
    probe_loaded=loadkernel(probe_path, probe_name)
    probe_bounds = get_orbit_coverage(probe_path, probe_name, -96)

    sta_path = Path(config["stereokernel"]["path"])
    sta_name = config["stereokernel"]["name"]
    sta_loaded=loadkernel(sta_path, sta_name)
    sta_bounds = get_orbit_coverage(sta_path, sta_name, -234)

    bc_path = Path(config["bepikernel"]["path"])
    bc_name = config["bepikernel"]["name"]
    bc_loaded=loadkernel(bc_path, bc_name)
    bc_bounds = get_orbit_coverage(bc_path, bc_name, -121)


    et_bounds=get_solo_coverage(mk_path)

    et_cp = sp.str2et(config["phases"]["cp"])
    et_np = sp.str2et(config["phases"]["np"])
    et_ep = sp.str2et(config["phases"]["ep"])    
    vgams = get_NAV_windows("SOLO","VENUS",et_bounds,10000.0)
    egams = get_NAV_windows("SOLO","EARTH",et_bounds,10000.0)
    gams = np.sort(np.concatenate((vgams,egams)),axis=None)
    
    downlink = daily_avg_downlink(config["downlink"]["sifecs"],"TX_ON",config["downlink"]["zones"])
    
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
    LTPs = get_planning_periods((et_cp,et_bounds[1]),rsws)
    conj = get_conjunctions(et_bounds, 3.0, 0.0)
    bout = get_conjunctions(et_bounds, 5.0, 3.0)
    ets = np.arange(et_bounds[0],et_bounds[1],21600)
    ets[len(ets)-1]=et_bounds[1]

    first_common_et_probe = np.min(np.where(ets >= probe_bounds[0]))
    last_common_et_probe = np.max(np.where(ets < probe_bounds[1]))
    probe_ets = np.arange(ets[first_common_et_probe],ets[last_common_et_probe],21600)
    probe_ets[len(probe_ets)-1]=ets[last_common_et_probe]

    first_common_et_sta = np.min(np.where(ets >= sta_bounds[0]))
    last_common_et_sta = np.max(np.where(ets < sta_bounds[1]))
    sta_ets = np.arange(ets[first_common_et_sta],ets[last_common_et_sta],21600)
    sta_ets[len(sta_ets)-1]=ets[last_common_et_sta]

    first_common_et_bc = np.min(np.where(ets >= bc_bounds[0]))
    last_common_et_bc = np.max(np.where(ets < bc_bounds[1]))
    bc_ets = np.arange(ets[first_common_et_bc],ets[last_common_et_bc],21600)
    bc_ets[len(bc_ets)-1]=ets[last_common_et_bc]

    mer_first_et=np.max(bc_ets)+21600.0    
    mer_last_et=sp.str2et("2028-05-01 06:00:00.000 UTC")
    mer_ets = np.arange(mer_first_et,mer_last_et,21600)
    
    [solo_GSE_pos, ltime] = sp.spkpos("SOLO",ets,"SOLO_GSE","NONE","EARTH")   
    [solo_HCI_state, ltime] = sp.spkezr("SOLO",ets,"SOLO_HCI","NONE","SUN") 
    [earth_HCI_pos, ltime] = sp.spkpos("EARTH",ets,"SOLO_HCI","NONE","SUN")
    [probe_GSE_pos, ltime] = sp.spkpos("SPP",probe_ets,"SOLO_GSE","NONE","EARTH")
    [probe_HCI_pos, ltime] = sp.spkpos("SPP",probe_ets,"SOLO_HCI","NONE","SUN")
    [sta_GSE_pos, ltime] = sp.spkpos("STEREO AHEAD",sta_ets,"SOLO_GSE","NONE","EARTH")
    [bc_GSE_pos, ltime] = sp.spkpos("BEPICOLOMBO MPO",bc_ets,"SOLO_GSE","NONE","EARTH")
    [bc_HCI_pos, ltime] = sp.spkpos("BEPICOLOMBO MPO",bc_ets,"SOLO_HCI","NONE","SUN")
    [mer_GSE_pos, ltime] = sp.spkpos("MERCURY",mer_ets,"SOLO_GSE","NONE","EARTH")
    [mer_HCI_pos, ltime] = sp.spkpos("MERCURY",mer_ets,"SOLO_HCI","NONE","SUN")
    
    bc_ets = np.concatenate((bc_ets,mer_ets))
    bc_GSE_pos = np.concatenate((np.array(bc_GSE_pos),np.array(mer_GSE_pos)))
    bc_HCI_pos = np.concatenate((np.array(bc_HCI_pos),np.array(mer_HCI_pos)))
    
    solo_HCI_state = np.array(solo_HCI_state)
    solo_GSE_pos = np.array(solo_GSE_pos)/AU 
    solo_HCI_pos = solo_HCI_state[:,0:3]
    solo_HCI_vel = solo_HCI_state[:,3:6]    
    solo_hdis = np.zeros(len(ets))
    solo_hlon = np.zeros(len(ets))
    solo_hlat = np.zeros(len(ets))    
    earth_HCI_pos = np.array(earth_HCI_pos)
    earth_hlat = np.zeros(len(ets))
    for i, void in enumerate(ets):
        [buffer,buffer,earth_hlat[i]] = sp.reclat(earth_HCI_pos[i,:])
        [solo_hdis[i],solo_hlon[i],solo_hlat[i]] = sp.reclat(solo_HCI_pos[i,:])
    solo_hdis = solo_hdis/AU
    solo_hlat = solo_hlat*sp.dpr()
    earth_hlat = earth_hlat*sp.dpr()
    delta_omega = calc_relative_rotation(solo_HCI_state)
    solo_clon = simple_carrington(solo_hlon,ets)
    
    probe_GSE_pos = np.array(probe_GSE_pos)/AU
    probe_HCI_pos = np.array(probe_HCI_pos)/AU
    probe_hlat = np.zeros(len(probe_ets))    
    for i, void in enumerate(probe_ets):
        [buffer,buffer,probe_hlat[i]] = sp.reclat(probe_HCI_pos[i,:])
    probe_hlat = probe_hlat*sp.dpr()

    sta_GSE_pos = np.array(sta_GSE_pos)/AU
    
    bc_GSE_pos = np.array(bc_GSE_pos)/AU
    bc_HCI_pos = np.array(bc_HCI_pos)/AU
    bc_hlat = np.zeros(len(bc_ets))    
    for i, void in enumerate(bc_ets):
        [buffer,buffer,bc_hlat[i]] = sp.reclat(bc_HCI_pos[i,:])
    bc_hlat = bc_hlat*sp.dpr()

    

# Big Plotting loop starts here.
    for LTP_counter, LTP_start in enumerate(LTPs[0]):
        print("Plotting LTP "+"{:02d}".format(LTP_counter+1))

        LTP_end  = LTPs[1][LTP_counter]
        LTP_startdex = np.min(np.where(ets >= LTP_start))
        LTP_enddex = np.max(np.where(ets < LTP_end))
        
        plot_probe = False
        probe_index_range = np.where(np.logical_and(np.greater_equal(probe_ets,LTP_start),np.less(probe_ets,LTP_end)))
        if len(probe_index_range[0]) > 0:
            plot_probe = True
            probe_startdex = np.min(probe_index_range)
            probe_enddex = np.max(probe_index_range)

        plot_sta = False
        sta_index_range = np.where(np.logical_and(np.greater_equal(sta_ets,LTP_start),np.less(sta_ets,LTP_end)))
        if len(sta_index_range[0]) > 0:
            plot_sta = True
            sta_startdex = np.min(sta_index_range)
            sta_enddex = np.max(sta_index_range)

        plot_bc = False
        bc_index_range = np.where(np.logical_and(np.greater_equal(bc_ets,LTP_start),np.less(bc_ets,LTP_end)))
        if len(bc_index_range[0]) > 0:
            plot_bc = True
            bc_startdex = np.min(bc_index_range)
            bc_enddex = np.max(bc_index_range)



        if LTP_start < et_np:
            LTP_phase = "CP"
            windict = rscwdict
            windows = rscws
        if LTP_start >= et_np:
            LTP_phase = "NMP"
            windict = rswdict
            windows = rsws
        if LTP_start >= et_ep:
            LTP_phase = "EMP"
            windict = rswdict
            windows = rsws
        

        LTP_namestr = "LTP "+"{:02d}".format(LTP_counter+1)+" ("+LTP_phase+")"
        LTP_startstr = sp.timout(LTP_start, "YYYY-MM-DD ::UTC")
        LTP_endstr = sp.timout(LTP_end, "YYYY-MM-DD ::UTC")
        LTP_datestr = LTP_startstr+" - "+LTP_endstr

        (page,orbit,hdist,hlats,omega,rolls) = setup_canvas()
        setup_time_axis(hdist,LTP_start,LTP_end, labels = True)
        setup_time_axis(hlats,LTP_start,LTP_end, labels = False)
        setup_time_axis(omega,LTP_start,LTP_end, labels = False)
        setup_time_axis(rolls,LTP_start,LTP_end, labels = False)
        
        overplot_zones(hdist, downlink["effective_start"], downlink["pass_end"], downlink["pass_zone"],zonedict)
        overplot_zones(hlats, downlink["effective_start"], downlink["pass_end"], downlink["pass_zone"],zonedict)
        overplot_zones(omega, downlink["effective_start"], downlink["pass_end"], downlink["pass_zone"],zonedict)
        overplot_zones(rolls, downlink["effective_start"], downlink["pass_end"], downlink["pass_zone"],zonedict)

        hlats.axhline(y=0,color="red",linestyle="dashed")
        rolls.axhline(y=0,color="red",linestyle="dashed")
        
        overplot_conj(rolls,bout[0],bout[1],"bout")
        overplot_conj(rolls,conj[0],conj[1],"conj")
#        overplot_conj_orb(orbit,ets[LTP_startdex:LTP_enddex],bout[0],bout[1],"bout")
#        overplot_conj_orb(orbit,ets[LTP_startdex:LTP_enddex],conj[0],conj[1],"conj")
        
        orbit.plot(solo_GSE_pos[LTP_startdex:LTP_enddex,0],solo_GSE_pos[LTP_startdex:LTP_enddex,1],linewidth=2,label="solo",color="black")     
        orbit_ticks(orbit,ets[LTP_startdex:LTP_enddex], "solo")
        if plot_probe: 
            orbit.plot(probe_GSE_pos[probe_startdex:probe_enddex,0],probe_GSE_pos[probe_startdex:probe_enddex,1], linewidth=1, label="psp", color = "darkorchid")
            orbit_ticks(orbit,probe_ets[probe_startdex:probe_enddex], "psp")
            hlats.plot(probe_ets,probe_hlat, linewidth=1,label="psp", color = "darkorchid")

        if plot_sta: 
            orbit.plot(sta_GSE_pos[sta_startdex:sta_enddex,0],sta_GSE_pos[sta_startdex:sta_enddex,1], linewidth=1, label="stereo", color = "olivedrab")
            orbit_ticks(orbit,sta_ets[sta_startdex:sta_enddex], "stereo")

        if plot_bc: 
            orbit.plot(bc_GSE_pos[bc_startdex:bc_enddex,0],bc_GSE_pos[bc_startdex:bc_enddex,1], linewidth=1, label="bc", color = "darkgoldenrod")
            orbit_ticks(orbit,bc_ets[bc_startdex:bc_enddex], "bc")
            hlats.plot(bc_ets,bc_hlat, linewidth=1,label="bc", color = "darkgoldenrod")

        hdist.plot(ets,solo_hdis, linewidth=2,label="solo", color = "black")
        hlats.plot(ets,solo_hlat, linewidth=2,label="solo", color = "black")
        hlats.plot(ets,earth_hlat, linewidth=2,label="earthlat", color = "green")
        omega.plot(ets,delta_omega, linewidth=2,label="solo", color = "black")  
        rolls.plot(ets,solo_clon, " .",label = "solo", color = "black", markersize=2)

        overplot_wins_orb(orbit, ets[LTP_startdex:LTP_enddex], windows, windict)
        overplot_wins_ts(hdist, windows, windict)
        overplot_wins_ts(hlats, windows, windict)
        overplot_wins_ts(omega, windows, windict)
        overplot_wins_ts(rolls, windows, windict)
        
        overplot_gam_ts(hdist,gams[3::])
        overplot_gam_ts(hlats,gams[3::])
        overplot_gam_ts(omega,gams[3::])
        overplot_gam_ts(rolls,gams[3::])
        overplot_gam_orb(orbit,gams[3::],ets[LTP_startdex:LTP_enddex])
        orbit_key(orbit, plot_sta, plot_probe, plot_bc)

        legend = add_legend(page, LTP_namestr, LTP_datestr,windict,zonedict)

        if LTP_counter == 0:
            commissioning_enddex = np.max(np.where(ets <= et_cp))
            orbit.plot(solo_GSE_pos[LTP_startdex:commissioning_enddex,0],solo_GSE_pos[LTP_startdex:commissioning_enddex,1],linewidth=2,label="commissioning")
            hdist.plot(ets[LTP_startdex:commissioning_enddex],solo_hdis[LTP_startdex:commissioning_enddex], linewidth=2)
            hlats.plot(ets[LTP_startdex:commissioning_enddex],solo_hlat[LTP_startdex:commissioning_enddex], linewidth=2)
            omega.plot(ets[LTP_startdex:commissioning_enddex],delta_omega[LTP_startdex:commissioning_enddex], linewidth=2)
            rolls.plot(ets[LTP_startdex:commissioning_enddex],solo_clon[LTP_startdex:commissioning_enddex], " .", markersize=2)
        

        if config["plot_type"] == "ltp":
            if plot_sta: orbit.plot(sta_GSE_pos[sta_startdex,0],sta_GSE_pos[sta_startdex,1]," sk",markersize=7, color="olivedrab")
            if plot_probe: orbit.plot(probe_GSE_pos[probe_startdex,0],probe_GSE_pos[probe_startdex,1]," sk",markersize=7, color="darkorchid")
            if plot_bc: orbit.plot(bc_GSE_pos[bc_startdex,0],bc_GSE_pos[bc_startdex,1]," sk",markersize=7, color="darkgoldenrod")
            if LTP_counter == 0:
                orbit.plot(solo_GSE_pos[commissioning_enddex,0],solo_GSE_pos[commissioning_enddex,1]," sk",markersize=15)
                hdist.plot(ets[commissioning_enddex],solo_hdis[commissioning_enddex]," sk",markersize=10)
                hlats.plot(ets[commissioning_enddex],solo_hlat[commissioning_enddex]," sk",markersize=10)
                omega.plot(ets[commissioning_enddex],delta_omega[commissioning_enddex]," sk",markersize=10)
                rolls.plot(ets[commissioning_enddex],solo_clon[commissioning_enddex]," sk",markersize=10)
            else:
                orbit.plot(solo_GSE_pos[LTP_startdex,0],solo_GSE_pos[LTP_startdex,1]," sk",markersize=15)
            filename = "GSE_Orbit_Plot_LTP"+"{:02d}".format(LTP_counter+1)+"_bc."+config["outformat"]
            page.savefig(filename)
        if config["plot_type"] == "for_movie":
            dir_string = "LTP"+"{:02d}".format(LTP_counter+1)
            if not os.path.exists(dir_string): os.makedirs(dir_string)
            solo_indices = np.arange(LTP_startdex,LTP_enddex)
            if plot_probe: probe_indices = np.arange(probe_startdex,probe_enddex)
            if plot_sta: sta_indices = np.arange(sta_startdex,sta_enddex)
            if plot_bc: bc_indices = np.arange(bc_startdex,bc_enddex)
            timestring_old = "Period Start"
            for j in solo_indices:
                j_others = j-np.min(solo_indices)
                plot_probe2 = False
                plot_sta2 = False
                plot_bc2 = False
                if plot_probe:
                    if j_others < len(probe_indices): plot_probe2 = True
                if plot_sta:
                    if j_others < len(sta_indices): plot_sta2 = True
                if plot_bc:
                    if j_others < len(bc_indices): plot_bc2 = True
                timestring = sp.timout(ets[j], "YYYY-MM-DD HR:MN ::UTC")
                for t in legend.texts:
                    if t.get_text() == timestring_old:
                        t.set_text(timestring)
                timestring_old = timestring
                orbit.plot(solo_GSE_pos[j,0],solo_GSE_pos[j,1]," sk",markersize=15, label="moveme")
                hdist.plot(ets[j],solo_hdis[j]," sk",markersize=10, label="moveme")
                hlats.plot(ets[j],solo_hlat[j]," sk",markersize=10, label="moveme")
                omega.plot(ets[j],delta_omega[j]," sk",markersize=10, label="moveme")
                rolls.plot(ets[j],solo_clon[j]," sk",markersize=10, label="moveme")
                if plot_sta2: orbit.plot(sta_GSE_pos[sta_indices[j_others],0],sta_GSE_pos[sta_indices[j_others],1]," sk",markersize=7, color="olivedrab",label="moveme")
                if plot_probe2:
                    orbit.plot(probe_GSE_pos[probe_indices[j_others],0],probe_GSE_pos[probe_indices[j_others],1]," sk",markersize=7, color="darkorchid", label="moveme")
                    hlats.plot(probe_ets[probe_indices[j_others]],probe_hlat[probe_indices[j_others]]," sk",markersize=5,label="moveme", color = "darkorchid")
                if plot_bc2:
                    orbit.plot(bc_GSE_pos[bc_indices[j_others],0],bc_GSE_pos[bc_indices[j_others],1]," sk",markersize=7, color="darkgoldenrod", label="moveme")
                    hlats.plot(bc_ets[bc_indices[j_others]],bc_hlat[bc_indices[j_others]]," sk",markersize=5,label="moveme", color = "darkgoldenrod")

                filename_time = sp.timout(ets[j], "YYYYMMDDHRMN ::UTC")
                filename = dir_string+"/GSE_Orbit_Plot_LTP"+"{:02d}".format(LTP_counter+1)+"_"+filename_time+"_bc."+config["outformat"]
                page.savefig(filename)
                lines = orbit.lines
                for l in lines:
                    if l.get_label() == "moveme":
                        l.remove()
                lines = hdist.lines
                for l in lines:
                    if l.get_label() == "moveme":
                        l.remove()
                lines = hlats.lines
                for l in lines:
                    if l.get_label() == "moveme":
                        l.remove()
                lines = omega.lines
                for l in lines:
                    if l.get_label() == "moveme":
                        l.remove()
                lines = rolls.lines
                for l in lines:
                    if l.get_label() == "moveme":
                        l.remove()
                
        plt.close(page)



                
                    
            
#            legend = add_legend(page, LTP_namestr, LTP_datestr,windict,zonedict)
#        print("LTP")
#        print(LTP_counter+1)
#        print(et2datetime64(LTP_start))
#        print(et2datetime64(LTP_end))

    #test = setup_canvas()
    #legtest = add_legend(test,"LTP 01 (CP)","2020/01/01 - 2020/07/01", rscwdict, zonedict)
    #test.savefig("test.png")
    #test.savefig("test.svg")

#    loaded_old = loaded
#    loaded=unloadkernel(mk_path, mk_name)   
#    print([loaded_old,loaded])
    sp.kclear()

main()
    