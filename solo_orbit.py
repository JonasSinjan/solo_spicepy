import numpy as np
import os
import spiceypy as sp
from pathlib import Path
import spiceypy.utils.support_types as stypes
import json
import math
import matplotlib.pyplot as plt
import sys

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

def read_json(filename):
    if filename:
        with open(filename, 'r') as f:
            config = json.load(f)
    return(config)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main(time, plot = False):
    AU = 149598000.0 #in km
    "Does the Stuff"
    config = read_json("orbit_config_js.json")
    mk_path = Path(config["metakernel"]["path"])
    mk_name = config["metakernel"]["name"]
    loaded=loadkernel(mk_path, mk_name)

    et_bounds=get_solo_coverage(mk_path) #ephemeris time

    ets = np.linspace(et_bounds[0],et_bounds[1],15744)
    ets[len(ets)-1]=et_bounds[1]

    [solo_GSE_pos, ltime_gse] = sp.spkpos("SOLO",ets,"SOLO_GSE","NONE","EARTH")
    [solo_HCI_state, ltime] = sp.spkezr("SOLO",ets,"SOLO_HCI","NONE","SUN") #sp.spkezr("SOLO",ets,"SUN_INERTIAL","NONE","SUN")
    [earth_HCI_pos, ltime] = sp.spkpos("EARTH",ets,"SOLO_HCI","NONE","SUN") 
    [venus_HCI_pos, ltime] = sp.spkpos("VENUS",ets,"SOLO_HCI","NONE","SUN")

    solo_HCI_state = np.array(solo_HCI_state)
    solo_GSE_pos = np.array(solo_GSE_pos)/AU 
    solo_HCI_pos = solo_HCI_state[:,0:3]

    solo_HCI_vel = solo_HCI_state[:,3:6] #in km/s    
    
    solo_hdis = np.zeros(len(ets))
    solo_hlon = np.zeros(len(ets))
    solo_hlat = np.zeros(len(ets)) 

    earth_HCI_pos = np.array(earth_HCI_pos)
    earth_hlat = np.zeros(len(ets))
    earth_hlon = np.zeros(len(ets))

    venus_HCI_pos = np.array(venus_HCI_pos)
    venus_hlat = np.zeros(len(ets))

    for i, void in enumerate(ets):
        [solo_hdis[i],solo_hlon[i],solo_hlat[i]] = sp.reclat(solo_HCI_pos[i,:])
        [buffer,earth_hlon[i],earth_hlat[i]] = sp.reclat(earth_HCI_pos[i,:])
        [buffer,buffer,venus_hlat[i]] = sp.reclat(venus_HCI_pos[i,:])

    solo_hdis = solo_hdis/AU   
    solo_hlat *= sp.dpr()
    earth_hlat *= sp.dpr()

    solo_hlon *= sp.dpr()
    earth_hlon *= sp.dpr()

    solo_HCI_pos /= AU
    earth_HCI_pos /= AU
    venus_HCI_pos /= AU

    begin = et2datetime64(et_bounds[0])
    end = et2datetime64(et_bounds[1])

    begin_str = np.datetime_as_string(begin, unit = 'D')[0]
    end_str = np.datetime_as_string(end, unit = 'D')[0]
    
    if time <= begin or time >= end:
        print(f"Requested time is not covered by the Kernel. Begin is {begin_str}. End is {end_str}")

    else:
        print("\nSolar Orbiter Orbit Information \n")

        print(f"Your desired time is {time} \n")

        desired_et = datetime642et(time)

        nearest_idx_et = find_nearest(ets, desired_et)
        
        HCI_pos = solo_HCI_pos[nearest_idx_et]

        #print(HCI_pos)

        distance_to_sun = solo_hdis[nearest_idx_et]

        print(f"Distance to the Sun is: {distance_to_sun:.3g} AU \n")

        solo_hlat_inst = solo_hlat[nearest_idx_et]

        solo_hlon_inst = solo_hlon[nearest_idx_et]

        print(f"Latitude = {solo_hlat_inst:.4g} \n") #range between -90 and 90 - angle from XY plane of the ray from origin to point

        print(f"Longitude = {solo_hlon_inst:.4g} \n") #range is between -180 and 180

        hrt_fov_deg = 1024*0.5/3600 #degrees of half the fov in one direction

        hrt_fov_radians = hrt_fov_deg/180 *math.pi

        solar_radius = 696340

        hrt_sol_radius = distance_to_sun*hrt_fov_radians*AU/solar_radius #(in km)

        print(f"Solar Radius visible in HRT FOV along X or Y direction is {hrt_sol_radius:.4g} solar radii \n")

        hrt_fov_deg_diag = math.sqrt(2*1024**2)*0.5/3600 #degrees of half the fov in one direction

        hrt_fov_radians_diag = hrt_fov_deg_diag/180 *math.pi

        solar_radius = 696340

        hrt_sol_radius_diag = distance_to_sun*hrt_fov_radians_diag*AU/solar_radius

        print(f"Solar Radius visible in HRT FOV along diagonals is {hrt_sol_radius_diag:.4g} solar radii \n")

        print(f"Light time to Earth is {ltime_gse[nearest_idx_et]:.4g} seconds\n")

        hrt_pixel_size = 0.5/3600/180*math.pi*distance_to_sun*AU

        print(f"HRT Pixel size on solar surface is {hrt_pixel_size:.4g} km \n")

        earth_hlon_inst = earth_hlon[nearest_idx_et]
        earth_hlat_inst = earth_hlat[nearest_idx_et]

        solo_out_earth_line = abs(solo_hlon_inst - earth_hlon_inst)

        if solo_out_earth_line > 180:

            solo_out_earth_line = 180-abs(solo_hlon_inst) + 180-abs(earth_hlon_inst)

        print(f"Solo latitude out of Earth-Sun Line is {solo_out_earth_line:.4g} degrees \n")

        print(f"solo_hlon = {solo_hlon_inst:.4g} degrees ")
        print(f"earth_hlon = {earth_hlon_inst:.4g} degrees\n")

        print(f"solo_hlat = {solo_hlat_inst:.4g} degrees")
        print(f"earth_hlat = {earth_hlat_inst:.4g} degrees\n")

    if plot:

        positions = solo_GSE_pos.T # positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
        fig = plt.figure(figsize=(9, 9))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(positions[0], positions[1], positions[2])
        plt.title(f'Solo GSE Position from {begin_str} to {end_str}')
        ax.set_xlabel("X (AU)")
        ax.set_ylabel("Y (AU)")
        ax.set_zlabel("Z (AU)")
        #plt.show()

        plt.savefig("./plots/solo_orbit_plot_gse")

        positions_hci = solo_HCI_pos.T # positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
        fig = plt.figure(figsize=(9, 9))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(positions_hci[0], positions_hci[1], positions_hci[2])
        plt.title(f'Solo HCI Position from {begin_str} to {end_str}')
        ax.set_xlabel("X (AU)")
        ax.set_ylabel("Y (AU)")
        ax.set_zlabel("Z (AU)")
        #plt.show()

        plt.savefig("./plots/solo_orbit_plot")

        inst_pos_hci = HCI_pos.T
        inst_earth_hci = earth_HCI_pos[nearest_idx_et].T
        inst_venus_hci = venus_HCI_pos[nearest_idx_et].T
        fig = plt.figure(figsize=(9, 9))
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(inst_pos_hci[0], inst_pos_hci[1], inst_pos_hci[2], label = "SOLO")
        ax.scatter(inst_earth_hci[0], inst_earth_hci[1], inst_earth_hci[2], label = "EARTH")
        ax.scatter(0, 0, 0, label = "SUN")
        plt.legend()
        plt.title(f'Solo HCI Position from at {time}')
        ax.set_xlabel("X (AU)")
        ax.set_ylabel("Y (AU)")
        ax.set_zlabel("Z (AU)")
        
        #ax.view_init(elev = 90, azim = 90)

        plt.savefig("./plots/inst_position_3d")

        points = 100
        earth_orbit_traj = earth_HCI_pos[nearest_idx_et-points*10: nearest_idx_et+5*points].T
        venus_orbit_traj = venus_HCI_pos[nearest_idx_et-points*10: nearest_idx_et+5*points].T

        solo_orbit_traj = solo_HCI_pos[:nearest_idx_et+points].T #nearest_idx_et-points*10

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(inst_pos_hci[0], inst_pos_hci[1], label = "SOLO", color = "blue", s = 60, edgecolors='black')
        ax.scatter(inst_earth_hci[0], inst_earth_hci[1], label = "EARTH", color = "green", s = 60, edgecolors='black')
        ax.scatter(inst_venus_hci[0], inst_venus_hci[1], label = "VENUS", color = "orange", s = 60, edgecolors='black')

        ax.plot(earth_orbit_traj[0],earth_orbit_traj[1], linestyle = '--', color = "green")
        ax.plot(venus_orbit_traj[0],venus_orbit_traj[1], linestyle = '--', color = "orange")
        #earth_traj = plt.Circle((0,0), 1, fill = False, color = "green", linestyle = '--')
        #ax.add_patch(earth_traj) earth's trajectory not perfect circle - interested in the difference
        ax.plot(solo_orbit_traj[0],solo_orbit_traj[1], linestyle = '--', color = "blue")
        ax.scatter(0, 0, label = "SUN", color = "yellow", s = 100, edgecolors='black')
        plt.legend()
        plt.title(f'Solo HCI Position at {time} on Ecliptical plane')
        ax.set_xlabel("X (AU)")
        ax.set_ylabel("Y (AU)")
        
        #ax.view_init(elev = 90, azim = 90)

        plt.savefig("./plots/inst_position_2d")


    #plt.

if len(sys.argv) > 1:

    input_time = np.datetime64(str(sys.argv[1]))

    if len(sys.argv) > 2:
        
        plot_bool = sys.argv[2]

    else:

        plot_bool = False

else:

    input_time = np.datetime64('2020-05-28T17:00')

    plot_bool = False

main(time = input_time, plot = plot_bool)