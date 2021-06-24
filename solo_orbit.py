import numpy as np
import os
from numpy.core.einsumfunc import _einsum_path_dispatcher
import spiceypy as sp
from pathlib import Path
import spiceypy.utils.support_types as stypes
import json
import xml.etree.ElementTree as xmlet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import more_itertools as mit

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

def main():
    AU = 149598000.0
    "Does the Stuff"
    config = read_json("orbit_config_js.json")
    mk_path = Path(config["metakernel"]["path"])
    mk_name = config["metakernel"]["name"]
    loaded=loadkernel(mk_path, mk_name)

    et_bounds=get_solo_coverage(mk_path)

    ets = np.arange(et_bounds[0],et_bounds[1],21600)
    ets[len(ets)-1]=et_bounds[1]

    [solo_GSE_pos, ltime] = sp.spkpos("SOLO",ets,"SOLO_GSE","NONE","EARTH")
    [solo_HCI_state, ltime] = sp.spkezr("SOLO",ets,"SUN_INERTIAL","NONE","SUN") 

    solo_HCI_state = np.array(solo_HCI_state)
    solo_GSE_pos = np.array(solo_GSE_pos)/AU 
    solo_HCI_pos = solo_HCI_state[:,0:3]
    solo_HCI_vel = solo_HCI_state[:,3:6]    
    solo_hdis = np.zeros(len(ets))
    solo_hlon = np.zeros(len(ets))
    solo_hlat = np.zeros(len(ets)) 

    solo_hdis = solo_hdis/AU   

    solo_hlat = solo_hlat*sp.dpr()

    #print(ets, ets.shape)
    #print(solo_GSE_pos, solo_GSE_pos.shape)

    # plt.figure()
    # plt.plot(ets, solo_GSE_pos)

    begin = et2datetime64(et_bounds[0])

    end = et2datetime64(et_bounds[1])

    begin_str = np.datetime_as_string(begin, unit = 'D')[0]
    end_str = np.datetime_as_string(end, unit = 'D')[0]

    print(begin_str)
    print(end_str)

    #print(solo_HCI_pos, solo_HCI_pos.shape)
    #print(solo_hdis, solo_hdis.shape)
    #print(solo_hlat, solo_hlat.shape)

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

    #plt.





main()