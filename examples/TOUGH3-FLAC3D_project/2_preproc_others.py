"""
1) CELL 1
Script to load the toughio MESH plain text file and find the injection cell (point closest to 0, 0, 0) and all cells
corresponding to the monitoring boreholes sections, after coordinate conversion from Bedretto coordinates to MESH
2) CELL 2
...
@ by Victor Clasen Repollés (victor.clasen@sed.ethz.ch)
"""

import toughio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from scipy.spatial import KDTree

# import utils

#%% Load right plotting settings

import matplotlib
matplotlib.use('Qt5Agg')

# setting up for plotting
plt.rcParams['figure.dpi'] = 70
plt.rcParams['font.size'] = 24
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.facecolor'] = 'white'

#%% Load MESH and find injection cell and all cells at all monitoring points in the entire MB's length

# 1) Load MESH and find injection cell id

project_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/'

#mesh_dict = toughio.read_mesh(project_path + '0_preprocessing_OUTPUT/MESH_bound') # coarser mesh
mesh_dict = toughio.read_mesh(project_path + '0_preprocessing_OUTPUT/fine_mesh_3110/MESH') # finer mesh

# Find out the injection cell (cell closest to the injection point, which is (0, 0, 0))
cell_ids = np.asarray(list(mesh_dict['elements'].keys()))

vals = []
center_pts = []
for i in range(len(cell_ids)):
    vals.append(np.sum(np.square(mesh_dict['elements'][cell_ids[i]]['center'])))
    center_pts.append(mesh_dict['elements'][cell_ids[i]]['center'])
inj_ind = np.asarray(vals).argmin()
inj_cell = cell_ids[inj_ind]

# injection cell (ST1)
print(str(inj_cell) + ': ' + str(mesh_dict['elements'][inj_cell]['center']))

# 2) Load borehole information and convert tunnel coordinates to MESH coordinates

ST1_inj_depth = 107.675         # depth along ST1

ST1_df = pd.read_csv(project_path + '0_boreholes/ST1_borehole_information.csv', index_col=0)

idx = np.argmin(np.abs(np.array(ST1_df['Depth (m)']) - ST1_inj_depth))                                                  # index of injection point at ST1

# use spatial data structure KD-Tree to find the closest 3D coordinates
kdtree = KDTree(center_pts)

# create the rotation matrix from the Bedretto coordinates to the coordinates of the mesh ...
"""
55° rotation clockwise around the z-axis (elevation)
"""
def z_rot_matrix(angle):
    return np.array([[np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0],
                     [-np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
                     [0, 0, 1]])

rotation_angle = 55

# 1) perform a clockwise rotation around the z-axis
rot_z = z_rot_matrix(rotation_angle)
# 2) OR perform a counterclockwise rotation around the z-axis (with negative angle)
# rot_z = utils.rotation_matrix(-rotation_angle, axis=2)

boreholes_fileslist = [
    'MB1_borehole_information',
    'MB2_borehole_information',
    'MB4_borehole_information',
    'MB5_borehole_information',
    'MB8_borehole_information',
    'ST1_borehole_information',
]

coord_tunnel = ['Easting (m)', 'Northing (m)', 'Elevation (m)']
coord_axis = ['X', 'Y', 'Z']

for i in range(len(boreholes_fileslist)):
    MB_df = pd.read_csv(project_path + '0_boreholes/' + boreholes_fileslist[i] + '.csv', index_col=0)
    new_coords_cs1 = np.zeros((len(MB_df), 3))
    new_coords_cs2 = np.copy(new_coords_cs1)
    for j in range(len(new_coords_cs1[0, :])):      # compute tunnel coord distance from ST1 for all points in the monitoring borehole
        new_coords_cs1[:, j] = np.array(MB_df[coord_tunnel[j]]) - np.array(ST1_df[coord_tunnel[j]])[idx]
    # swap first two columns because Easting = Y, and Northing = X (this is not correct!)
    # new_coords[:, [0, 1]] = new_coords[:, [1, 0]]
    for l in range(len(new_coords_cs1)):
        new_coords_cs2[l, :] = np.dot(rot_z, new_coords_cs1[l, :]) * np.array([-1.0, -1.0, 1.0]) # easting is opposite to positive x-axis direction
    for j in range(len(new_coords_cs2[0, :])):      # correct to slight variation in the injection point center point in the mesh, since it is not exactly at 0, 0, 0
        MB_df[coord_axis[j]+'_comp'] = new_coords_cs2[:, j] + mesh_dict['elements'][inj_cell]['center'][j]
    coordinates = list(np.asarray((list(MB_df.X_comp), list(MB_df.Y_comp), list(MB_df.Z_comp))).T)
    closest_cells = []
    materials_list = []
    cells_center_pts = np.empty((len(coordinates), 3))
    count = 0
    for coord in coordinates:                   # search for indices with closest coordinates, copy the coordinates and save the cell id, add to dataframe
        distance, index = kdtree.query(coord)
        for j in range(len(cells_center_pts[0, :])):
            cells_center_pts[count, j] = center_pts[index][j]
        closest_cells.append(cell_ids[index])
        materials_list.append(mesh_dict['elements'][cell_ids[index]]['material'])
        count = count + 1
    MB_df['cell_id'] = closest_cells
    MB_df['domain'] = materials_list
    for j in range(len(new_coords_cs2[0, :])):
        MB_df[coord_axis[j]+'_mesh'] = cells_center_pts[:, j]
    MB_df.to_csv(project_path + '0_boreholes/finer_mesh_3.0/' + boreholes_fileslist[i] + '_edited.csv', index=True)

#%% Filter and plot hydraulic data for INFILE

def plot_raw_hydraulics(df_raw, df_filtered, fs=20):
    """
    Plots flow rate (raw and filtered for the model GENER) and observed pressure in a given stimulation interval
    """

    fig = plt.figure(figsize=[24, 15])
    # fig = plt.figure(figsize=[12, 12])
    gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.0, wspace=0.0)
    ax1 = fig.add_subplot(gs[0])
    ax11 = ax1.twinx()

    ax1.plot(df_raw.time / 3600.0, df_raw.flowrate * 60.0, color='tab:cyan', ls='-', lw=4, label='inj. flow rate (unfiltered)')
    ax1.plot(df_filtered.time / 3600.0, df_filtered.flowrate * 60.0, color='tab:blue', ls='-', lw=2, label='TOUGH injection (filtered)')
    ax11.plot(df_raw.time / 3600.0, df_raw.pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_11, labels_11 = ax11.get_legend_handles_labels()
    lns = lines_1 + lines_11
    lbls = labels_1 + labels_11

    ax1.legend(lns, lbls, loc='best', fontsize=fs)
    # ax1.legend(loc='best', fontsize=fs)
    ax1.set_xlabel('Time[hours] from start of simulation', fontsize=fs+2)
    ax1.set_ylabel('Flow rate [L/min]', fontsize=fs+2)
    ax1.yaxis.set_tick_params(labelsize=fs+2)
    ax1.set_xlim((0.0, 48.0))
    # ax1.set_ylim((0.0, 155.0))
    ax11.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    ax11.yaxis.set_tick_params(labelsize=fs+2)

    fig.savefig('hydraulics_plot.svg')
    fig.show()

    return

input_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/DSS_DTS_py_project/Valter_phase1_interval13/'

# load hydraulics
hydraulics = pd.read_hdf(input_path + 'Hydraulics_unfiltered_int13.h5', key='df', mode='r')
# do a copy
hydraulics_copy = hydraulics.copy()
# reset index
hydraulics_copy.reset_index(inplace=True)
# load key times
times = pd.read_csv('hydraulic_times', sep=',')
times.rename(columns={'TIMES': 'time', 'AVERAGING': 'identifier'}, inplace=True)

# hydraulics_filtered = hydraulics_copy.merge(times, left_on='time', right_on='time', how='inner')

flowrate_new = np.zeros((len(times)))
flowrate_new[0] = hydraulics.flowrate[0]
for i in range((len(flowrate_new)-1)):
    if times.identifier[i]==0:
        ind = hydraulics_copy.index[hydraulics_copy['time']==times['time'][i+1]].to_list()[0]
        flowrate_new[i+1] = hydraulics_copy.iloc[ind]['flowrate']
    elif times.identifier[i]==1:
        ind_low = hydraulics_copy.index[hydraulics_copy['time']==times['time'][i]].to_list()[0]
        ind_high = hydraulics_copy.index[hydraulics_copy['time']==times['time'][i+1]].to_list()[0]
        flowrate_new[i+1] = hydraulics_copy.iloc[ind_low:ind_high+1]['flowrate'].mean()
    else:
        flowrate_new[i+1] = np.nan

# get the lower and higher bounds of the backflow
ind = times.index[times['identifier']==2].to_list()
ind_low = hydraulics_copy.index[hydraulics_copy['time']==times.time[ind[0]+1]].to_list()[0]
ind_high = hydraulics_copy.index[hydraulics_copy['time']==times.time[ind[-1]+1]].to_list()[0]
hyd_backflow = hydraulics_copy.iloc[ind_low:ind_high+1]
# resampling the backflow
hyd_backflow = hyd_backflow.resample('120s', on='dttime').median()
# TODO: RESCALING OF BACKFLOW (in case it is necessary if simulation breaks down) + COARSER DISCRETIZATION
hyd_backflow['flowrate'] = hyd_backflow['flowrate'] / 5.0 # 5 works good!
# find the index of the first NaN value
vals_nan = np.where(np.isnan(flowrate_new))[0]
if vals_nan.size > 0:
    # if Nan is found, remove all NaN values
    flowrate_new_ = flowrate_new[~np.isnan(flowrate_new)]
# delete also the time entries
mask = np.ones(np.asarray(times.time).shape, dtype=bool)
mask[vals_nan] = False
time_new_ = np.asarray(times.time)[mask]
# include the backflow
flowrate_array = np.insert(flowrate_new_, vals_nan[0], np.asarray(hyd_backflow.flowrate), axis=0)
time_array = np.insert(time_new_, vals_nan[0], np.asarray(hyd_backflow.time), axis=0)
# get the dttime column also
hyd_backflow.reset_index(inplace=True)
# also add dttime and pressure (for avoiding problems in later routines for plotting)
overlap_times = hydraulics_copy[hydraulics_copy['time'].isin(time_array)].index
pressure_array = np.insert(np.asarray(hydraulics_copy['pressure'][overlap_times.values]),
                           vals_nan[0], np.asarray(hyd_backflow.pressure)[1:-1], axis=0)
dttime_array = np.insert(np.asarray(hydraulics_copy['dttime'][overlap_times.values]),
                         vals_nan[0], np.asarray(hyd_backflow.dttime)[1:-1], axis=0)
# create new dataframe
columns = ['time', 'flowrate']
hydraulics_filtered = pd.DataFrame(columns=columns)
hydraulics_filtered['time'] = time_array
hydraulics_filtered['flowrate'] = flowrate_array
hydraulics_filtered['pressure'] = pressure_array
hydraulics_filtered['dttime'] = dttime_array
hydraulics_filtered.set_index('dttime', inplace=True)

# SAVE #
hydraulics_filtered.to_hdf(input_path + 'Hydraulics_filtered_int13_alt4.h5', key='df', mode='w')

# # PLOT #
# plot_raw_hydraulics(hydraulics, hydraulics_filtered, fs=24)

#%% CALCULATE PARAMETERS FOR WELL CONNECTION TO MESH

def get_params(input_params, r_well, V_total, known='radius_well'):
    if known=='radius_well':
        # cylinder
        A_l_side = 2 * np.pi * r_well * input_params['h_cell']              # area of the side of the cylinder
        A_h_side = np.pi * r_well**2                                        # area of the top/bottom of the cylinder
        #A_total = A_l_side + 2*A_h_side                                     # total area of the cylinder
        V_single = np.pi * ((r_well)**2) * input_params['h_cell']           # volume of one cylinder
        V_total = input_params['n_cells'] * V_single                        # total volume of the well
        # cell
        l = input_params['l_cell'] - r_well                                 # distance from the cylinder to the intersecting cell area (in l direction)
        d = input_params['d_cell'] - r_well                                 # distance from the cylinder to the intersecting cell area (in d direction)
    elif known=='volume_total':
        V_single = V_total / input_params['n_cells']
        r_well = np.sqrt(V_single/np.pi*input_params['h_cell'])
        # calculate now the area of the side and the top/bottom
        A_l_side = 2 * np.pi * r_well * input_params['h_cell']
        A_h_side = np.pi * r_well ** 2
        # cell
        l = input_params['l_cell'] - r_well
        d = input_params['d_cell'] - r_well

    print('--------------------------------')
    print('Flow in XZ direction up parallel to the fault: ')
    print('ROCK1WELL1: | ISOT: 1 | D1: ' + str(np.round(l, decimals=3)) + ' | D2: ' + str(np.round(r_well, decimals=3)) + ' | AREAX: ' + str(np.round(A_l_side, decimals=3)) + ' | BETAX: ' + str(np.round(np.cos(np.deg2rad(90 + input_params['dip_fault'])), decimals=3)) + ' |')
    print('Flow in XZ direction down parallel to the fault: ')
    print('ROCK1WELL1: | ISOT: 1 | D1: ' + str(np.round(l, decimals=3)) + ' | D2: ' + str(np.round(r_well, decimals=3)) + ' | AREAX: ' + str(np.round(A_l_side, decimals=3)) + ' | BETAX: ' + str(np.round(np.cos(np.deg2rad(90 - input_params['dip_fault'])), decimals=3)) + ' |')
    print('Flow in positive/negative Y direction along the fault: ')
    print('ROCK1WELL1: | ISOT: 2 | D1: ' + str(np.round(d, decimals=3)) + ' | D2: ' + str(np.round(r_well, decimals=3)) + ' | AREAX: ' + str(np.round(A_l_side, decimals=3)) + ' | BETAX: ' + str(np.round(np.cos(np.deg2rad(90)), decimals=3)) + ' |')
    print('Flow between well elements perpendicular to the fault: ')
    print('WELL1WELL2: | ISOT: 3 | D1: ' + str(np.round(input_params['h_cell']/2, decimals=3)) + ' | D2: ' + str(np.round(input_params['h_cell']/2, decimals=3)) + ' | AREAX: ' + str(np.round(A_h_side, decimals=3)) + ' | BETAX: ' + str(np.round(np.cos(np.deg2rad(input_params['dip_fault'])), decimals=3)) + ' |')
    print('--------------------------------')
    print('Volume of a single cell of the well: ')
    print('Volume single well cell: ')
    print(str(np.round(V_single, decimals=3)))
    print('Volume total well: ')
    print(str(np.round(V_total, decimals=3)))
    print('Radius of the well: ')
    print(str(np.round(r_well, decimals=3)))
    print('--------------------------------')

    return

# input_params = {
#     'h_cell': 2.5,          # height of the cell (in XZ direction parallel to the fault)
#     'l_cell': 5.0,          # length of the cell (in XZ direction perpendicular to the fault)
#     'd_cell': 2.0,          # depth of the cell (in Y direction along the fault)
#     'n_cells': 5,           # number of cells intersecting with the injection well
#     'dip_fault': 60.0,      # dip of the fault
# }
#
# r_well = 0.075              # radius of the well
# V_total = 0.66838             # total volume of the well
#
# get_params(input_params, r_well, V_total, known='volume_total')