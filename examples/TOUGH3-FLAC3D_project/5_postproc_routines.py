"""
Different routines for postprocessing results from TOUGH and TOUGH-FLAC outputs
@ by Victor Clasen Repollés (victor.clasen@sed.ethz.ch)
"""

# Import the required libraries
import numpy as np
import pandas as pd
import pickle
import toughio

from scipy.spatial.transform import Rotation

# import library for functions
import utils

#%% READ & DISTRUBUTE OUTPUT DATA FOR MONITORING BOREHOLES AND PERFORM STRAIN ROTATION

# path of the simulation results
PATH_RES = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/3_Tough3Flac3Dv7_coupled_inj/041223/case_1/'
# PATH_RES = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TF_testrun/00_base_case/3_coupling/f3out/'
# path of the monitoring boreholes information
PATH_MBS = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/0_boreholes/finer_mesh_3.0/'

boreholes_fileslist = [
    'MB1_borehole_information_edited.csv',
    'MB4_borehole_information_edited.csv',
    'MB5_borehole_information_edited.csv',
    'MB8_borehole_information_edited.csv',
]

results_fileslist = [
    'hist_MB1.csv',
    'hist_MB4.csv',
    'hist_MB5.csv',
    'hist_MB8.csv',
]

rotated_strain_fileslist = [
    'MB1_strain_rotated.csv',
    'MB4_strain_rotated.csv',
    'MB5_strain_rotated.csv',
    'MB8_strain_rotated.csv',
]

n = 0
for FILE in boreholes_fileslist:
    MB = pd.read_csv(PATH_MBS + FILE)
    # create new reduced dataframe (to unique cell ids)
    MB_reduced = pd.DataFrame()
    MB_reduced['cell_id'] = MB['cell_id'].unique()

    columns_names = ['Depth (m)', 'Azimuth mean (deg)', 'Dip mean (deg)',
                     'Easting (m)', 'Northing (m)', 'Elevation (m)', 'X_comp', 'Y_comp', 'Z_comp']
    dump_matrix = np.zeros((len(MB_reduced['cell_id']), len(columns_names)))
    for i in range(len(MB_reduced['cell_id'])):
        index_ids = np.where(MB['cell_id']==MB_reduced['cell_id'][i])[0]
        for j in range(len(columns_names)):
            dump_matrix[i, j] = np.mean(MB[columns_names[j]][index_ids[0]:index_ids[-1]+1])
    # add dump matrix to new dataframe
    col = 0
    for col_name in columns_names:
        MB_reduced[col_name] = dump_matrix[:, col]
        col = col + 1

    # now read all output variables
    variables_names = ['temperature', 'pore_pressure', 'stress_xx', 'stress_yy', 'stress_zz',
                       'stress_xz', 'stress_yz', 'stress_xy', 'strain_xx', 'strain_yy', 'strain_zz',
                       'strain_xz', 'strain_yz', 'strain_xy', 'disp_x', 'disp_y', 'disp_z']
    MB_results = {}

    val = 0
    for v in range(len(variables_names)):
        MB_results[variables_names[v]] = np.asarray(pd.read_csv(PATH_RES + results_fileslist[n], header=None, skiprows=1, usecols=list(np.arange(1, len(MB_reduced['cell_id']) + 1, 1) + val)))
        val = val + len(MB_reduced['cell_id'])

    # get the times array
    sim_times = np.asarray(pd.read_csv(PATH_RES + results_fileslist[n], header=None, skiprows=1, usecols=[0]))

    # create an empty entry to the results dictionary for the rotated strain
    MB_results['strain_rotated'] = np.zeros_like(MB_results['strain_xx'])
    for t in range(len(sim_times)):     # times
        for d in range(len(MB_reduced['cell_id'])):    # depth/distance along the monitoring borehole
            # collect the strain tensor
            strain_tensor = np.array([
                [MB_results['strain_xx'][t, d], MB_results['strain_xy'][t, d], MB_results['strain_xz'][t, d]],
                [MB_results['strain_xy'][t, d], MB_results['strain_yy'][t, d], MB_results['strain_yz'][t, d]],
                [MB_results['strain_xz'][t, d], MB_results['strain_yz'][t, d], MB_results['strain_zz'][t, d]]
            ])
            # TODO: CHECK HOW I CAN CALCULATE THE SAME CORRECTLY WITH MY METHOD
            # # perform a rotation in z-axis direction
            # alpha = (MB_reduced['Azimuth mean (deg)'][d] - 55.0) * - 1.0
            # """
            # Explanation to the angle alpha:
            # We correct for the already rotated mesh w.r.t. northing and multiply times -1.0 to rotate clockwise, as the
            # function is defined for counterclockwise rotation along the specified axis (with 2 being the z axis, default)
            # """
            # strain_tensor_rot = utils.rotate_tensor(strain_tensor, alpha, axis=2)
            # # now we get the dip angle from the data
            # dip = np.radians(MB_reduced['Dip mean (deg)'][d])
            # """
            # Since the y-axis of the mesh was placed in the borehole azimuth direction, we dip in y-axis direction
            # """
            # # we define the directional vector along the borehole (tangential direction)
            # vector = np.array([0, np.cos(dip), np.sin(dip)])
            # MB_results['strain_rotated'][t, d] = np.dot(np.dot(strain_tensor_rot, vector), vector)

            # perform a rotation in z-axis direction
            # r = Rotation.from_euler('z', 55.0, degrees=True)
            # strain_tensor_rot = r.apply(strain_tensor)
            strain_tensor_rot = utils.rotate_tensor(strain_tensor, 55, 2)

            # now we get the azimuth and the dip angle from the data
            alpha = MB_reduced['Azimuth mean (deg)'][d]
            dip = MB_reduced['Dip mean (deg)'][d]

            # r = Rotation.from_euler('zy', [-alpha, dip], degrees=True)
            # strain_tensor_rot = r.apply(strain_tensor_rot)
            # TODO: CHECK DIFFERENCE between two upper rows and two lower rows
            strain_tensor_rot = utils.rotate_tensor(strain_tensor_rot, -alpha, 2)
            strain_tensor_rot = utils.rotate_tensor(strain_tensor_rot, -dip, 1)

            MB_results['strain_rotated'][t, d] = np.sum(strain_tensor_rot, axis=0)[0]

    # save rotated strain to file
    SR = np.insert(np.insert(MB_results['strain_rotated'], 0, sim_times.T[0], axis=1),
                   0, np.insert(np.asarray(MB_reduced['Depth (m)']), 0, np.nan, axis=0), axis=0)
    df = pd.DataFrame(SR)
    df.to_csv(PATH_MBS + rotated_strain_fileslist[n], index=False, header=False)
    n = n + 1


#%% DISTRUBUTE ALL BOREHOLES TO DICTIONARY FORMAT AND SAVE TO PICKLE

# create empty dictionary for all sensors to dump the data
DSS_SENSORS = ['MB1', 'MB4', 'MB5', 'MB8']
# DSS_SENSORS = ['MB1']
DSS_dict = utils.create_dict(DSS_SENSORS)

PATH_IN = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/0_boreholes/finer_mesh_3.0/'

# PATH_OUT = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/DSS_DTS_py_project/Valter_phase1_interval13/'
PATH_OUT = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/3_Tough3Flac3Dv7_coupled_inj/041223/case_1/'

FILENAMES = [
    'MB1_strain_rotated.csv',
    'MB4_strain_rotated.csv',
    'MB5_strain_rotated.csv',
    'MB8_strain_rotated.csv'
]

for i in range(len(FILENAMES)):
    depth_array = np.asarray(pd.read_csv(PATH_IN + FILENAMES[i], header=None, nrows=1))[0][1:]
    time_array = np.asarray(pd.read_csv(PATH_IN + FILENAMES[i], usecols=[0])).T
    strain_matrix = pd.read_csv(PATH_IN + FILENAMES[i], header=None, skiprows=1, usecols=list(np.arange(1, len(depth_array)+1, 1)))
    # save all variables into dictionary
    DSS_dict[DSS_SENSORS[i]]['distance'] = depth_array
    DSS_dict[DSS_SENSORS[i]]['time'] = time_array
    DSS_dict[DSS_SENSORS[i]]['strain'] = strain_matrix

with open(PATH_OUT + 'Int13_DSS_TOUGH-FLAC_X_041223.pkl', 'wb') as f:
    pickle.dump(DSS_dict, f)

#%% PLOT STRAIN RESULTS FROM MODELLING

# # routines for plotting
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
#
# # setting up for plotting
# plt.rcParams['figure.dpi'] = 70
# plt.rcParams['font.size'] = 24
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['figure.facecolor'] = 'white'
#
# sensor = 'MB8'
#
# fig = utils.plot_strain(DSS_dict,
#                         fs=20,
#                         bottomdepth=150,
#                         topdepth=90,
#                         strain_min=-100,
#                         strain_max=100,
#                         time_start=0,
#                         time_end=14,
#                         sensor=sensor,
#                         cbar_range='max_min')
# fig.show()

#%% READ SAVE/INCON FILE AND DELETE POROSITY VALUES
# """
# This routine reads the created SAVE/INCON file by the steady state calculation and deletes the porosity entries.
# Then, the INFILE for the injection must contain porosity values for the domains in the block ROCKS with the START block
# as well
# """
#
# filepath = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/execute/'
#
# print('A')
# file = toughio.read_output(filepath + 'SAVE', file_format='save')
# print('B')
#
# # del file.data['porosity']
#
# # save new file
# toughio.write_output(filepath + 'SAVE_', file, file_format='save')