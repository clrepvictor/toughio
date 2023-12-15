"""
Different read and plotting routines
@ by Victor Clasen Repollés (victor.clasen@sed.ethz.ch)
"""

# Import the required libraries
import toughio
import os
import pickle
import datetime
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from matplotlib import gridspec
# from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from collections import OrderedDict

import utils

import matplotlib
matplotlib.use('Qt5Agg')

# setting up for plotting
plt.rcParams['figure.dpi'] = 90
plt.rcParams['font.size'] = 40
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.facecolor'] = 'white'

#%% Convert OUTPUT file (and plot in paraview)
# """
# Converts the OUTPUT from TOUGH3 as .csv file to a .xdmf format to be read by paraview
#
# 1) go to the terminal and open the folder where the file 'OUTPUT_ELEME.csv' is saved.
# 2) type: path/toughio-export --file-format xdmf OUTPUT_ELEME.csv
#    -> e.g.: /home/victor/.local/bin/toughio-export --file-format xdmf OUTPUT_ELEME.csv
# 3) open paraview and open exported file for visualization
# 4) alternatively, run it here:
# """
#
# path = '/home/victor/.local/bin/toughio-export --file-format '
# format = 'xdmf '    # options are: tecplot for .dat, vtk for .vtk, vtu for .vtu, xdmf for .xdmf
# filepath = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/execute/'
# filename = 'OUTPUT_ELEME.csv'       # optionally add the path in front if in a different folder ...
#
# # export file
# os.system(path + format + filepath + filename)
#
# # execute paraview
# os.system('paraview')

#%% Create a 3D plot of the MESH
# """
# This routine is to create a 3D plot of the MESH showing the monitoring boreholes and the fault positions
# """
# import meshio
# import pyvista as pv
# import itasca as it
#
# input_path_1 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/'
#
# input_path_2 = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/0_preprocessing_OUTPUT/25.08/'
#
# mesh_dict = toughio.read_mesh(input_path_1 + 'MESH_injection')
#
# mesh_toughio_obj = toughio.from_meshio(meshio.flac3d.read(input_path_2 + 'OUTPUT1_mesh.f3grid'))
#
# mesh_itasca_obj = it.command("'zone import '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/0_preprocessing_OUTPUT/25.08/'")

# x_lim = np.array([-200.0, 200.0])
# y_lim = np.array([-200.0, 200.0])
# z_lim = np.array([-200.0, 200.0])
#
# grid = pv.RectilinearGrid(x_lim, y_lim, z_lim)
#
# grid.plot()

# cell_ids = list(mesh_dict['elements'].keys())
# material_list = []
# for i in range(len(cell_ids)):
#     material_list.append(mesh_dict['elements'][cell_ids[i]]['material'])


#%% READ, FILTER, AND INTERPOLATE INCON/SAVE FILES WITH MESH

# input_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/preprocessing_OUTPUT/'
#
# mesh_data = toughio.read_mesh(input_path + 'MESH')
#
# incon_data = toughio.read_input(input_path + 'INCON')
#
# ele_list = list(mesh_data['elements'].keys())
#
# # save coordinates form MESH
# x_pts = []
# y_pts = []
# z_pts = []
# # save primary variables from INCON
# pv_1 = []           # pressure
# pv_2 = []           # air mass fraction
# pv_3 = []           # temperature
# for i in range(len(ele_list)):
#     x_pts.append(mesh_data['elements'][ele_list[i]]['center'][0])
#     y_pts.append(mesh_data['elements'][ele_list[i]]['center'][1])
#     z_pts.append(mesh_data['elements'][ele_list[i]]['center'][2])
#     pv_1.append(incon_data['initial_conditions'][ele_list[i]]['values'][0])
#     pv_2.append(incon_data['initial_conditions'][ele_list[i]]['values'][1])
#     pv_3.append(incon_data['initial_conditions'][ele_list[i]]['values'][2])
#
# n = 2
# for l in range(8):
#     # delete every n-th element of the lists due to memory issues
#     del x_pts[::n]
#     del y_pts[::n]
#     del z_pts[::n]
#     del pv_1[::n]
#     del pv_2[::n]
#     del pv_3[::n]
#
# x_coord = np.unique(np.sort(np.asarray(x_pts)))
# y_coord = np.unique(np.sort(np.asarray(y_pts)))
# z_coord = np.unique(np.sort(np.asarray(z_pts)))
#
# X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord)
#
# pv_1_volume = griddata((np.asarray(x_pts), np.asarray(y_pts), np.asarray(z_pts)),
#                        np.asarray(pv_1), (X, Y, Z), method='nearest')
# pv_2_volume = griddata((np.asarray(x_pts), np.asarray(y_pts), np.asarray(z_pts)),
#                        np.asarray(pv_2), (X, Y, Z), method='nearest')
# pv_3_volume = griddata((np.asarray(x_pts), np.asarray(y_pts), np.asarray(z_pts)),
#                        np.asarray(pv_3), (X, Y, Z), method='nearest')
#
# #%% 3D PLOTTING
#
# # setting up for plotting
# plt.rcParams['figure.dpi'] = 60
# plt.rcParams['font.size'] = 30
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['figure.facecolor'] = 'white'
#
# # 3D plot of initial condition primary variable (INCON)
#
# kw = {
#     'vmin': pv_1_volume.min(),
#     'vmax': pv_1_volume.max(),
#     'levels': np.linspace(pv_1_volume.min(), pv_1_volume.max(), 10)
# }
#
# fig = plt.figure(figsize=[12, 12])
# gs = gridspec.GridSpec(nrows=1, ncols=1)
# ax1 = fig.add_subplot(gs[0], projection='3d')
#
# cp1 = ax1.contourf(X[:, :, 0], Y[:, :, 0], pv_1_volume[:, :, 0],
#                    zdir='z', offset=0, **kw)
# cp2 = ax1.contourf(X[0, :, :], pv_1_volume[0, :, :], Z[0, :, :],
#                    zdir='y', offset=0, **kw)
# cp3 = ax1.contourf(pv_1_volume[:, -1, :], Y[:, -1, :], Z[:, -1, :],
#                    zdir='x', offset=X.max(), **kw)
#
# # set limits of the plot from coordinate limits
# xmin, xmax = X.min(), X.max()
# ymin, ymax = Y.min(), Y.max()
# zmin, zmax = Z.min(), Z.max()
# ax1.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
#
# # set labels and ticks
# ax1.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
#
# # set colorbar
# fig.colorbar(cp3, ax=ax1, fraction=0.02, pad=0.1, label='Name [units]')
#
# fig.show()

#%% READ AND PLOT TIME SERIES OUTPUT FROM INJECTION AT A GRID BLOCK
"""
This routine is to compare the uncoupled TOUGH3 output with the measured data
"""

# 1) read the injection data (flowrate, pressure, and temperature #TO DO!)

input_path_1 = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/DSS_DTS_py_project/Valter_phase1_interval13/'

hydraulics_raw = pd.read_hdf(input_path_1 + 'Hydraulics_filtered_int13_alt3.h5', key='df', mode='r')
hydraulics_raw_unfiltered = pd.read_hdf(input_path_1 + 'Hydraulics_unfiltered_int13.h5', key='df', mode='r')

# Alternatively read temperature data from DTS (distributed temperature data)

input_path_1_1 = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/DSS_DTS_py_project/DTS_RAWDATA/Int 13/'

mat_file_st1 = sc.io.loadmat(input_path_1_1 + '20220323_20220330_DTS_ST1_Int13.mat')
mat_file_mb5 = sc.io.loadmat(input_path_1_1 + '20220323_20220330_DTS_MB5_Int13.mat')
mat_file_mb8 = sc.io.loadmat(input_path_1_1 + '20220323_20220330_DTS_MB8_Int13.mat')

P13 = 112.91 # depth of the pressure sensor in interval 13
P13_mb5 = 106.218 # depth of the sensor in MB5
P13_mb8 = 123.85 # depth of the sensor in MB8

temp_st1_all, temp_st1_sensor = utils.prepare_temp_data(mat_file_st1, hydraulics_raw.index[0], hydraulics_raw.time[hydraulics_raw.index[-1]], P13)
temp_mb5_all, temp_mb5_sensor = utils.prepare_temp_data(mat_file_mb5, hydraulics_raw.index[0], hydraulics_raw.time[hydraulics_raw.index[-1]], P13_mb5, apply_filter=True)
temp_mb8_all, temp_mb8_sensor = utils.prepare_temp_data(mat_file_mb8, hydraulics_raw.index[0], hydraulics_raw.time[hydraulics_raw.index[-1]], P13_mb8, apply_filter=True)

# 2) read time series data (pressure and temperature) from TOUGH3 simulation

# path of results of TOUGH simulation only
# input_path_2 = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/2_Tough3_inj_only/08.09/'
input_path_2 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/execute/'
# input_path_2 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/simulations/171123/04/'
# path of results of coupled TOUGH-FLAC simulation
input_path_3 = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/3_Tough3Flac3Dv7_coupled_inj/041223/case_1/'
# input_path_3 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/simulations/260923/06/'
# input_path_3 = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/3_Tough3Flac3Dv7_coupled_inj/28.09/case_2/'
input_path_4 = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/3_Tough3Flac3Dv7_coupled_inj/29.09/case_3/'

# TODO: delete the following plot or repair ...
# plot_hydraulics(hydraulics_raw, hydraulics_result, input_path_2, fs=24)

# 3) Plot injection point and monitoring points combined
"""
pres_plot_all -> plot of the pressure solutions from TOUGH3 and measured pressure at the corresponding sensor at the monitoring borehole
temp_plot_all -> plot of the temperature solutions from TOUGH3 and measured temperature with DTS at the monitoring borehole
"""

# raw hydraulic data from the monitoring points
hydraulics_raw_mb2 = pd.read_hdf(input_path_1 + 'Hydraulics_MB2_int13.h5', key='df', mode='r')
hydraulics_raw_mb5 = pd.read_hdf(input_path_1 + 'Hydraulics_MB5_int13.h5', key='df', mode='r')
hydraulics_raw_mb8 = pd.read_hdf(input_path_1 + 'Hydraulics_MB8_int13.h5', key='df', mode='r')

# TOUGH uncoupled and TOUGH-FLAC coupled
# hyd_results_all = utils.sort_tough_foft_output([input_path_2, input_path_3, input_path_4], ['FOFT_A9P73.csv', 'FOFT_A9Q68.csv', 'FOFT_A9O69.csv', 'FOFT_A9P65.csv'])
# hyd_results_all = utils.sort_tough_foft_output([input_path_2, input_path_3, input_path_4], ['FOFT_ACO76.csv', 'FOFT_ACR61.csv', 'FOFT_ACL64.csv', 'FOFT_ACO52.csv'])
hyd_results_all = utils.sort_tough_foft_output([input_path_2], ['FOFT_AI342.csv', 'FOFT_AHT90.csv', 'FOFT_AIB37.csv', 'FOFT_AI280.csv', 'FOFT_A1F33.csv'])

# pressure all
utils.pres_plot_all(hydraulics_raw, hydraulics_raw_unfiltered, hydraulics_raw_mb2, hydraulics_raw_mb5, hydraulics_raw_mb8,
              hyd_results_all, input_path_3, fs=24, coupled_simulation=False, double_coupled_simulation=False)

# temperature all
utils.temp_plot_all(hydraulics_raw, hydraulics_raw_unfiltered, temp_st1_sensor, temp_mb5_sensor, temp_mb8_sensor,
              hyd_results_all, input_path_3, fs=24, coupled_simulation=False, double_coupled_simulation=False, separate_y_axis=True)

# # pressure and temperature combined
# utils.plot_pres_temp_comb(hydraulics_raw, hydraulics_raw_unfiltered, temp_st1_sensor, hydraulics_raw_mb2, hydraulics_raw_mb5,
#                           temp_mb5_sensor, hydraulics_raw_mb8, temp_mb8_sensor, hyd_results_all, input_path_2, fs=24,
#                           coupled_simulation=False, double_coupled_simulation=False, separate_y_axis=True)

#%% READ AND PLOT TIME SERIES OUTPUT FOR SELECTED CELLS
# """
# For spatial and temporal evolution of a variable (e.g. pressure or temperature) at selected cells from the injection point
# """
#
# # input path for results from previous simulation
# # input_path_1 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/simulations/091123/mesh_3.0.2/temperature_analysis/foft_files/21/'
# input_path_1 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/simulations/151123/check_2/09/'
# # input path for results from new simulation
# input_path_2 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/execute/'
# # input_path_2 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/simulations/091123/mesh_3.0.2/temperature_analysis/foft_files/21/'
# # input path for results from a further simulation
# # input_path_3 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/simulations/091123/mesh_3.0.2/temperature_analysis/foft_files/14/'
# # path to save the plots
# plot_path = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/simulations/151123/check_2/10/'
#
# hyd_res_1 = utils.sort_tough_foft_output([input_path_1, input_path_2], ['FOFT_ASU63.csv', 'FOFT_AI342.csv', 'FOFT_AI347.csv', 'FOFT_AI352.csv', 'FOFT_AIA82.csv', 'FOFT_AIA87.csv'])
# hyd_res_2 = utils.sort_tough_foft_output([input_path_1, input_path_2], ['FOFT_ASU63.csv', 'FOFT_AI342.csv', 'FOFT_AI367.csv', 'FOFT_AI392.csv', 'FOFT_AI417.csv', 'FOFT_AI442.csv'])
# hyd_res_3 = utils.sort_tough_foft_output([input_path_1, input_path_2], ['FOFT_ASU63.csv', 'FOFT_AI342.csv', 'FOFT_AI317.csv', 'FOFT_AI292.csv', 'FOFT_AI267.csv', 'FOFT_AI242.csv'])
#
# hyd_res_all = (hyd_res_1, hyd_res_2, hyd_res_3)
#
# # Plot pressure and/or temperature at selected mesh cells
# utils.plot_cells_timeseries(hyd_res_all, plot_path, variable_y='temperature', scaling_time=3600.0, fs=20)
# # Plot pressure and/or temperature difference between selected mesh cells
# # utils.plot_cells_timeseries_diff(hyd_res_all, plot_path, variable_y='temperature', scaling_time=3600.0, fs=20)

#%% PLOT TEMPERATURE, PRESSURE, POROSITY, AND PERMEABILITY AT MONITORING POINTS FOR COUPLED SIMULATION

# proj_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/3_Tough3Flac3Dv7_coupled_inj/201123/case_17/'
#
# filename = 'MPs.csv'
#
# # dictionaries
# cells_all_ = [['AI342', 'AI347', 'AI352', 'AIA82', 'AIA87', 'AIA92'],     # in positive Y direction
#               ['AI342', 'AI367', 'AI392', 'AI417', 'AI442', 'AI467'],     # in XZ direction up the fault
#               ['AI342', 'AI317', 'AI292', 'AI267', 'AI242', 'AI217']]     # in XZ direction down the fault
#
# cells_all = list(OrderedDict.fromkeys(list(cells_all_[0] + cells_all_[1] + cells_all_[2])))
#
# res_df = pd.read_csv(proj_path + filename, header=None, skiprows=1)
# res_dict = dict()
#
# for i in range(len(cells_all)):
#     res_dict[cells_all[i]] = dict()
#     res_dict[cells_all[i]]['temperature'] = res_df.iloc[:, i+1]
#     res_dict[cells_all[i]]['pressure'] = res_df.iloc[:, i+17] / (1.e6)
#     res_dict[cells_all[i]]['porosity'] = res_df.iloc[:, i+33]
#     res_dict[cells_all[i]]['permeability'] = res_df.iloc[:, i+49]
#
# time = res_df.iloc[:, 0]
#
# # comparison with...
# proj_path_ = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/3_Tough3Flac3Dv7_coupled_inj/201123/case_18/'
#
# res_df_ = pd.read_csv(proj_path_ + filename, header=None, skiprows=1)
# res_dict_ = dict()
#
# for i in range(len(cells_all)):
#     res_dict_[cells_all[i]] = dict()
#     res_dict_[cells_all[i]]['temperature'] = res_df_.iloc[:, i+1]
#     res_dict_[cells_all[i]]['pressure'] = res_df_.iloc[:, i+17] / (1.e6)
#     res_dict_[cells_all[i]]['porosity'] = res_df_.iloc[:, i+33]
#     res_dict_[cells_all[i]]['permeability'] = res_df_.iloc[:, i+49]
#
# # Plotting
#
# # in positive Y-direction
# utils.plot_hydr_params((res_dict, res_dict_), cells_all_[0], time, proj_path_, 'monitoring_plot_y.png', fs=18, comparison=True)
# # # in XZ direction up the fault
# # utils.plot_hydr_params((res_dict, res_dict_), cells_all_[1], time, proj_path, 'monitoring_plot_xz_up.png', fs=18, comparison=True)
# # # in XZ direction down the fault
# # utils.plot_hydr_params((res_dict, res_dict_), cells_all_[2], time, proj_path, 'monitoring_plot_xz_down.png', fs=18, comparison=True)

#%% PLOT AND COMPARE STRAIN DATA

# # DSS dictionaries data path
# PATH = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/DSS_DTS_py_project/Valter_phase1_interval13/'
# PATH_ = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/3_Tough3Flac3Dv7_coupled_inj/041223/case_1/'
# PATH_OUTPUT = '/home/victor/Desktop/PhD@SED/08_TOUGHFLAC_for_FEAR/Figures/'
#
# # read corrected DSS data
# with open(PATH + 'Int13_DSS_DTS_combined_Antonio_full.pkl', 'rb') as f:
#     S_dss = pickle.load(f)
#
# # read rotated strain data from TOUGH-FLAC simulation
# with open(PATH_ + 'Int13_DSS_TOUGH-FLAC_X_041223.pkl', 'rb') as g:
#     S_tf = pickle.load(g)
#
# SENSOR = 'MB8'
#
# # read also unfiltered raw hydraulics
# hyd_raw_st1 = pd.read_hdf(PATH + 'Hydraulics_unfiltered_int13.h5', key='df', mode='r')
# if SENSOR=='MB1':
#     hyd_raw_mb = hyd_raw_st1    # no pressure data for MB1
# elif SENSOR=='MB2':             # no strain data for MB2
#     hyd_raw_mb = pd.read_hdf(PATH + 'Hydraulics_MB2_int13.h5', key='df', mode='r')
# elif SENSOR=='MB4':
#     hyd_raw_mb = hyd_raw_st1    # no pressure data for MB4
# elif SENSOR=='MB5':
#     # hyd_raw_mb = pd.read_hdf(PATH + 'Hydraulics_MB5_int13.h5', key='df', mode='r')
#     hyd_raw_mb = hyd_raw_st1
# elif SENSOR=='MB8':
#     hyd_raw_mb = pd.read_hdf(PATH + 'Hydraulics_MB8_int13.h5', key='df', mode='r')
# else:
#     TypeError('Not a valid sensor')
#
# # read also pressure from TOUGH only and coupled TOUGH-FLAC
# absolute_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/'
# # path_1 = absolute_path + '2_Tough3_inj_only/05.09/'
# path_1 = '/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/simulations/171123/04/'
# # path_2 = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TF_testrun/00_base_case/3_coupling/'
# path_2 = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/3_Tough3Flac3Dv7_coupled_inj/041223/case_1/'
# # FOFT output files for (in this order!): ST1, MB2, MB5, MB8
# # hyd_results_all = utils.sort_tough_foft_output([path_1, path_2], ['FOFT_A9P73.csv', 'FOFT_A9Q68.csv', 'FOFT_A9O69.csv', 'FOFT_A9P65.csv'])
# # hyd_results_all = utils.sort_tough_foft_output([path_1, path_2], ['FOFT_ACO76.csv', 'FOFT_ACR61.csv', 'FOFT_ACL64.csv', 'FOFT_ACO52.csv'])
# hyd_results_all = utils.sort_tough_foft_output([path_1, path_2], ['FOFT_AI342.csv', 'FOFT_AHT90.csv', 'FOFT_AIB37.csv', 'FOFT_AI280.csv'])
#
# # starting time of the simulations
# # simulation_start = datetime.datetime(2022, 3, 23, 0, 00, 00, 00)
# # simulation_end = datetime.datetime(2022, 3, 24, 23, 59, 59, 59)
# simulation_start = datetime.datetime(2022, 3, 23, 10, 00, 00, 00)
# simulation_end = datetime.datetime(2022, 3, 24, 6, 59, 59, 59)
#
# # # MB8 plot
# # figure1 = utils.plot_strain_comparison(hyd_raw_st1, hyd_raw_mb, hyd_results_all[0][3], hyd_results_all[1][3], S_dss[SENSOR], S_tf[SENSOR], simulation_start,
# #                                        simulation_end, fs=22, bottomdepth=150, topdepth=90, strain_min=-100, strain_max=100,
# #                                        sensor=SENSOR, cbar_range='val_range', plot_strain_timeseries=True)
# # # figure1.savefig(PATH_OUTPUT + 'DSS_'+str(SENSOR)+'_waterfall_comparison_new.png')
# # figure1.show()
# # # # Plot DSS data only (without modelling data)
# # figure2 = utils.plot_dss_strain(S_dss,
# #                                 fs=20,
# #                                 bottomdepth=150,
# #                                 topdepth=90,
# #                                 strain_min=-100,
# #                                 strain_max=100,
# #                                 time_start=0,
# #                                 time_end=10,
# #                                 sensor=SENSOR,
# #                                 cbar_range='val_range')
# # figure2.show()

#%% PLOT AND FIT PARAMETERS MB8

# # calculate depth of FUPPE and FBOTT interfaces
#
# if SENSOR=='MB1':
#     FUPPE = (104.7+101.9)/2             # FUPPE depth
#     FBOTT = (120+117.1)/2               # FBOTT depth
#     ts1 = 105.8                         # timeseries 1 (on DSS data)
#     ts2 = 106.0                         # timeseries 2 (on tf results)
#     bd = 150                            # bottomdepth
#     td = 90                             # topdepth
# elif SENSOR=='MB4':
#     FUPPE = (99.8+96.9)/2
#     FBOTT = (115.3+112.5)/2
#     ts1 = 107.8
#     ts2 = 107.8
#     bd = 140
#     td = 80
# elif SENSOR=='MB5':
#     FUPPE = (99.4+96.6)/2
#     FBOTT = (114.8+111.9)/2
#     ts1 = 110.5
#     ts2 = 111.1
#     bd = 140
#     td = 80
# else: # MB8
#     FUPPE = (103.9+106.9)/2
#     FBOTT = (120 + 123.1)/2
#     # ts1 = 118.7
#     ts1 = 110.2
#     # ts2 = 119.25
#     ts2 = 108.7
#     bd = 150
#     td = 90
#
# figure3 = utils.plot_MB8_results(hyd_raw_st1,
#                                  hyd_raw_mb,
#                                  hyd_results_all[0][3],
#                                  hyd_results_all[1][3],
#                                  S_dss[SENSOR],
#                                  S_tf[SENSOR],
#                                  simulation_start,
#                                  simulation_end,
#                                  [FUPPE, FBOTT],
#                                  [ts1, ts2],
#                                  fs=40,
#                                  bottomdepth=bd,
#                                  topdepth=td,
#                                  strain_min=-100,
#                                  strain_max=100,
#                                  sensor=SENSOR,
#                                  cbar_range='val_range',
#                                  plot_strain_timeseries=True)
#
# figure3.show()
# figure3.savefig(path_2 + str(SENSOR) + '_strain_and_pressure_plot.png')