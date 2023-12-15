"""
utils functions for TOUGH-FLAC pre- and post-processing routines
@ by Victor Clasen Repollés (victor.clasen@sed.ethz.ch)
"""

# Import required libraries
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datetime
import matplotlib as mpl

def rotation_matrix(angle, axis=2):
    """
    Return rotation matrix for input angle and axis

    Parameters
    ----------
    angle : scalar
        Rotation angle.
    axis : int (0, 1 or 2), optional, default z
        Rotation axis:
        - 0: X axis,
        - 1: Y axis,
        - 2: Z axis.

    Returns
    -------
    array_like
        Rotation matrix.

    """
    if axis not in {0, 1, 2}:
        raise ValueError()

    theta = np.deg2rad(angle)
    ct, st = np.cos(theta), np.sin(theta)
    R = {
        0: np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct],]),
        1: np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct],]),
        2: np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0],]),
    }
    return R[axis]

def rotate_tensor(tensor, angle, axis=2):
    """
    Rotate a tensor along an axis

    Parameters
    ----------
    tensor : array_like
        Tensor to rotate.
    angle : scalar
        Rotation angle.
    axis : int (0, 1 or 2), optional, default 2
        Rotation axis:
        - 0: X axis,
        - 1: Y axis,
        - 2: Z axis.

    Returns
    -------
    array_like
        Rotated tensor.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError()
    if tensor.shape != (3, 3):
        raise ValueError()

    R = rotation_matrix(angle, axis)
    return np.dot(np.dot(R, tensor), R.T)

def create_dict(sensors):
    d = {}
    for i in range(len(sensors)):
        d[sensors[i]] = {
            "distance": {},
            "time": {},
            "strain": {},
        }
    return d

def prepare_temp_data(old_dict, datetime_zero, t_max, sensor_depth, apply_filter=False):
    new_dict = {
        'distance': (old_dict['distance'] - old_dict['distance'][0])[:, 0],
        'datetime': (pd.to_datetime(old_dict['datenum_value'][0]-719529, unit='d').round('s')).to_numpy(),
        'temp_data': old_dict['tempC'],
    }
    time_list = []
    for i in range(len(new_dict['datetime'])):
        time_list.append(((new_dict['datetime'][i] + np.timedelta64(1, 'h')) - datetime_zero).total_seconds())
    new_dict['time'] = np.asarray(time_list)

    # find index of sensor depth and final simulation time
    index_dist = min(range(len(new_dict['distance'])), key=lambda i: abs(new_dict['distance'][i] - sensor_depth))
    index_time = min(range(len(new_dict['time'])), key=lambda i:abs(new_dict['time'][i] - t_max))

    new_df = pd.DataFrame()

    new_df['time'] = new_dict['time'][:(index_time+1)]

    # Option of temporal filter
    if apply_filter==True:
        new_df['temperature'] = sc.signal.medfilt(new_dict['temp_data'], [81, 3])[index_dist, :(index_time+1)]
    else:
        new_df['temperature'] = new_dict['temp_data'][index_dist, :(index_time+1)]

    return new_dict, new_df

def sort_tough_foft_output(PATH, FILES):
    # TODO: generalize this function for all possibilities (loop over foft files, import names, etc.)
    hyd_results_all = []
    for i in range(len(PATH)):
        hyd_results = []
        for j in range(len(FILES)):
            hyd_result = pd.read_csv(PATH[i] + FILES[j])
            # replace names in the csv file to have the same format and not complicate things
            hyd_result.rename(columns={hyd_result.columns.values[0]: 'time', hyd_result.columns.values[1]: 'pressure',
                                       hyd_result.columns.values[2]: 'temperature'}, inplace=True)
            hyd_results.append(hyd_result)
        hyd_results_all.append(hyd_results)

    return hyd_results_all

def inj_cells_distributing_1(hydraulics, mesh, cell_ids):
    cell_ids_all = list(mesh['elements'].keys())
    ind_all = []
    vols_all = []
    for i in range(len(cell_ids)):
        ind_all.append([index for index, value in enumerate(cell_ids_all) if value == cell_ids[i]])
        vols_all.append(mesh['elements'][cell_ids[i]]['volume'])
    vols_total = np.sum(np.asarray(vols_all))
    hydraulics_new = np.zeros((len(cell_ids), len(hydraulics.flowrate)))
    for i in range(len(cell_ids)):
        for j in range(len(hydraulics.flowrate)):
            hydraulics_new[i, j] = hydraulics.flowrate[j] * (vols_all[i] / vols_total)
    return hydraulics_new

def inj_cells_distributing_2(hydraulics, mesh, cell_ids, weights):
    hydraulics_total = []
    if len(cell_ids)==len(weights):
        for i in range(len(weights)):
            hydraulics_total.append(inj_cells_distributing_1(hydraulics, mesh, cell_ids[i]) * weights[i])
    else:
         raise ValueError('length of cell id layers is not equal to length of defined weights')
    return hydraulics_total

########################################################################################################################
# PLOTTING #
########################################################################################################################

def plot_hydraulics(hydraulics_raw, hydraulics_result, INT_PATH, fs=20):
    """
    Plots flow rate and pressure, (and temperature) from the observations in a stimulation interval,
    as well as the results from an uncoupled TOUGH3 simulation
    -> created as a routine to test the development of the TOUGH-FLAC model for VALTER phase 1 interval 13 stimulation
    """

    fig = plt.figure(figsize=[24, 15])
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax1 = fig.add_subplot(gs[0])
    ax11 = ax1.twinx()

    ax1.plot(hydraulics_raw.time / 3600.0, hydraulics_raw.flowrate * 60.0, color='tab:blue', ls='-', lw=2, label='injected flow rate')
    ax11.plot(hydraulics_raw.time / 3600.0, hydraulics_raw.pressure / (1.e6), color='tab:green', ls='-', lw=2, label='measured pressure')
    ax11.plot(hydraulics_result.time / 3600.0, hydraulics_result.pressure / (1.e6), color='tab:olive', ls='-', lw=2, label='TOUGH3 pressure')
    ax1.set_xlabel('Time [hours] from injection', fontsize=fs+2)
    ax1.xaxis.set_tick_params(labelsize=fs+2)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_11, labels_11 = ax11.get_legend_handles_labels()
    lns = lines_1 + lines_11
    lbls = labels_1 + labels_11

    ax1.legend(lns, lbls, loc='best', fontsize=fs)
    ax1.set_ylabel('Flow rate [L/min]', fontsize=fs+2)
    ax1.yaxis.set_tick_params(labelsize=fs+2)
    ax11.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    ax11.yaxis.set_tick_params(labelsize=fs+2)

    fig.savefig(INT_PATH + 'hydraulics_plot.svg')
    fig.show()

    return

def pres_plot_all(hyd_st1_raw, hyd_st1_raw_unfiltered, hyd_mb2_raw, hyd_mb5_raw, hyd_mb8_raw, hyd_result_all, PATH,
                  fs=20, coupled_simulation=False, double_coupled_simulation=False):

    fig = plt.figure(figsize=[24, 20])
    gs = gridspec.GridSpec(nrows=4, ncols=1, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    # ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[3])

    # ST1
    ax1.plot(hyd_st1_raw_unfiltered.time / 3600.0, hyd_st1_raw_unfiltered.flowrate * 60.0, color='tab:cyan', ls='-',
             lw=3, alpha=0.5, label='injected flow rate (unfiltered)')
    ax1.plot(hyd_st1_raw.time / 3600.0, hyd_st1_raw.flowrate * 60.0, color='tab:blue', ls='-', lw=2, label='TOUGH injection')

    ax1.set_ylabel('Flow rate [L/min]', fontsize=fs+2)
    ax1.yaxis.set_tick_params(labelsize=fs+2)
    ax1.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax1.legend(loc='lower right', fontsize=fs)
    ax1.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax1.get_ylim())
    x_coord = - (0 - np.min(ax1.get_xlim()))/1.15 + 1.35

    ax1.text(x_coord, y_coord, 'a) Injection point at ST1', fontsize=fs+2, va='top')

    ax2.plot(hyd_st1_raw_unfiltered.time / 3600.0, hyd_st1_raw_unfiltered.pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure')
    ax2.plot(hyd_result_all[0][0].time / 3600.0, hyd_result_all[0][0].pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')

    if coupled_simulation==True:
        ax2.plot(hyd_result_all[1][0].time / 3600.0, hyd_result_all[1][0].pressure / (1.e6), color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC pressure')
    elif double_coupled_simulation==True:
        ax2.plot(hyd_result_all[1][0].time / 3600.0, hyd_result_all[1][0].pressure / (1.e6), color='tab:orange', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (1)')
        ax2.plot(hyd_result_all[2][0].time / 3600.0, hyd_result_all[2][0].pressure / (1.e6), color='tab:purple', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (2)')
    elif coupled_simulation==True & double_coupled_simulation==True:
        raise TypeError('Both booleans cannot be true!')

    ax2.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    ax2.yaxis.set_tick_params(labelsize=fs+2)
    ax2.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax2.legend(loc='lower right', fontsize=fs)
    ax2.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax2.get_ylim())
    x_coord = - (0 - np.min(ax2.get_xlim()))/1.15 + 1.35

    ax2.text(x_coord, y_coord, 'b) Injection point at ST1', fontsize=fs+2, va='top')

    # MB2
    ax3.plot(hyd_mb2_raw.time / 3600.0, hyd_mb2_raw.pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure')
    ax3.plot(hyd_result_all[0][1].time / 3600.0, hyd_result_all[0][1].pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')

    if coupled_simulation==True:
        ax3.plot(hyd_result_all[1][1].time / 3600.0, hyd_result_all[1][1].pressure / (1.e6), color='tab:orange', label='TOUGH-FLAC pressure')
    elif double_coupled_simulation == True:
        ax3.plot(hyd_result_all[1][1].time / 3600.0, hyd_result_all[1][1].pressure / (1.e6), color='tab:orange', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (1)')
        ax3.plot(hyd_result_all[2][1].time / 3600.0, hyd_result_all[2][1].pressure / (1.e6), color='tab:purple', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (2)')
    elif coupled_simulation == True & double_coupled_simulation == True:
        raise TypeError('Both booleans cannot be true!')

    ax3.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    ax3.yaxis.set_tick_params(labelsize=fs+2)
    ax3.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax3.legend(loc='lower right', fontsize=fs)
    ax3.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax3.get_ylim())
    x_coord = - (0 - np.min(ax3.get_xlim()))/1.15 + 1.35

    ax3.text(x_coord, y_coord, 'c) Monitoring point at MB2', fontsize=fs+2, va='top')

    """Commented because the sensor at MB5 is actually broken"""
    # # MB5
    # ax4.plot(hyd_mb5_raw.time / 3600.0, hyd_mb5_raw.pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure')
    # ax4.plot(hyd_result_all[0][2].time / 3600.0, hyd_result_all[0][2].pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')
    #
    # if coupled_simulation==True:
    #     ax4.plot(hyd_result_all[1][2].time / 3600.0, hyd_result_all[1][2].pressure / (1.e6), color='tab:orange', label='TOUGH-FLAC pressure')
    #
    # ax4.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    # ax4.yaxis.set_tick_params(labelsize=fs+2)
    # ax4.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    # ax4.legend(loc='lower right', fontsize=fs)
    # ax4.set_xlim((10.0, 20.0))
    #
    # # coordiantes
    # y_coord = 0.95 * np.max(ax4.get_ylim())
    # x_coord = - (0 - np.min(ax4.get_xlim()))/1.15 + 1.35
    #
    # ax4.text(x_coord, y_coord, 'd) Monitoring point at MB5', fontsize=fs+2, va='top')

    # MB8
    ax5.plot(hyd_mb8_raw.time / 3600.0, hyd_mb8_raw.pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure')
    ax5.plot(hyd_result_all[0][3].time / 3600.0, hyd_result_all[0][3].pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')

    if coupled_simulation==True:
        ax5.plot(hyd_result_all[1][3].time / 3600.0, hyd_result_all[1][3].pressure / (1.e6), color='tab:orange', label='TOUGH-FLAC pressure')
    elif double_coupled_simulation==True:
        ax5.plot(hyd_result_all[1][3].time / 3600.0, hyd_result_all[1][3].pressure / (1.e6), color='tab:orange', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (1)')
        ax5.plot(hyd_result_all[2][3].time / 3600.0, hyd_result_all[2][3].pressure / (1.e6), color='tab:purple', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (2)')
    elif coupled_simulation==True & double_coupled_simulation==True:
        raise TypeError('Both booleans cannot be true!')

    ax5.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    ax5.yaxis.set_tick_params(labelsize=fs+2)
    ax5.set_xlabel('Time [hours] from start of simulation', fontsize=fs+2)
    ax5.xaxis.set_tick_params(labelsize=fs+2)
    ax5.legend(loc='lower right', fontsize=fs)
    ax5.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax5.get_ylim())
    x_coord = - (0 - np.min(ax5.get_xlim()))/1.15 + 1.35

    ax5.text(x_coord, y_coord, 'd) Monitoring point at MB8', fontsize=fs+2, va='top')

    fig.suptitle('Pressure timeseries for injection and monitoring points', fontsize=fs+4)

    gs.tight_layout(fig, rect=[0.0, 0.0, 1.0, 1.0])

    fig.savefig(PATH + 'pressure_plot_all.png')
    fig.show()

    return

def temp_plot_all(hyd_st1_raw, hyd_st1_raw_unfiltered, temp_st1_raw, temp_mb5_raw, temp_mb8_raw, hyd_result_all, PATH,
                  fs=20, coupled_simulation=False, double_coupled_simulation=False, separate_y_axis=False):

    fig = plt.figure(figsize=[24, 20])
    gs = gridspec.GridSpec(nrows=4, ncols=1, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # ST1
    ax1.plot(hyd_st1_raw_unfiltered.time / 3600.0, hyd_st1_raw_unfiltered.flowrate * 60.0, color='tab:cyan', ls='-', lw=3, alpha=0.5, label='injected flowrate (unfiltered)')
    ax1.plot(hyd_st1_raw.time / 3600.0, hyd_st1_raw.flowrate * 60.0, color='tab:blue', ls='-', lw=2, label='TOUGH injection')

    ax1.set_ylabel('Flow rate [L/min]', fontsize=fs+2)
    ax1.yaxis.set_tick_params(labelsize=fs+2)
    ax1.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax1.legend(loc='lower right', fontsize=fs)
    ax1.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax1.get_ylim())
    x_coord = - (0 - np.min(ax1.get_xlim()))/1.15 + 1.35

    ax1.text(x_coord, y_coord, 'a) Injection point at ST1', fontsize=fs+2, va='top')

    ax2.plot(temp_st1_raw.time / 3600.0, temp_st1_raw.temperature, color='k', ls='-', lw=2, label='measured temperature (DTS)')
    ax2.plot(hyd_result_all[0][0].time / 3600.0, hyd_result_all[0][0].temperature, color='tab:red', ls='-', lw=2, label='TOUGH temperature')

    if coupled_simulation==True:
        ax2.plot(hyd_result_all[1][0].time / 3600.0, hyd_result_all[1][0].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature')
    elif double_coupled_simulation==True:
        ax2.plot(hyd_result_all[1][0].time / 3600.0, hyd_result_all[1][0].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature (1)')
        ax2.plot(hyd_result_all[2][0].time / 3600.0, hyd_result_all[2][0].temperature, color='tab:purple', ls='-', lw=2, label='TOUGH-FLAC temperature (2)')

    ax2.set_ylabel('Temperature [°C]', fontsize=fs+2)
    ax2.yaxis.set_tick_params(labelsize=fs+2)
    ax2.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax2.legend(loc='lower right', fontsize=fs)
    ax2.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax2.get_ylim()) + 0.93
    x_coord = - (0 - np.min(ax2.get_xlim()))/1.15 + 1.35

    ax2.text(x_coord, y_coord, 'b) Injection point at ST1', fontsize=fs+2, va='top')

    # MB5
    ax3.plot(temp_mb5_raw.time / 3600.0, temp_mb5_raw.temperature, color='k', ls='-', lw=2, label='measured temperature (DTS)')
    # ax3.plot(hyd_result_all[0][2].time / 3600.0, hyd_result_all[0][2].temperature, color='tab:red', ls='-', lw=2, label='TOUGH temperature')

    # if coupled_simulation==True:
    #     ax3.plot(hyd_result_all[1][2].time / 3600.0, hyd_result_all[1][2].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature')

    ax3.set_ylabel('T [°C] data', fontsize=fs+2)
    ax3.yaxis.set_tick_params(labelsize=fs+2)
    ax3.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    # ax3.legend(loc='lower right', fontsize=fs)
    ax3.set_xlim((10.0, 20.0))

    # coordinates
    y_coord = 0.95 * np.max(ax3.get_ylim()) + 0.95
    x_coord = - (0 - np.min(ax3.get_xlim()))/1.15 + 1.35

    ax3.text(x_coord, y_coord, 'c) Monitoring point at MB5', fontsize=fs+2, va='top')

    if separate_y_axis==True:
        ax33 = ax3.twinx()
        ax33.plot(hyd_result_all[0][2].time / 3600.0, hyd_result_all[0][2].temperature, color='tab:red', ls='-', lw=2, label='TOUGH temperature')
        if coupled_simulation==True:
            ax33.plot(hyd_result_all[1][2].time / 3600.0, hyd_result_all[1][2].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature (1)')
        elif double_coupled_simulation==True:
            ax33.plot(hyd_result_all[1][2].time / 3600.0, hyd_result_all[1][2].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature (1)')
            ax33.plot(hyd_result_all[2][2].time / 3600.0, hyd_result_all[2][2].temperature, color='tab:purple', ls='-', lw=2, label='TOUGH-FLAC temperature (2)')

        lines_3, labels_3 = ax3.get_legend_handles_labels()
        lines_33, labels_33 = ax33.get_legend_handles_labels()
        lns_3 = lines_3 + lines_33
        lbls_3 = labels_3 + labels_33

        ax33.set_ylabel('T [°C] models', fontsize=fs+2)
        ax33.yaxis.set_label_position('right')
        ax33.yaxis.set_ticks_position('right')
        ax33.legend(lns_3, lbls_3, loc='lower right', fontsize=fs)
        # ax33.yaxis.label.set_color('tab:pink')
        # ax33.tick_params(axis='y', colors='tab:pink')

    # MB8
    ax4.plot(temp_mb8_raw.time / 3600.0, temp_mb8_raw.temperature, color='k', ls='-', lw=2, label='measured temperature (DTS)')
    # ax4.plot(hyd_result_all[0][3].time / 3600.0, hyd_result_all[0][3].temperature, color='tab:red', ls='-', lw=2, label='TOUGH3 temperature')

    # if coupled_simulation==True:
    #     ax4.plot(hyd_result_all[1][3].time / 3600.0, hyd_result_all[1][3].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature')

    ax4.set_ylabel('T [°C] data', fontsize=fs + 2)
    ax4.yaxis.set_tick_params(labelsize=fs + 2)
    ax4.set_xlabel('Time [hours] from start of simulation', fontsize=fs+2)
    ax4.xaxis.set_tick_params(labelsize=fs + 2)
    # ax4.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    # ax4.legend(loc='lower right', fontsize=fs)
    ax4.set_xlim((10.0, 20.0))

    # coordinates
    y_coord = 0.95 * np.max(ax4.get_ylim()) + 0.98
    x_coord = - (0 - np.min(ax4.get_xlim())) / 1.15 + 1.35

    ax4.text(x_coord, y_coord, 'd) Monitoring point at MB8', fontsize=fs + 2, va='top')

    if separate_y_axis==True:
        ax44 = ax4.twinx()
        ax44.plot(hyd_result_all[0][3].time / 3600.0, hyd_result_all[0][3].temperature, color='tab:red', ls='-', lw=2, label='TOUGH temperature')
        if coupled_simulation==True:
            ax44.plot(hyd_result_all[1][3].time / 3600.0, hyd_result_all[1][3].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature (1)')
        elif double_coupled_simulation==True:
            ax44.plot(hyd_result_all[1][3].time / 3600.0, hyd_result_all[1][3].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature (1)')
            ax44.plot(hyd_result_all[2][3].time / 3600.0, hyd_result_all[2][3].temperature, color='tab:purple', ls='-', lw=2, label='TOUGH-FLAC temperature (2)')

        lines_4, labels_4 = ax4.get_legend_handles_labels()
        lines_44, labels_44 = ax44.get_legend_handles_labels()
        lns_4 = lines_4 + lines_44
        lbls_4 = labels_4 + labels_44

        ax44.set_ylabel('T [°C] models', fontsize=fs+2)
        ax44.yaxis.set_label_position('right')
        ax44.yaxis.set_ticks_position('right')
        ax44.legend(lns_4, lbls_4, loc='lower right', fontsize=fs)
        # ax44.yaxis.label.set_color('tab:pink')
        # ax44.tick_params(axis='y', colors='tab:pink')

    fig.suptitle('Temperature timeseries for injection and monitoring points', fontsize=fs+4)

    gs.tight_layout(fig, rect=[0.0, 0.0, 1.0, 1.0])

    fig.savefig(PATH + 'temp_plot_inj_point.png')
    fig.show()

    return

def plot_pres_temp_comb(hyd_st1_raw, hyd_st1_raw_unfiltered, temp_st1_raw, hyd_mb2_raw, hyd_mb5_raw, temp_mb5_raw,
                        hyd_mb8_raw, temp_mb8_raw, hyd_result_all, PATH, fs=20, coupled_simulation=False, double_coupled_simulation=False,
                        separate_y_axis=False):

    fig = plt.figure(figsize=[28, 20])
    gs = gridspec.GridSpec(nrows=4, ncols=2, wspace=0.2, hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax11 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 0])
    ax33 = fig.add_subplot(gs[2, 1])
    ax4 = fig.add_subplot(gs[3, 0])
    ax44 = fig.add_subplot(gs[3, 1])

    """
    PRESSURE
    """

    # Central injection cell

    ax1.plot(hyd_result_all[0][5].time / 3600.0, hyd_result_all[0][5].pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')

    ax1.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    ax1.yaxis.set_tick_params(labelsize=fs+2)
    ax1.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax1.legend(loc='lower right', fontsize=fs)
    ax1.set_xlim((10.0, 20.0))

    # coordinates
    y_coord = 0.95 * np.max(ax1.get_ylim())
    x_coord = - (0 - np.min(ax1.get_xlim()))/1.15 + 1.35

    ax1.text(x_coord, y_coord, 'a) Central injection cell', fontsize=fs+2, va='top')

    # ST1

    ax2.plot(hyd_st1_raw_unfiltered.time / 3600.0, hyd_st1_raw_unfiltered.pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure')
    ax2.plot(hyd_result_all[0][0].time / 3600.0, hyd_result_all[0][0].pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')

    if coupled_simulation==True:
        ax2.plot(hyd_result_all[1][0].time / 3600.0, hyd_result_all[1][0].pressure / (1.e6), color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC pressure')
    elif double_coupled_simulation==True:
        ax2.plot(hyd_result_all[1][0].time / 3600.0, hyd_result_all[1][0].pressure / (1.e6), color='tab:orange', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (1)')
        ax2.plot(hyd_result_all[2][0].time / 3600.0, hyd_result_all[2][0].pressure / (1.e6), color='tab:purple', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (2)')
    elif coupled_simulation==True & double_coupled_simulation==True:
        raise TypeError('Both booleans cannot be true!')

    ax2.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    ax2.yaxis.set_tick_params(labelsize=fs+2)
    ax2.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax2.legend(loc='lower right', fontsize=fs)
    ax2.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax2.get_ylim())
    x_coord = - (0 - np.min(ax2.get_xlim()))/1.15 + 1.35

    ax2.text(x_coord, y_coord, 'a) Injection point at ST1', fontsize=fs+2, va='top')

    # MB2
    ax3.plot(hyd_mb2_raw.time / 3600.0, hyd_mb2_raw.pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure')
    ax3.plot(hyd_result_all[0][1].time / 3600.0, hyd_result_all[0][1].pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')

    if coupled_simulation==True:
        ax3.plot(hyd_result_all[1][1].time / 3600.0, hyd_result_all[1][1].pressure / (1.e6), color='tab:orange', label='TOUGH-FLAC pressure')
    elif double_coupled_simulation == True:
        ax3.plot(hyd_result_all[1][1].time / 3600.0, hyd_result_all[1][1].pressure / (1.e6), color='tab:orange', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (1)')
        ax3.plot(hyd_result_all[2][1].time / 3600.0, hyd_result_all[2][1].pressure / (1.e6), color='tab:purple', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (2)')
    elif coupled_simulation == True & double_coupled_simulation == True:
        raise TypeError('Both booleans cannot be true!')

    ax3.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    ax3.yaxis.set_tick_params(labelsize=fs+2)
    ax3.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax3.legend(loc='lower right', fontsize=fs)
    ax3.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax3.get_ylim())
    x_coord = - (0 - np.min(ax3.get_xlim()))/1.15 + 1.35

    ax3.text(x_coord, y_coord, 'b) Monitoring point at MB2', fontsize=fs+2, va='top')

    # MB8
    ax4.plot(hyd_mb8_raw.time / 3600.0, hyd_mb8_raw.pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure')
    ax4.plot(hyd_result_all[0][3].time / 3600.0, hyd_result_all[0][3].pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')

    if coupled_simulation==True:
        ax4.plot(hyd_result_all[1][3].time / 3600.0, hyd_result_all[1][3].pressure / (1.e6), color='tab:orange', label='TOUGH-FLAC pressure')
    elif double_coupled_simulation==True:
        ax4.plot(hyd_result_all[1][3].time / 3600.0, hyd_result_all[1][3].pressure / (1.e6), color='tab:orange', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (1)')
        ax4.plot(hyd_result_all[2][3].time / 3600.0, hyd_result_all[2][3].pressure / (1.e6), color='tab:purple', ls='-',
                 lw=2, label='TOUGH-FLAC pressure (2)')
    elif coupled_simulation==True & double_coupled_simulation==True:
        raise TypeError('Both booleans cannot be true!')

    ax4.set_ylabel('Pressure [MPa]', fontsize=fs+2)
    ax4.yaxis.set_tick_params(labelsize=fs+2)
    ax4.set_xlabel('Time [hours] from start of simulation', fontsize=fs+2)
    ax4.xaxis.set_tick_params(labelsize=fs+2)
    ax4.legend(loc='lower right', fontsize=fs)
    ax4.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax4.get_ylim())
    x_coord = - (0 - np.min(ax4.get_xlim()))/1.15 + 1.35

    ax4.text(x_coord, y_coord, 'c) Monitoring point at MB8', fontsize=fs+2, va='top')

    """
    TEMPERATURE
    """

    # Central injection cell

    ax11.plot(hyd_result_all[0][5].time / 3600.0, hyd_result_all[0][5].temperature, color='tab:red', ls='-', lw=2, label='TOUGH pressure')

    ax11.set_ylabel('Temperature [°C]', fontsize=fs+2)
    ax11.yaxis.set_tick_params(labelsize=fs+2)
    ax11.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax11.legend(loc='lower right', fontsize=fs)
    ax11.set_xlim((10.0, 20.0))

    # coordinates
    y_coord = 0.95 * np.max(ax11.get_ylim())
    x_coord = - (0 - np.min(ax11.get_xlim()))/1.15 + 1.35

    ax11.text(x_coord, y_coord, 'a) Central injection cell', fontsize=fs+2, va='top')

    # ST1

    ax22.plot(temp_st1_raw.time / 3600.0, temp_st1_raw.temperature, color='k', ls='-', lw=2, label='measured temperature (DTS)')
    ax22.plot(hyd_result_all[0][0].time / 3600.0, hyd_result_all[0][0].temperature, color='tab:red', ls='-', lw=2, label='TOUGH temperature')

    if coupled_simulation==True:
        ax22.plot(hyd_result_all[1][0].time / 3600.0, hyd_result_all[1][0].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature')

    ax22.set_ylabel('Temperature [°C]', fontsize=fs+2)
    ax22.yaxis.set_tick_params(labelsize=fs+2)
    ax22.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax22.legend(loc='lower right', fontsize=fs)
    ax22.set_xlim((10.0, 20.0))

    # coordiantes
    y_coord = 0.95 * np.max(ax22.get_ylim()) + 0.93
    x_coord = - (0 - np.min(ax22.get_xlim()))/1.15 + 1.35

    ax22.text(x_coord, y_coord, 'a) Injection point at ST1', fontsize=fs+2, va='top')

    # MB5
    ax33.plot(temp_mb5_raw.time / 3600.0, temp_mb5_raw.temperature, color='k', ls='-', lw=2, label='measured temperature (DTS)')
    # ax3.plot(hyd_result_all[0][2].time / 3600.0, hyd_result_all[0][2].temperature, color='tab:red', ls='-', lw=2, label='TOUGH temperature')

    # if coupled_simulation==True:
    #     ax3.plot(hyd_result_all[1][2].time / 3600.0, hyd_result_all[1][2].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature')

    ax33.set_ylabel('T [°C] data', fontsize=fs+2)
    ax33.yaxis.set_tick_params(labelsize=fs+2)
    ax33.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    # ax3.legend(loc='lower right', fontsize=fs)
    ax33.set_xlim((10.0, 20.0))

    # coordinates
    y_coord = 0.95 * np.max(ax33.get_ylim()) + 0.95
    x_coord = - (0 - np.min(ax33.get_xlim()))/1.15 + 1.35

    ax33.text(x_coord, y_coord, 'b) Monitoring point at MB5', fontsize=fs+2, va='top')

    if separate_y_axis==True:
        ax33_ = ax33.twinx()
        ax33_.plot(hyd_result_all[0][2].time / 3600.0, hyd_result_all[0][2].temperature, color='tab:red', ls='-', lw=2, label='TOUGH temperature')
        if coupled_simulation==True:
            ax33_.plot(hyd_result_all[1][2].time / 3600.0, hyd_result_all[1][2].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature (1)')

        lines_33, labels_33 = ax33.get_legend_handles_labels()
        lines_33_, labels_33_ = ax33_.get_legend_handles_labels()
        lns_33 = lines_33 + lines_33_
        lbls_33 = labels_33 + labels_33_

        ax33_.set_ylabel('T [°C] models', fontsize=fs+2)
        ax33_.yaxis.set_label_position('right')
        ax33_.yaxis.set_ticks_position('right')
        ax33_.legend(lns_33, lbls_33, loc='lower right', fontsize=fs)
        # ax33.yaxis.label.set_color('tab:pink')
        # ax33.tick_params(axis='y', colors='tab:pink')

    # MB8
    ax44.plot(temp_mb8_raw.time / 3600.0, temp_mb8_raw.temperature, color='k', ls='-', lw=2, label='measured temperature (DTS)')
    # ax44.plot(hyd_result_all[0][3].time / 3600.0, hyd_result_all[0][3].temperature, color='tab:red', ls='-', lw=2, label='TOUGH3 temperature')

    # if coupled_simulation==True:
    #     ax44.plot(hyd_result_all[1][3].time / 3600.0, hyd_result_all[1][3].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature')

    ax44.set_ylabel('T [°C] data', fontsize=fs + 2)
    ax44.yaxis.set_tick_params(labelsize=fs + 2)
    ax44.set_xlabel('Time [hours] from start of simulation', fontsize=fs+2)
    ax44.xaxis.set_tick_params(labelsize=fs + 2)
    # ax44.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    # ax44.legend(loc='lower right', fontsize=fs)
    ax44.set_xlim((10.0, 20.0))

    # coordinates
    y_coord = 0.95 * np.max(ax44.get_ylim()) + 0.98
    x_coord = - (0 - np.min(ax44.get_xlim())) / 1.15 + 1.35

    ax44.text(x_coord, y_coord, 'c) Monitoring point at MB8', fontsize=fs + 2, va='top')

    if separate_y_axis==True:
        ax44_ = ax44.twinx()
        ax44_.plot(hyd_result_all[0][3].time / 3600.0, hyd_result_all[0][3].temperature, color='tab:red', ls='-', lw=2, label='TOUGH temperature')
        if coupled_simulation==True:
            ax44_.plot(hyd_result_all[1][3].time / 3600.0, hyd_result_all[1][3].temperature, color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC temperature (1)')

        lines_44, labels_44 = ax44.get_legend_handles_labels()
        lines_44_, labels_44_ = ax44_.get_legend_handles_labels()
        lns_44 = lines_44 + lines_44_
        lbls_44 = labels_44 + labels_44_

        ax44_.set_ylabel('T [°C] models', fontsize=fs+2)
        ax44_.yaxis.set_label_position('right')
        ax44_.yaxis.set_ticks_position('right')
        ax44_.legend(lns_44, lbls_44, loc='lower right', fontsize=fs)
        # ax44.yaxis.label.set_color('tab:pink')
        # ax44.tick_params(axis='y', colors='tab:pink')

    fig.suptitle('Pressure timeseries for injection and monitoring points', fontsize=fs+4)

    gs.tight_layout(fig, rect=[0.0, 0.0, 1.0, 1.0])

    fig.savefig(PATH + 'pres_temp_plot_comb.png')
    fig.show()

    return

def plot_cells_timeseries(hyd_res_all, PATH, variable_y='temperature', scaling_time=3600.0, fs=20):

    fig = plt.figure(figsize=[35, 25])
    gs = gridspec.GridSpec(nrows=3, ncols=5, wspace=0.3, hspace=0.0)
    # y-axis direction
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax03 = fig.add_subplot(gs[0, 3])
    ax04 = fig.add_subplot(gs[0, 4])
    axs0 = (ax00, ax01, ax02, ax03, ax04)
    # XZ up the fault
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    ax13 = fig.add_subplot(gs[1, 3])
    ax14 = fig.add_subplot(gs[1, 4])
    axs1 = (ax10, ax11, ax12, ax13, ax14)
    # XZ down the fault
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax22 = fig.add_subplot(gs[2, 2])
    ax23 = fig.add_subplot(gs[2, 3])
    ax24 = fig.add_subplot(gs[2, 4])
    axs2 = (ax20, ax21, ax22, ax23, ax24)

    axs = (axs0, axs1, axs2)

    colors = ['tab:red', 'tab:blue']
    labels1 = ['P (before)', 'P (after)']
    labels2 = ['T (before)', 'T (after)']

    # colors = ['tab:red', 'tab:blue', 'tab:green']
    # labels1 = ['P', 'P', 'P']
    # labels2 = ['T 18°C', 'T 14°C', 'T 10°C']

    for i in range(len(axs)):
        if variable_y=='pressure':
            axs[i][0].set_ylabel('Pressure [MPa]', fontsize=fs)
        else:
            axs[i][0].set_ylabel('Temperature [°C]', fontsize=fs)
        for j in range(len(axs[0])):
            axs[i][j].yaxis.set_tick_params(labelsize=fs)
            for k in range(len(hyd_res_all[0])):
                if variable_y=='pressure':
                    axs[i][j].plot(hyd_res_all[i][k][j].time / scaling_time, hyd_res_all[i][k][j].pressure / (1.e6),
                                   color=colors[k], ls='-', lw=2, label=labels1[k])
                    # axs[i][j].set_ylabel('Pressure [MPa]', fontsize=fs)
                    # path = PATH + 'pressure/'
                    path = PATH
                    filename = 'pres.png'
                else:
                    axs[i][j].plot(hyd_res_all[i][k][j].time / scaling_time, hyd_res_all[i][k][j].temperature,
                                   color=colors[k], ls='-', lw=2, label=labels2[k])
                    # axs[i][j].set_ylabel('Temperature [°C]', fontsize=fs)
                    # path = PATH + 'temperature/'
                    path = PATH
                    filename = 'temp.png'
                if i==int(len(axs)-1):
                    axs[i][j].set_xlabel('Time [hours] from injection', fontsize=fs)
                    axs[i][j].xaxis.set_tick_params(labelsize=fs)
                else:
                    axs[i][j].tick_params(axis='x', bottom=False, top=False, labelbottom=False)
                axs[i][j].legend(loc='upper right', fontsize=fs)

    fig.savefig(path + filename)
    fig.show()

    return

def plot_cells_timeseries_diff(hyd_res_all, PATH, variable_y='temperature', scaling_time=3600.0, fs=20):

    fig = plt.figure(figsize=[35, 25])
    gs = gridspec.GridSpec(nrows=3, ncols=4, wspace=0.3, hspace=0.0)
    # y-axis direction
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax03 = fig.add_subplot(gs[0, 3])
    axs0 = (ax00, ax01, ax02, ax03)
    # XZ up the fault
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    ax13 = fig.add_subplot(gs[1, 3])
    axs1 = (ax10, ax11, ax12, ax13)
    # XZ down the fault
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax22 = fig.add_subplot(gs[2, 2])
    ax23 = fig.add_subplot(gs[2, 3])
    axs2 = (ax20, ax21, ax22, ax23)

    axs = (axs0, axs1, axs2)

    colors = ['tab:red', 'tab:blue']
    labels1 = ['P (before)', 'P (after)']
    labels2 = ['T (before)', 'T (after)']

    for i in range(len(axs)):
        if variable_y == 'pressure':
            axs[i][0].set_ylabel('Pressure [MPa]', fontsize=fs)
        else:
            axs[i][0].set_ylabel('Temperature [°C]', fontsize=fs)
        for j in range(len(axs[0])):
            axs[i][j].yaxis.set_tick_params(labelsize=fs)
            for k in range(len(hyd_res_all[0])):
                if variable_y == 'pressure':
                    axs[i][j].plot(hyd_res_all[i][k][j].time / scaling_time,
                                   (hyd_res_all[i][k][j].pressure - hyd_res_all[i][k][j+1].pressure) / (1.e6),
                                   color=colors[k], ls='-', lw=2, label=labels1[k])
                    # axs[i][j].set_ylabel('Pressure [MPa]', fontsize=fs)
                    path = PATH + 'pressure/'
                    filename = 'pres_diff.png'
                else:
                    axs[i][j].plot(hyd_res_all[i][k][j].time / scaling_time,
                                   (hyd_res_all[i][k][j].temperature - hyd_res_all[i][k][j+1].temperature),
                                   color=colors[k], ls='-', lw=2, label=labels2[k])
                    # axs[i][j].set_ylabel('Temperature [°C]', fontsize=fs)
                    path = PATH + 'temperature/'
                    filename = 'temp_diff.png'
                if i == int(len(axs) - 1):
                    axs[i][j].set_xlabel('Time [hours] from injection', fontsize=fs)
                    axs[i][j].xaxis.set_tick_params(labelsize=fs)
                else:
                    axs[i][j].tick_params(axis='x', bottom=False, top=False, labelbottom=False)
                axs[i][j].legend(loc='upper right', fontsize=fs)

    fig.savefig(path + filename)
    fig.show()

    return

def plot_hydr_params(res_dict, cell_ids, time, PATH, FILENAME, scaling_time=3600.0, fs=20, comparison=False):

    fig = plt.figure(figsize=[35, 25])
    gs = gridspec.GridSpec(nrows=4, ncols=6, wspace=0.3, hspace=0.0)
    # first row
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax03 = fig.add_subplot(gs[0, 3])
    ax04 = fig.add_subplot(gs[0, 4])
    ax05 = fig.add_subplot(gs[0, 5])
    axs0 = (ax00, ax01, ax02, ax03, ax04, ax05)
    # second row
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    ax13 = fig.add_subplot(gs[1, 3])
    ax14 = fig.add_subplot(gs[1, 4])
    ax15 = fig.add_subplot(gs[1, 5])
    axs1 = (ax10, ax11, ax12, ax13, ax14, ax15)
    # third row
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax22 = fig.add_subplot(gs[2, 2])
    ax23 = fig.add_subplot(gs[2, 3])
    ax24 = fig.add_subplot(gs[2, 4])
    ax25 = fig.add_subplot(gs[2, 5])
    axs2 = (ax20, ax21, ax22, ax23, ax24, ax25)
    # fourth row
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1])
    ax32 = fig.add_subplot(gs[3, 2])
    ax33 = fig.add_subplot(gs[3, 3])
    ax34 = fig.add_subplot(gs[3, 4])
    ax35 = fig.add_subplot(gs[3, 5])
    axs3 = (ax30, ax31, ax32, ax33, ax34, ax35)

    axs = (axs0, axs1, axs2, axs3)

    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:purple']
    label = ['temperature', 'pressure', 'porosity', 'permeability']
    axis_name = ['Temperature [°C]', 'Pressure [MPa]', 'Porosity [-]', 'Permeability [m²]']

    for i in range(len(axs)):
        for j in range(len(axs[0])):
            if i!=int(len(axs)-1):
                axs[i][j].tick_params(axis='x', bottom=False, top=False, labelbottom=False)
            else:
                axs[i][j].xaxis.set_tick_params(labelsize=fs)
                axs[i][j].set_xlabel('Time [hours] from injection', fontsize=fs)
            axs[i][j].plot(time / scaling_time, res_dict[0][cell_ids[j]][label[i]], color=colors[i], ls='-', lw=2, label='case_1')
            if comparison==True:
                axs[i][j].plot(time / scaling_time, res_dict[1][cell_ids[j]][label[i]], color='k', ls='--', lw=1, label='case_2')
            axs[i][j].legend(loc='upper right', fontsize=fs)
            axs[i][j].yaxis.set_tick_params(labelsize=fs)
            if j==0:
                axs[i][j].set_ylabel(axis_name[i], fontsize=fs)
            if i!=int(len(axs)-1):
                axs[i][j].ticklabel_format(axis='y', style='plain', useOffset=False)
                mpl.rcParams['axes.labelsize'] = plt.gca().yaxis.get_ticklabels()[0].get_fontsize()
            else:
                axs[i][j].set_yscale('log')
                axs[i][j].tick_params(axis='y', labelsize=fs)

    fig.savefig(PATH + FILENAME)
    fig.show()

    return

def plot_strain(strain_res,
                fs=14,
                bottomdepth=150,
                topdepth=50,
                strain_min=-100,
                strain_max=100,
                time_start=0,
                time_end=10,
                sensor='MB8',
                cbar_range='max_min'):

    """
    Parameters
    ----------
    strain_res:         strain modelling results along with time and distance/depth data
    fs:                 default font size
    bottomdepth:        bottom depth cut limit
    topdepth:           top depth cut limit
    strain_min:         min strain limit for colorbar (if cbar_range!=max_min)
    strain_max:         max strain limit for colorbar (if cbar_range!=max_min)
    time_start:         lower time cut limit
    time_end:           higher time cut limit
    sensor:             monitoring borehole sensor to plot
    cbar_range:         colorbar range mode

    Returns
    -------
    fig (shows strain results from modelling)
    """

    time_corr = strain_res[sensor]['time'][0] - strain_res[sensor]['time'][0, 0]
    d_idx_1 = int(np.argmin(np.abs(strain_res[sensor]['distance'] - topdepth)))
    d_idx_2 = int(np.argmin(np.abs(strain_res[sensor]['distance'] - bottomdepth)))

    DT, DX = np.meshgrid(time_corr/3600.0, strain_res[sensor]['distance'])
    # DT, DX = np.meshgrid(time_corr/3600.0, strain_res[sensor]['distance'][d_idx_1:d_idx_2])

    fig = plt.figure(figsize=(8, 6))
    cmap1 = plt.cm.get_cmap('RdBu_r')   # for strain
    plot_dss = np.asarray(strain_res[sensor]['strain'])[:, d_idx_1:d_idx_2]

    if cbar_range=='max_min':
        max_val = max(strain_res[sensor]['strain'].max()) * 1e6  # use the max function directly on the array
        min_val = min(strain_res[sensor]['strain'].min()) * 1e6  # use the min function directly on the array
        levels1 = np.linspace(min_val, max_val, 100)
        strain_map = plt.contourf(DT, DX, plot_dss.T*1e6, levels=levels1, cmap=cmap1, vmin=min_val, vmax=max_val, extend='both')
    else:
        max_val = strain_max
        min_val = strain_min
        levels1 = np.linspace(min_val, max_val, 100)
        strain_map = plt.contourf(DT, DX, plot_dss.T*1e6, levels=levels1, cmap=cmap1, vmin=min_val, vmax=max_val, extend='both')

    #min_extr = np.round(np.min((np.abs(min_val), max_val)), decimals=0)
    strain_cbar = plt.colorbar(strain_map, ticks=np.linspace(min_val, max_val, 9), shrink=0.8) # create and add a colorbar to the plot
    plt.gca().invert_yaxis() # flip y-axis
    plt.ylabel('Borehole length [m]', fontsize=fs)
    plt.xlabel('Time since start of simulation [hours]', fontsize=fs)
    strain_cbar.set_label(r'Strain [$\mu\epsilon$]', fontsize=fs)
    # Increase font size for tick labels
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    return fig

def plot_dss_strain(strain_dss,
                    fs=14,
                    bottomdepth=150,
                    topdepth=50,
                    strain_min=-100,
                    strain_max=100,
                    time_start=0,
                    time_end=10,
                    sensor='MB8',
                    cbar_range='max_min'):

    """
    Parameters
    ----------
    strain_dss:         processed DSS strain data
    fs:                 default font size
    bottomdepth:        bottom depth cut limit
    topdepth:           top depth cut limit
    strain_min:         min strain limit for colorbar (if cbar_range!=max_min)
    strain_max:         max strain limit for colorbar (if cbar_range!=max_min)
    time_start:         lower time cut limit
    time_end:           higher time cut limit
    sensor:             monitoring borehole sensor to plot
    cbar_range:         colorbar range mode

    Returns
    -------
    fig (shows raw dss strain data)
    """

    # time corrections
    strain_dss[sensor]['time'] = np.asarray([dt - datetime.timedelta(hours=1) for dt in strain_dss[sensor]['time']])
    strain_dss[sensor]['time_sec'] = np.asarray([(dt - strain_dss[sensor]['time'][0]).total_seconds() for dt in strain_dss[sensor]['time']])
    # cut the time at 14 hours
    t_idx_2 = int(np.argmin(np.abs(strain_dss[sensor]['time_sec'] - 14.0 * 3600.0)))

    # depth corrections
    d_idx_1 = int(np.argmin(np.abs(strain_dss[sensor]['distance'] - topdepth)))
    d_idx_2 = int(np.argmin(np.abs(strain_dss[sensor]['distance'] - bottomdepth)))

    DT, DX = np.meshgrid(strain_dss[sensor]['time_sec'][:t_idx_2] / 3600.0, strain_dss[sensor]['distance'][d_idx_1:d_idx_2])

    # strain corrections
    plot_dss = np.asarray(strain_dss[sensor]['strain_filtered'].T - (strain_dss[sensor]['strain_filtered'].T)[:, 0][:, np.newaxis])[d_idx_1:d_idx_2, :t_idx_2]

    # plotting
    fig = plt.figure(figsize=(8, 6))
    cmap1 = plt.cm.get_cmap('RdBu_r')  # for strain

    if cbar_range == 'max_min':
        # max_val = strain_dss[sensor]['strain_filtered'].max()  # use the max function directly on the array
        # min_val = strain_dss[sensor]['strain_filtered'].min()  # use the min function directly on the array
        max_val = plot_dss.max()
        min_val = plot_dss.min()
        min_extr = np.round(np.min((np.abs(min_val), max_val)), decimals=0)
        levels1 = np.linspace(-min_extr, min_extr, 100)
        strain_map = plt.contourf(DT, DX, plot_dss, levels=levels1, cmap=cmap1, vmin=-min_extr, vmax=min_extr, extend='both')
    else:
        max_val = strain_max
        min_val = strain_min
        levels1 = np.linspace(min_val, max_val, 100)
        strain_map = plt.contourf(DT, DX, plot_dss, levels=levels1, cmap=cmap1, vmin=min_val, vmax=max_val, extend='both')

    # min_extr = np.round(np.min((np.abs(min_val), max_val)), decimals=0)
    strain_cbar = plt.colorbar(strain_map, ticks=np.linspace(min_val, max_val, 9), shrink=0.8)  # create and add a colorbar to the plot
    plt.gca().invert_yaxis()  # flip y-axis
    plt.ylabel('Borehole length [m]', fontsize=fs)
    plt.xlabel('Time since start of simulation [hours]', fontsize=fs)
    strain_cbar.set_label(r'Strain [$\mu\epsilon$]', fontsize=fs)
    # Increase font size for tick labels
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    return fig

def plot_strain_comparison(hyd_st1, hyd_mb, hyd_res_1, hyd_res_2, strain_dss, strain_tf, sim_start, sim_end, fs=20, bottomdepth=150,
                           topdepth=50, strain_min=-100, strain_max=100, sensor='MB8', cbar_range='max_min', plot_strain_timeseries=False):

    fig = plt.figure(figsize=[48, 36])
    gs = gridspec.GridSpec(nrows=3, ncols=2, hspace=0.05, wspace=0, width_ratios=[10, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_ = fig.add_subplot(gs[0, 1])
    ax1_.axis('off')
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_ = fig.add_subplot(gs[1, 1])
    ax2_.axis('off')
    ax3 = fig.add_subplot(gs[2, 0])
    ax3_ = fig.add_subplot(gs[2, 1])
    ax3_.axis('off')

    # check the start and end datetimes
    strain_dss['time'] = np.asarray([dt - datetime.timedelta(hours=1) for dt in strain_dss['time']])
    sorted_times = sorted([strain_dss['time'][0], strain_dss['time'][-1], sim_start, sim_end])
    starttime, endtime = sorted_times[1], sorted_times[2]
    # find the indices of datetime objects between the starttime and endtime range (DSS data)
    time_indices = [index for index, dt in enumerate(strain_dss['time']) if starttime <= dt <= endtime]
    t_ind_1, t_ind_2 = time_indices[0], time_indices[-1]
    # find the indices of datetime objects between the starttime and endtime range (hydraulic data)
    datetime_objects = [datetime.datetime.utcfromtimestamp(dt.astype(datetime.datetime)//10**9) for dt in np.asarray(hyd_st1.index)]
    time_indices_ = [index for index, dt in enumerate(datetime_objects) if starttime <= dt <= endtime]
    t_ind_1_, t_ind_2_ = time_indices_[0], time_indices_[-1]

    if (strain_tf['time'][0, 0] - hyd_st1.time[t_ind_1_]) > 0:
        t_ind_1_ = int(0)
    else:
        t_ind_1_ = int(np.argmin(np.abs(strain_tf['time'][0] - hyd_st1.time[t_ind_1_])))
    if (strain_tf['time'][0, -1] - hyd_st1.time[t_ind_2_]) < 0:
        t_ind_2_ = t_ind_2_
    else:
        t_ind_2_ = int(np.argmin(np.abs(strain_tf['time'][0] - hyd_st1.time[t_ind_2_])))

    # or get the time-shift for the x-axis in seconds, for correction
    t_shift = (starttime - datetime_objects[0]).total_seconds()
    # correct time in hydraulics and in strain from TOUGH-FLAC
    strain_tf['time_shifted'] = strain_tf['time'] - t_shift
    hyd_st1['time_shifted'] = hyd_st1['time'] - t_shift
    hyd_mb['time_shifted'] = hyd_mb['time'] - t_shift
    # correct also the hydraulics results
    hyd_res_1['time_shifted'] = hyd_res_1.time - t_shift
    hyd_res_2['time_shifted'] = hyd_res_2.time - t_shift

    # DSS compute meshgrids for strain and filtered strain difference since t_init
    strain_dss['time_sec'] = np.asarray([(dt - starttime).total_seconds() for dt in strain_dss['time']])
    DT, DX = np.meshgrid(strain_dss['time_sec'][t_ind_1:t_ind_2] / 3600.0, strain_dss['distance'])
    strain_diff = strain_dss['strain_filtered'][t_ind_1:t_ind_2,:].T - (strain_dss['strain_filtered'][t_ind_1:t_ind_2,:].T)[:, 0][:, np.newaxis]
    # TOUGH-FLAC
    DT_, DX_ = np.meshgrid(strain_tf['time_shifted'][0][t_ind_1_:t_ind_2_] / 3600.0, strain_tf['distance'])
    strain_diff_ = (np.asarray(strain_tf['strain'])[t_ind_1_:t_ind_2_,:].T - (np.asarray(strain_tf['strain'])[t_ind_1_:t_ind_2_,:].T)[:, 0][:, np.newaxis]) * (1.e6) # from strain to microstrain

    x_axis_max = strain_tf['time_shifted'][0][t_ind_1_:t_ind_2_][-1]/3600.0

    if plot_strain_timeseries==False:
        # Plot hydraulics
        ax1.plot(hyd_st1.time[t_ind_1_:t_ind_2_] / 3600.0, hyd_st1.flowrate[t_ind_1_:t_ind_2_] * 60.0, color='tab:blue', ls='-', lw=2, label='inj. flow rate (ST1) unfiltered')
        ax11 = ax1.twinx()
        # ax11.plot(hyd_st1[t_ind_1_:t_ind_2_].time / 3600.0, hyd_st1[t_ind_1_:t_ind_2_].pressure / (1.e6), color='tab:red', ls='-', lw=2, label='measured pressure (ST1)')
        ax11.plot(hyd_mb[t_ind_1_:t_ind_2_].time / 3600.0, hyd_mb[t_ind_1_:t_ind_2_].pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure ('+str(sensor)+')')
        ax11.plot(hyd_res_1.time / 3600.0, hyd_res_1.pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')
        ax11.plot(hyd_res_2.time / 3600.0, hyd_res_2.pressure / (1.e6), color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC pressure')
        ax11.yaxis.set_label_position('right')
        ax11.yaxis.set_ticks_position('right')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_11, labels_11 = ax11.get_legend_handles_labels()
        lns_1 = lines_1 + lines_11
        lbls_1 = labels_1 + labels_11

        ax1.set_xlim(0.0, x_axis_max)
        ax1.set_ylabel('Flow rate [L/min]', fontsize=fs+2)
        ax1.tick_params(axis='x', bottom=False, top=False, labelbottom=False)

        ax11.set_xlim(0.0, x_axis_max)
        ax11.legend(lns_1, lbls_1, loc='upper right', fontsize=fs)
        ax11.set_ylabel('Pressure [MPa]', fontsize=fs+2)
        ax11.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    else:
        # Plot strain timeseries
        ind1 = np.argmin(np.abs(DX[:,0] - 119.5))
        ax1.plot(strain_dss['time_sec'][t_ind_1:t_ind_2] / 3600.0, strain_diff[ind1, :], color='tab:purple', ls='-', lw=2, label='DSS data timeseries at 119.5 m')
        # ax1.plot(strain_dss['time_sec'][t_ind_1:t_ind_2] / 3600.0, (strain_diff[ind1, :] - min(strain_diff[ind1, :])) / (max(strain_diff[ind1, :]) - min(strain_diff[ind1, :])), color='tab:purple', ls='-', lw=2, label='DSS data timeseries at 119.5 m')
        ax11 = ax1.twinx()
        ind2 = np.argmin(np.abs(DX_[:,0] - 113.7))
        ax11.plot(strain_tf['time_shifted'][0][t_ind_1_:t_ind_2_] / 3600.0, strain_diff_[ind2, :], color='tab:green', ls='-', lw=2, label='TOUGH-FLAC model timeseries at 113.7 m')
        # ax1.plot(strain_tf['time_shifted'][0][t_ind_1_:t_ind_2_] / 3600.0, (strain_diff_[ind2, :] - min(strain_diff_[ind2, :])) / (max(strain_diff_[ind2, :]) - min(strain_diff_[ind2, :])), color='tab:green', ls='-', lw=2, label='TOUGH-FLAC model timeseries at 113.7 m')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_11, labels_11 = ax11.get_legend_handles_labels()
        lns_1 = lines_1 + lines_11
        lbls_1 = labels_1 + labels_11

        ax1.set_xlim(0.0, x_axis_max)
        ax1.set_ylabel(r'Strain [$\mu\epsilon$]')
        ax1.legend(lns_1, lbls_1, loc='upper right', fontsize=fs)
        ax1.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
        ax1.yaxis.label.set_color('tab:purple')
        ax1.tick_params(axis='y', colors='tab:purple')
        ax11.yaxis.set_label_position('right')
        ax11.yaxis.set_ticks_position('right')
        ax11.yaxis.label.set_color('tab:green')
        ax11.tick_params(axis='y', colors='tab:green')
        ax11.set_ylabel(r'Strain [$\mu\epsilon$]')

    # colormaps
    cmap1 = plt.cm.get_cmap('RdBu_r')   # for strain

    # DSS strain
    if cbar_range=='max_min':
        min_val = strain_diff.min()
        max_val = strain_diff.max()
        min_extr = np.round(np.min((np.abs(min_val), max_val)), decimals=0)
        levels1 = np.linspace(-min_extr, min_extr, 100)
        strain_map = ax2.contourf(DT, DX, strain_diff, levels1, cmap=cmap1, vmin=-min_extr, vmax=min_extr, extend='both')
        strain_cbar = fig.colorbar(strain_map, ax=ax2_, ticks=np.linspace(-min_extr, min_val, 7), shrink=0.8)
    else:
        min_val = strain_min
        max_val = strain_max
        levels1 = np.linspace(min_val, max_val, 100)
        strain_map = ax2.contourf(DT, DX, strain_diff, levels1, cmap=cmap1, vmin=min_val, vmax=max_val, extend='both')
        strain_cbar = fig.colorbar(strain_map, ax=ax2_, ticks=np.linspace(min_val, max_val, 7), shrink=0.8)

    if plot_strain_timeseries==True:
        ax2.axhline(DX[ind1][0], color='tab:purple', ls='--', lw=4, label='119.5 m')
        ax2.legend(loc='upper right', fontsize=fs)

    ax2.set_xlim(0.0, x_axis_max)
    ax2.set_ylim(bottomdepth, topdepth)
    ax2.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax2.set_ylabel('Borehole length [m]', fontsize=fs+2)
    strain_cbar.set_label(r'Strain [$\mu\epsilon$]', fontsize=fs+2)
    # legend

    # TOUGH-FLAC strain
    if cbar_range=='max_min':
        min_val = strain_diff_.min()
        max_val = strain_diff_.max()
        min_extr = np.round(np.min((np.abs(min_val), max_val)), decimals=0)
        levels2 = np.linspace(-min_extr, min_extr, 100)
        strain_map_ = ax3.contourf(DT_, DX_, strain_diff_, levels2, cmap=cmap1, vmin=-min_extr, vmax=min_extr, extend='both')
        strain_cbar_ = fig.colorbar(strain_map_, ax=ax3_, ticks=np.linspace(-min_extr, min_val, 7), shrink=0.8)
    else:
        min_val = strain_min / 2.
        max_val = strain_max / 2.
        levels2 = np.linspace(min_val, max_val, 100)
        strain_map_ = ax3.contourf(DT_, DX_, strain_diff_, levels2, cmap=cmap1, vmin=min_val, vmax=max_val, extend='both')
        strain_cbar_ = fig.colorbar(strain_map_, ax=ax3_, ticks=np.linspace(min_val, max_val, 7), shrink=0.8)

    if plot_strain_timeseries==True:
        ax3.axhline(DX_[ind2][0], color='tab:green', ls='--', lw=4, label='113.7 m')
        ax3.legend(loc='upper right', fontsize=fs)

    ax3.set_xlim(0.0, x_axis_max)
    ax3.set_ylim(bottomdepth, topdepth)
    # ax3.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax3.set_ylabel('Borehole length [m]', fontsize=fs+2)
    ax3.set_xlabel('Time since start of simulation [hours]', fontsize=fs+2)
    strain_cbar_.set_label(r'Strain [$\mu\epsilon$]', fontsize=fs+2)
    # legend

    fig.suptitle('DSS borehole ' + str(sensor) + ' - VALTER Phase I injection - Int. 13', y=0.95)

    # gs.tight_layout(fig, rect=[0.0, 0.0, 1.0, 1.0])

    return fig

def plot_MB8_results(hyd_st1,
                     hyd_mb,
                     hyd_res_1,
                     hyd_res_2,
                     strain_dss,
                     strain_tf,
                     sim_start,
                     sim_end,
                     fault_interfaces,
                     time_series_depths,
                     fs=20,
                     bottomdepth=150,
                     topdepth=50,
                     strain_min=-100,
                     strain_max=100,
                     sensor='MB8',
                     cbar_range='max_min',
                     plot_strain_timeseries=False):

    fig = plt.figure(figsize=[48, 36])
    gs = gridspec.GridSpec(nrows=2, ncols=4, hspace=0.05, wspace=0, width_ratios=[3.5, 0.5, 1.5, 3.5])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_ = fig.add_subplot(gs[0, 1])
    ax1_.axis('off')
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_ = fig.add_subplot(gs[1, 1])
    ax2_.axis('off')
    ax3 = fig.add_subplot(gs[0, 3:5])
    # ax33 = ax3.twinx()
    ax4 = fig.add_subplot(gs[1, 3:5])
    # ax44 = ax4.twinx()

    # check the start and end datetimes
    strain_dss['time'] = np.asarray([dt - datetime.timedelta(hours=1) for dt in strain_dss['time']])
    sorted_times = sorted([strain_dss['time'][0], strain_dss['time'][-1], sim_start, sim_end])
    starttime, endtime = sorted_times[1], sorted_times[2]
    # find the indices of datetime objects between the starttime and endtime range (DSS data)
    time_indices = [index for index, dt in enumerate(strain_dss['time']) if starttime <= dt <= endtime]
    t_ind_1, t_ind_2 = time_indices[0], time_indices[-1]
    # find the indices of datetime objects between the starttime and endtime range (hydraulic data)
    datetime_objects = [datetime.datetime.utcfromtimestamp(dt.astype(datetime.datetime) // 10 ** 9) for dt in np.asarray(hyd_st1.index)]
    time_indices_ = [index for index, dt in enumerate(datetime_objects) if starttime <= dt <= endtime]
    t_ind_1_, t_ind_2_ = time_indices_[0], time_indices_[-1]

    if (strain_tf['time'][0, 0] - hyd_st1.time[t_ind_1_]) > 0:
        t_ind_1_ = int(0)
    else:
        t_ind_1_ = int(np.argmin(np.abs(strain_tf['time'][0] - hyd_st1.time[t_ind_1_])))
    if (strain_tf['time'][0, -1] - hyd_st1.time[t_ind_2_]) < 0:
        t_ind_2_ = t_ind_2_
    else:
        t_ind_2_ = int(np.argmin(np.abs(strain_tf['time'][0] - hyd_st1.time[t_ind_2_])))

    # or get the time-shift for the x-axis in seconds, for correction
    t_shift = (starttime - datetime_objects[0]).total_seconds()
    # correct time in hydraulics and in strain from TOUGH-FLAC
    strain_tf['time_shifted'] = strain_tf['time'] - t_shift
    hyd_st1['time_shifted'] = hyd_st1['time'] - t_shift
    hyd_mb['time_shifted'] = hyd_mb['time'] - t_shift
    # correct also the hydraulics results
    hyd_res_1['time_shifted'] = hyd_res_1.time - t_shift
    hyd_res_2['time_shifted'] = hyd_res_2.time - t_shift

    # DSS compute meshgrids for strain and filtered strain difference since t_init
    strain_dss['time_sec'] = np.asarray([(dt - starttime).total_seconds() for dt in strain_dss['time']])
    DT, DX = np.meshgrid(strain_dss['time_sec'][t_ind_1:t_ind_2] / 3600.0, strain_dss['distance'])
    strain_diff = strain_dss['strain_filtered'][t_ind_1:t_ind_2, :].T - (strain_dss['strain_filtered'][t_ind_1:t_ind_2, :].T)[:, 0][:, np.newaxis]
    # TOUGH-FLAC
    DT_, DX_ = np.meshgrid(strain_tf['time_shifted'][0][t_ind_1_:t_ind_2_] / 3600.0, strain_tf['distance'])
    strain_diff_ = (np.asarray(strain_tf['strain'])[t_ind_1_:t_ind_2_, :].T - (np.asarray(strain_tf['strain'])[t_ind_1_:t_ind_2_, :].T)[:, 0][:, np.newaxis]) * (1.e6)  # from strain to microstrain

    x_axis_max = strain_tf['time_shifted'][0][t_ind_1_:t_ind_2_][-1] / 3600.0

    # colormaps
    cmap1 = plt.cm.get_cmap('RdBu_r')  # for strain

    # DSS strain
    if cbar_range == 'max_min':
        min_val = strain_diff.min()
        max_val = strain_diff.max()
        min_extr = np.round(np.min((np.abs(min_val), max_val)), decimals=0)
        levels1 = np.linspace(-min_extr, min_extr, 100)
        strain_map = ax1.contourf(DT, DX, strain_diff, levels1, cmap=cmap1, vmin=-min_extr, vmax=min_extr, extend='both')
        strain_cbar = fig.colorbar(strain_map, ax=ax1_, ticks=np.linspace(-min_extr, min_val, 9), shrink=0.8)
    else:
        min_val = strain_min
        max_val = strain_max
        levels1 = np.linspace(min_val, max_val, 100)
        strain_map = ax1.contourf(DT, DX, strain_diff, levels1, cmap=cmap1, vmin=min_val, vmax=max_val, extend='both')
        strain_cbar = fig.colorbar(strain_map, ax=ax1_, ticks=np.linspace(min_val, max_val, 9), shrink=0.8)

    # plot lower and upper fault interfaces
    ind_iu = np.argmin(np.abs(DX[:, 0] - fault_interfaces[0]))
    ax1.axhline(DX[ind_iu][0], color='k', ls='--', lw=1)
    ind_ib = np.argmin(np.abs(DX[:, 0] - fault_interfaces[1]))
    ax1.axhline(DX[ind_ib][0], color='k', ls='--', lw=1)

    if plot_strain_timeseries == True:
        ind1 = np.argmin(np.abs(DX[:, 0] - time_series_depths[0]))
        ax1.axhline(DX[ind1][0], color='tab:purple', ls='--', lw=2, label=str(time_series_depths[0]) + ' m')

    ax1.legend(loc='upper right', fontsize=fs)

    ax1.set_xlim(0.0, x_axis_max)
    ax1.set_ylim(bottomdepth, topdepth)
    ax1.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax1.set_ylabel('Borehole length [m]', fontsize=fs + 2)
    strain_cbar.set_label(r'Strain [$\mu\epsilon$]', fontsize=fs + 2)
    # legend

    # TOUGH-FLAC strain
    if cbar_range == 'max_min':
        min_val = strain_diff_.min()
        max_val = strain_diff_.max()
        min_extr = np.round(np.min((np.abs(min_val), max_val)), decimals=0)
        levels2 = np.linspace(-min_extr, min_extr, 100)
        strain_map_ = ax2.contourf(DT_, DX_, strain_diff_, levels2, cmap=cmap1, vmin=-min_extr, vmax=min_extr, extend='both')
        strain_cbar_ = fig.colorbar(strain_map_, ax=ax2_, ticks=np.linspace(-min_extr, min_val, 9), shrink=0.8)
    else:
        min_val = strain_min / 2.
        max_val = strain_max / 2.
        levels2 = np.linspace(min_val, max_val, 100)
        strain_map_ = ax2.contourf(DT_, DX_, strain_diff_, levels2, cmap=cmap1, vmin=min_val, vmax=max_val, extend='both')
        strain_cbar_ = fig.colorbar(strain_map_, ax=ax2_, ticks=np.linspace(min_val, max_val, 9), shrink=0.8)

    # plot lower and upper fault interfaces
    ind_iu = np.argmin(np.abs(DX_[:, 0] - fault_interfaces[0]))
    ax2.axhline(DX_[ind_iu][0], color='k', ls='--', lw=1)
    ind_ib = np.argmin(np.abs(DX_[:, 0] - fault_interfaces[1]))
    ax2.axhline(DX_[ind_ib][0], color='k', ls='--', lw=1)

    if plot_strain_timeseries == True:
        ind2 = np.argmin(np.abs(DX_[:, 0] - time_series_depths[1]))
        ax2.axhline(DX_[ind2][0], color='tab:green', ls='--', lw=2, label=str(time_series_depths[1]) + ' m')
    ax2.legend(loc='upper right', fontsize=fs)

    ax2.set_xlim(0.0, x_axis_max)
    ax2.set_ylim(bottomdepth, topdepth)
    # ax3.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax2.set_ylabel('Borehole length [m]', fontsize=fs + 2)
    ax2.set_xlabel('Time since start of simulation [hours]', fontsize=fs + 2)
    strain_cbar_.set_label(r'Strain [$\mu\epsilon$]', fontsize=fs + 2)
    # legend

    # Plot hydraulics
    if sensor in ['MB1', 'MB4', 'MB5']:
        sensor_label = 'ST1'
    else:
        sensor_label = sensor
    ax3.plot(hyd_mb[t_ind_1_:t_ind_2_].time_shifted / 3600.0, hyd_mb[t_ind_1_:t_ind_2_].pressure / (1.e6), color='k', ls='-', lw=2, label='measured pressure (' + str(sensor_label) + ')')
    ax3.plot(hyd_res_1.time_shifted / 3600.0, hyd_res_1.pressure / (1.e6), color='tab:red', ls='-', lw=2, label='TOUGH pressure')
    ax3.plot(hyd_res_2.time_shifted / 3600.0, hyd_res_2.pressure / (1.e6), color='tab:orange', ls='-', lw=2, label='TOUGH-FLAC pressure')
    # ax33.yaxis.set_label_position('right')
    # ax33.yaxis.set_ticks_position('right')

    # lines_3, labels_3= ax3.get_legend_handles_labels()
    # lines_33, labels_33 = ax33.get_legend_handles_labels()
    # lns_3 = lines_3 + lines_33
    # lbls_3 = labels_3 + labels_33

    ax3.set_xlim(0.0, x_axis_max)
    ax3.legend(loc='upper right', fontsize=fs)
    ax3.set_ylabel('Pressure [MPa]', fontsize=fs + 2)
    ax3.tick_params(axis='x', bottom=False, top=False, labelbottom=False)

    # ax33.set_xlim(0.0, x_axis_max)
    # ax33.legend(lns_3, lbls_3, loc='upper right', fontsize=fs)
    # ax33.set_ylabel('Permeability change [m²]', fontsize=fs + 2)
    # ax33.tick_params(axis='x', bottom=False, top=False, labelbottom=False)

    # Plot strain timeseries
    ind1 = np.argmin(np.abs(DX[:, 0] - time_series_depths[0]))
    ax4.plot(strain_dss['time_sec'][t_ind_1:t_ind_2] / 3600.0, strain_diff[ind1, :], color='tab:purple', ls='-', lw=2, label='DSS data - ' + str(time_series_depths[0]) + ' m')
    # ax4.plot(strain_dss['time_sec'][t_ind_1:t_ind_2] / 3600.0, (strain_diff[ind1, :] - min(strain_diff[ind1, :])) / (max(strain_diff[ind1, :]) - min(strain_diff[ind1, :])), color='tab:purple', ls='-', lw=2, label='DSS data timeseries at 119.5 m')
    ax44 = ax4.twinx()
    ind2 = np.argmin(np.abs(DX_[:, 0] - time_series_depths[1]))
    ax44.plot(strain_tf['time_shifted'][0][t_ind_1_:t_ind_2_] / 3600.0, strain_diff_[ind2, :], color='tab:green', ls='-', lw=2, label='TOUGH-FLAC - ' + str(time_series_depths[1]) + ' m')
    # ax4.plot(strain_tf['time_shifted'][0][t_ind_1_:t_ind_2_] / 3600.0, (strain_diff_[ind2, :] - min(strain_diff_[ind2, :])) / (max(strain_diff_[ind2, :]) - min(strain_diff_[ind2, :])), color='tab:green', ls='-', lw=2, label='TOUGH-FLAC model timeseries at 113.7 m')

    lines_4, labels_4 = ax4.get_legend_handles_labels()
    lines_44, labels_44 = ax44.get_legend_handles_labels()
    lns_4 = lines_4 + lines_44
    lbls_4 = labels_4 + labels_44

    ax4.set_xlim(0.0, x_axis_max)
    ax4.set_ylabel(r'Strain [$\mu\epsilon$]', fontsize=fs+2)
    ax4.legend(lns_4, lbls_4, loc='upper right', fontsize=fs)
    # ax4.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax4.set_xlabel('Time since start of simulation [hours]', fontsize=fs + 2)
    ax4.yaxis.label.set_color('tab:purple')
    ax4.tick_params(axis='y', colors='tab:purple')
    ax44.yaxis.set_label_position('right')
    ax44.yaxis.set_ticks_position('right')
    ax44.yaxis.label.set_color('tab:green')
    ax44.tick_params(axis='y', colors='tab:green')
    ax44.set_ylabel(r'Strain [$\mu\epsilon$]', fontsize=fs+2)

    fig.suptitle('Strain and Pressure at borehole ' + str(sensor) + ' - VALTER Phase I injection - Int. 13', y=0.95)

    # gs.tight_layout(fig, rect=[0.0, 0.0, 1.0, 1.0])

    return fig