"""
@ by Victor Clasen Repollés (victor.clasen@sed.ethz.ch)
"""

# Import the required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.interpolate import griddata
from matplotlib import gridspec
import matplotlib.colors as mcolors
import cmcrameri as crmi
import math

import matplotlib
matplotlib.use('Qt5Agg')

# setting up for plotting
plt.rcParams['figure.dpi'] = 50
plt.rcParams['font.size'] = 40
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Plotting function

# Read the output produced by TOUGH3
filepath = 'OUTPUT_ELEME.csv'

ele_raw = pd.read_csv(filepath, index_col=False, usecols=[0])
ids_ele1 = ele_raw[ele_raw.values == '             A11 0'].index.values
ids_t = ids_ele1 - 1
times_raw = ele_raw.values[ids_t, 0]
times_corr = []
[times_corr.append(float(times_raw[i][11:])) for i in range(len(times_raw))]

# print output times:
print('Found output at the following times:', )
[print('TIME [seconds]: ' + str(int(times_corr[i]))) for i in range(len(times_corr))]

data = []
header_names_raw = pd.read_csv(filepath, nrows=0).columns
header_names = pd.Index([element.strip() for element in header_names_raw])
for j in range(len(ids_t)):
    if j==0:
        df = pd.read_csv(filepath, header=None, index_col=False, skiprows=list(range(0, ids_t[j]+2)), nrows=((ids_t[j+1]-1)-ids_t[j]))
    elif j==(len(ids_t)-1):
        df = pd.read_csv(filepath, header=None, index_col=False, skiprows=list(range(0, ids_t[j]+2)), nrows=((len(ele_raw.values)-1)-ids_t[j]))
    else:
        df = pd.read_csv(filepath, header=None, index_col=False, skiprows=list(range(0, ids_t[j]+2)), nrows=((ids_t[j+1]-1)-ids_t[j]))
    df = df.reset_index(drop=True)
    df.columns = header_names
    data.append(df)

pres_mat = np.zeros((len(data[0].X), len(times_corr)))

for l in range(len(pres_mat[0,:])):
    pres_mat[:, l] = data[l].PRES

time_array = np.array(times_corr)
dist_array = data[0].X.values

def injection_protocol(flow_rate, time_array, inj_times, inj_rates):
    idx_all = []
    for j in range(len(inj_times)):
        idx_all.append(int(np.argmin(np.abs(time_array - inj_times[j]))))
    idx_uniq = np.unique(idx_all)
    if idx_uniq[-1]!=(len(time_array)-1):
        for t in range(len(idx_uniq)):
            flow_rate[idx_uniq[t]:] = inj_rates[t]
    else:
        """
        TO DO: improve this part to be adaptive to several step changes, now it is only true for the last step!
        """
        flow_rate[idx_uniq[0]:] = inj_rates[0]
    return flow_rate

def plot_pres_evol(time_array, dist_array, flow_rate, pres_mat, scaling_time = 3600):
    fig = plt.figure(figsize=[48, 12])
    gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.plot(time_array / scaling_time, flow_rate, color='tab:blue', lw=2, ls='-')
    ax11 = ax1.twinx()
    ax11.plot(time_array / scaling_time, pres_mat[0, :] / (1.e6), color='tab:red', lw=2, ls='-')

    ax1.set_xlabel('Time from injection [h]')

    ax1.set_ylabel('Flow rate [L/min]')
    ax1.yaxis.label.set_color('tab:blue')
    ax1.tick_params(axis='y', colors='tab:blue')
    ax11.set_ylabel('Pressure [MPa]')
    ax11.yaxis.label.set_color('tab:red')
    ax11.tick_params(axis='y', colors='tab:red')

    # How many spatial positions to plot?
    nr_seq = 8
    len(pres_mat[:, 0]) / nr_seq
    dist_ind_array = np.linspace(0, len(pres_mat[:, 0]), nr_seq + 1)[:-1]

    # color sequence
    csteps = math.floor(len(crmi.cm.hawaii.colors) / nr_seq)

    ax2.plot(time_array / scaling_time, flow_rate, color='tab:blue', lw=2, ls='-')
    ax22 = ax2.twinx()
    nrun = 0
    for i in range(len(dist_ind_array)):
        ax22.plot(time_array / scaling_time, pres_mat[int(dist_ind_array[i]), :] / (1.e6),
                  color=crmi.cm.hawaii.colors[nrun * csteps, :], lw=2, ls='-',
                  label=str(np.round(dist_array[int(dist_ind_array[i])], decimals=2)) + ' m')
        nrun = nrun + 1

    ax2.set_xlabel('Time from injection [h]')

    ax2.set_ylabel('Flow rate [L/min]')
    ax2.yaxis.label.set_color('tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')
    ax22.set_ylabel('Pressure [MPa]')

    lns, lbls = ax22.get_legend_handles_labels()
    ax22.legend(lns, lbls, loc='best')

    # How many timesteps to plot?
    nr_seq = 8
    len(pres_mat[0, :]) / nr_seq
    time_ind_array = np.linspace(0, len(pres_mat[0, :]), nr_seq + 1)[:-1]

    ax33 = ax3.twinx()

    nrun = 0
    for i in range(len(time_ind_array)):
        ax33.plot(dist_array, pres_mat[:, int(time_ind_array[i])] / (1.e6),
                  color=crmi.cm.hawaii.colors[nrun * csteps, :], lw=2, ls='-',
                  label=str(np.round(time_array[int(time_ind_array[i])] / scaling_time, decimals=2)) + ' h')
        nrun = nrun + 1

    ax3.set_yticks([])
    # ax3.set_xticks([])
    ax3.set_xlabel('Distance from injection point [m]')
    ax33.set_ylabel('Pressure [MPa]')

    lns, lbls = ax33.get_legend_handles_labels()
    ax33.legend(lns, lbls, loc='best')

    return

# # Load injection protocol
# path_folder = '/home/victor/Desktop/toughio/examples/1D_fluid_flow_4_HM1/'
# filename = 'inj_protocol.pkl'
# with open(path_folder + filename, 'rb') as f:
#     inj_times, inj_rates = pickle.load(f)
#
# flow_rate = np.zeros_like(time_array, dtype=float)
# flow_rate = injection_protocol(flow_rate, time_array, inj_times, inj_rates) * (1000 * 60) # flow rate from m³/s to L/min

# Load injection protocol (from generation rate in output file)
filepath = 'OUTPUT_GENER.csv'

"""
Uncomment if necessary to show also for flow rate data ...
"""
# ele_raw = pd.read_csv(filepath, index_col=False, usecols=[0])
# ids_ele1 = ele_raw[ele_raw.values == '             A11 0'].index.values
# ids_t = ids_ele1 - 1
# times_raw = ele_raw.values[ids_t, 0]
# times_corr = []
# [times_corr.append(float(times_raw[i][11:])) for i in range(len(times_raw))]
#
# # print output times:
# print('Found output at the following times:', )
# [print('TIME [seconds]: ' + str(int(times_corr[i]))) for i in range(len(times_corr))]

data2 = []
header_names_raw = pd.read_csv(filepath, nrows=0).columns
header_names = pd.Index([element.strip() for element in header_names_raw])

df_raw = pd.read_csv(filepath)
dp_array = (np.linspace(0, len(df_raw)-3, int((len(df_raw)-3)/2)+1) + 3).astype(int)    # datapoints array

flow_rate = pd.read_csv(filepath, header=None, index_col=False, usecols=[5], skip_blank_lines=True).values[dp_array][:,0].astype(float) * 60 # from kg/s to L/min

# plot pressure evolution
plot_pres_evol(time_array, dist_array, flow_rate, pres_mat)