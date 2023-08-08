"""
@ by Victor Clasen Repollés (clvictor@sed.ethz.ch)
"""

# plot_with = 'pyvista'
plot_with = 'matplotlib'

########################################################################################
# Option 1: fast plot, with pyvista
########################################################################################

if plot_with=='pyvista':

    import toughio
    import pyvista

    mesh = toughio.read_mesh('mesh.pickle')
    mesh.read_output('OUTPUT_ELEME_1.csv', time_step=-1)

    pyvista_mesh = mesh.to_pyvista()
    # pyvista_mesh = pyvista.from_meshio(mesh)
    # optionally clip the mesh along the x-axis to reduce from 10 km to 1 km
    x_cut = 1000.0
    clipped_mesh = pyvista_mesh.clip(invert=True, normal=[1, 0, 0], origin=[x_cut, 0, 0])
    # and cut the large boundary cells from the plot ...
    z_cut = 0.0
    clipped_mesh = clipped_mesh.clip(invert=True, normal=[0, 0, 1], origin=[0, 0, z_cut])

    p = pyvista.Plotter(window_size=(1000, 1000))
    p.add_mesh(
        clipped_mesh,
        scalars='TEMP',
        cmap='viridis_r',
        # clim=(308.0, 315.0),
        n_colors=50,
        show_edges=False,
        edge_color=(0.5, 0.5, 0.5),
        scalar_bar_args={
            'title': 'Temperature',
            'position_y': 0.01,
            'vertical': False,
            'n_labels': 6,
            'fmt': '%.1f',
            'title_font_size': 20,
            'font_family': 'arial',
            'shadow': True,
        }
    )
    p.show_grid(
        show_xaxis=True,
        show_yaxis=False,
        show_zaxis=True,
        xlabel='Distance (m)',
        zlabel='Elevation (m)',
        ticks='outside',
        font_family='arial'
    )
    p.view_xz()
    p.show()

########################################################################################
# Option 2: matplotlib
########################################################################################

elif plot_with=='matplotlib':

########################################################################################
    # INPUTS (required)

    # Specify what to plot (see available names in the dictionary below)
    name_plot_1 = 'gas_saturation'
    name_plot_2 = 'temperature'

    # primary variable to be read from initial conditions for difference in time plot
    prim_var = 'temperature'

    # ... at what time moment (integer values)
    time_moment_plot = -13                               # -1 for the last time moment

    # ... and cut the plot at the following radial position from zero (note that the boundary is cut automatically)
    x_cut = 1000.0                                      # in meters

########################################################################################
    # Import the required libraries

    import toughio
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.interpolate import griddata
    from matplotlib import gridspec

    import matplotlib
    matplotlib.use('Qt5Agg')

    # setting up for plotting
    plt.rcParams['figure.dpi'] = 50
    plt.rcParams['font.size'] = 25
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    # Plotting function
    def contourmap_plot(x, z, map1, map2, name1, name2, time, set_levels=20, figname='contourplots.png'):

        fig = plt.figure(figsize=[24, 12])
        gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.5)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        cp1 = ax1.contourf(x, z, map1, levels=set_levels, cmap='jet')
        cb1 = plt.colorbar(cp1, ax=ax1, pad=0.1)
        cb1.set_label(name1)

        cp2 = ax2.contourf(x, z, map2, levels=set_levels, cmap='jet')
        cb2 = plt.colorbar(cp2, ax=ax2, pad=0.1)
        cb2.set_label(name2)

        ax1.set_xlabel('R (m)')
        ax1.set_ylabel('Z (m)')

        ax2.set_xlabel('R (m)')
        ax2.set_ylabel('Z (m)')

        fig.suptitle('t = ' + str(int(time/(60*60*24*365.25))) + ' years')
        fig.savefig(figname)
        # fig.show()
        # fig.close()

        return

    def diff_contourmap_plot(x, z, incon, map, name, times, set_levels=20, n_plots=3, figname='diff_contourplots.png'):

        fig = plt.figure(figsize=[24, 12])
        gs = gridspec.GridSpec(nrows=1, ncols=n_plots, wspace=0.6)
        ax1 = fig.add_subplot(gs[0])
        if n_plots==2:
            ax2 = fig.add_subplot(gs[1])

            cp1 = ax1.contourf(x[0], z[0], map[0] - incon, levels=set_levels, cmap='jet_r')
            cb1 = plt.colorbar(cp1, ax=ax1, pad=0.1)
            cb1.set_label(name + ' difference (°C)')

            cp2 = ax2.contourf(x[1], z[1], map[1] - incon, levels=set_levels, cmap='jet_r')
            cb2 = plt.colorbar(cp2, ax=ax2, pad=0.1)
            cb2.set_label(name + ' difference (°C)')

            ax1.set_xlabel('R (m)')
            ax1.set_ylabel('Z (m)')

            ax1.set_title('t = ' + str(int(times[0]/(60*60*24*365.25))) + ' years')

            ax2.set_xlabel('R (m)')
            ax2.set_ylabel('Z (m)')

            ax2.set_title('t = ' + str(int(times[1]/(60*60*24*365.25))) + ' years')

        elif n_plots==3:
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])

            cp1 = ax1.contourf(x[0], z[0], map[0] - incon, levels=set_levels, cmap='jet_r')
            cb1 = plt.colorbar(cp1, ax=ax1, pad=0.1)
            cb1.set_label(name + ' difference (°C)')

            cp2 = ax2.contourf(x[1], z[1], map[1] - incon, levels=set_levels, cmap='jet_r')
            cb2 = plt.colorbar(cp2, ax=ax2, pad=0.1)
            cb2.set_label(name + ' difference (°C)')

            cp3 = ax3.contourf(x[2], z[2], map[2] - incon, levels=set_levels, cmap='jet_r')
            cb3 = plt.colorbar(cp3, ax=ax3, pad=0.1)
            cb3.set_label(name + ' difference (°C)')

            ax1.set_xlabel('R (m)')
            ax1.set_ylabel('Z (m)')

            ax1.set_title('t = ' + str(int(times[0]/(60*60*24*365.25))) + ' years')

            ax2.set_xlabel('R (m)')
            ax2.set_ylabel('Z (m)')

            ax2.set_title('t = ' + str(int(times[1]/(60*60*24*365.25))) + ' years')

            ax3.set_xlabel('R (m)')
            ax3.set_ylabel('Z (m)')

            ax3.set_title('t = ' + str(int(times[2]/(60*60*24*365.25))) + ' years')

        else:   # 4 subplots (simualtions runs longer than the last set time snapshot)
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            ax4 = fig.add_subplot(gs[3])

            cp1 = ax1.contourf(x[0], z[0], map[0] - incon, levels=set_levels, cmap='jet_r')
            cb1 = plt.colorbar(cp1, ax=ax1, pad=0.1)
            cb1.set_label(name + ' difference (°C)')

            cp2 = ax2.contourf(x[1], z[1], map[1] - incon, levels=set_levels, cmap='jet_r')
            cb2 = plt.colorbar(cp2, ax=ax2, pad=0.1)
            cb2.set_label(name + ' difference (°C)')

            cp3 = ax3.contourf(x[2], z[2], map[2] - incon, levels=set_levels, cmap='jet_r')
            cb3 = plt.colorbar(cp3, ax=ax3, pad=0.1)
            cb3.set_label(name + ' difference (°C)')

            cp4 = ax4.contourf(x[3], z[3], map[3] - incon, levels=set_levels, cmap='jet_r')
            cb4 = plt.colorbar(cp4, ax=ax4, pad=0.1)
            cb4.set_label(name + ' difference (°C)')

            ax1.set_xlabel('R (m)')
            ax1.set_ylabel('Z (m)')

            ax1.set_title('t = ' + str(int(times[0]/(60.0*60.0*24.0*365.25))) + ' years')

            ax2.set_xlabel('R (m)')
            ax2.set_ylabel('Z (m)')

            ax2.set_title('t = ' + str(int(times[1]/(60.0*60.0*24.0*365.25))) + ' years')

            ax3.set_xlabel('R (m)')
            ax3.set_ylabel('Z (m)')

            ax3.set_title('t = ' + str(int(times[2]/(60.0*60.0*24.0*365.25))) + ' years')

            ax4.set_xlabel('R (m)')
            ax4.set_ylabel('Z (m)')

            ax4.set_title('t = ' + str(int(times[3]/(60.0*60.0*24.0*365.25))) + ' years')

        fig.savefig(figname)
        fig.show()

        return

    # Read the output produced by TOUGH3

    output_path = '/home/victor/Desktop/toughio/examples/heat_and_fluid_flow_from_magmatic_fluids/2_injection/'
    output_file = 'OUTPUT_ELEME.csv'
    output_filepath = output_path + output_file

    ele_raw = pd.read_csv(output_filepath, index_col=False, usecols=[0])
    ids_ele1 = ele_raw[ele_raw.values == '             A11 0'].index.values                                                 # 13 empty spaces before element ID, then 5 characters for element ID -> if chenges, find out with len(ele_raw.values[X,0])
    # ids_ele1 = ele_raw[ele_raw.values == '             A2  1'].index.values
    ids_t = ids_ele1 - 1
    times_raw = ele_raw.values[ids_t, 0]
    times_corr = []
    [times_corr.append(float(times_raw[i][11:])) for i in range(len(times_raw))]                                            # in seconds

    # print output times:
    print('Found output at the following times:', )
    [print('TIME [years]: ' + str(int(times_corr[i]/(60*60*24*365.25)))) for i in range(len(times_corr))]

    data = []
    header_names_raw = pd.read_csv(output_filepath, nrows=0).columns
    header_names = pd.Index([element.strip() for element in header_names_raw])
    for j in range(len(ids_t)):
        if j==0:    # only works if there are more than just one time output (!)
            df = pd.read_csv(output_filepath, header=None, index_col=False, skiprows=list(range(0, ids_t[j]+2)), nrows=((ids_t[j+1]-1)-ids_t[j]))
        elif j==(len(ids_t)-1):
            df = pd.read_csv(output_filepath, header=None, index_col=False, skiprows=list(range(0, ids_t[j]+2)), nrows=((len(ele_raw.values)-1)-ids_t[j]))
        else:
            df = pd.read_csv(output_filepath, header=None, index_col=False, skiprows=list(range(0, ids_t[j]+2)), nrows=((ids_t[j+1]-1)-ids_t[j]))
        df = df.reset_index(drop=True)
        df.columns = header_names
        data.append(df)

    # Create dictionary

    my_dict = dict(
        gas_density='DEN_G',
        liquid_density='DEN_L',
        capillary_pressure='PCAP_GL',
        porosity='POR',
        pressure='PRES',
        relative_permeability_gas='REL_G',
        relative_permeability_liquid='REL_L',
        gas_saturation='SAT_G',     # or volumetric gas fraction
        liquid_saturation='SAT_L',
        temperature='TEMP',
        mass_fraction_com1_gas='X_WATER_G',
        mass_fraction_com1_liquid='X_WATER_L',
        mass_fraction_com2_gas='X_CO2_G',
        mass_fraction_com2_liquid='X_CO2_L'
    )

    image_1_all_times = []          # these two empty lists will be filled with the two chosen parameter datasets for all saved simulation times
    image_2_all_times = []

    X_all_times = []                # same for X and Z matrices, in case they differ, which they should not!
    Z_all_times = []

    for t in range(len(times_corr)):

        # keep the data at specified time moment
        d = data[t]
        # # filter out the boundary elements (comment if there are no boundary elements)
        # d = d[d['ROCK']==1]
        # # reset indices
        # d = d.reset_index(drop=True)

        # prepare data for meshgrid and contourplot
        # 2D
        ids_low = d[d.X==min(d.X)].index.values
        x_array = d.X.values[ids_low[0]:ids_low[1]]
        # cut out until x_cut
        x_array = x_array[x_array<=x_cut]
        z_array = d.Z.values[ids_low]
        X, Z = np.meshgrid(x_array, z_array)
        X_all_times.append(X)
        Z_all_times.append(Z)
        # image_1 = griddata((d.X.values, d.Z.values), d[my_dict[name_plot_1]].values, (X, Z), method='linear')
        image_1_all_times.append(griddata((d.X.values, d.Z.values), d[my_dict[name_plot_1]].values, (X, Z), method='linear'))
        # image_2 = griddata((d.X.values, d.Z.values), d[my_dict[name_plot_2]].values, (X, Z), method='linear')
        image_2_all_times.append(griddata((d.X.values, d.Z.values), d[my_dict[name_plot_2]].values, (X, Z), method='linear'))

        # data preparation also for initial conditions only
        if t==len(times_corr)-1:

            # reading the initial conditions set up for the primary variables
            incon_data = toughio.read_input(output_path + 'INCON')
            # which data?
            if prim_var == 'pressure':
                ind = 0
            elif prim_var == 'temperature':
                ind = 1
            else:  # co2 partial pressure
                ind = 2
            # get the data without the boundary
            pv_init = []
            for i in range(len(d)):
                pv_init.append(incon_data['initial_conditions'][d.ELEM[i].strip()]['values'][ind])
            pv_init = np.asarray(pv_init)

            incon_image = griddata((d.X.values, d.Z.values), pv_init, (X, Z), method='linear')

    # plotting
    contourmap_plot(X_all_times[time_moment_plot], Z_all_times[time_moment_plot], image_1_all_times[time_moment_plot], image_2_all_times[time_moment_plot],
                    name_plot_1, name_plot_2, times_corr[time_moment_plot], set_levels=100, figname=output_path + 'contourplot_coarse.svg')

    # diff_contourmap_plot(X_all_times, Z_all_times, incon_image, image_2_all_times, name_plot_2, times_corr, set_levels=100,
    #                      n_plots=len(times_corr), figname=output_path + 'contourplot_diff_coarse.svg')
    # diff_contourmap_plot(X_all_times, Z_all_times, incon_image, image_2_all_times, name_plot_2, times_corr, set_levels=100,
    #                      n_plots=3, figname=output_path + 'contourplot_diff_coarse.svg')

    # refining the mesh (e.g. finer and linear instead of coarser and log)
    # x_fine = np.linspace(0.0, 1000.0, 150)
    # z_fine = np.linspace(0.0, 1500.0, 200) * (-1)
    # X_, Z_ = np.meshgrid(x_fine, z_fine)
    # image_1_ = griddata((d.X.values, d.Z.values), d[my_dict[name_plot_1]].values, (X_, Z_), method='linear')
    # image_2_ = griddata((d.X.values, d.Z.values), d[my_dict[name_plot_2]].values, (X_, Z_), method='linear')

    # plotting
    # Note: if I do the finer mesh I may need to get rid of the levels in plotting (check this)
    # contourmap_plot(X_, Z_, image_1_, image_2_, name_plot_1, name_plot_2, figname='contourplot_refined.png')