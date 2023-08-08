"""
@ by Victor Clasen Repollés (victor.clasen@sed.ethz.ch)
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
    mesh.read_output('OUTPUT_ELEME.csv', time_step=-1)

    pyvista_mesh = mesh.to_pyvista()
    # pyvista_mesh = pyvista.from_meshio(mesh)
    # optionally clip the mesh along the x-axis to reduce from 10 km to 1 km
    x_cut = 2000.0
    clipped_mesh = pyvista_mesh.clip(invert=True, normal=[1, 0, 0], origin=[x_cut, 0, 0])
    # and cut the large boundary cells from the plot ...
    # z_cut = 0.0
    # clipped_mesh = clipped_mesh.clip(invert=True, normal=[0, 0, 1], origin=[0, 0, z_cut])

    p = pyvista.Plotter(window_size=(1000, 1000))
    p.add_mesh(
        clipped_mesh,
        scalars='PRES',
        cmap='viridis_r',
        # clim=(308.0, 315.0),
        n_colors=50,
        show_edges=False,
        edge_color=(0.5, 0.5, 0.5),
        scalar_bar_args={
            'title': 'Pressure',
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
    name_plot_2 = 'pressure'

    # ... at what time moment (integer values)
    time_moment_plot = 0                               # -1 for the last time moment

    # ... and cut the plot at the following radial position from zero (note that the boundary is cut automatically)
    x_cut = 3000.0                                      # in meters

########################################################################################
    # Import the required libraries

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.interpolate import griddata
    from matplotlib import gridspec

    import matplotlib
    matplotlib.use('Qt5Agg')

    # setting up for plotting
    plt.rcParams['figure.dpi'] = 50
    plt.rcParams['font.size'] = 40
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    # Plotting function
    def contourmap_plot(x, z, map1, map2, name1, name2, figname='contourplots.png'):

        fig = plt.figure(figsize=[24, 12])
        gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.5)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        cp1 = ax1.contourf(x, z, map1, levels=100, cmap='jet')
        cb1 = plt.colorbar(cp1, ax=ax1, pad=0.1)
        cb1.set_label(name1)

        cp2 = ax2.contourf(x, z, map2, levels=100, cmap='jet')
        cb2 = plt.colorbar(cp2, ax=ax2, pad=0.1)
        cb2.set_label(name2)

        ax1.set_xlabel('R (m)')
        ax1.set_ylabel('Z (m)')

        ax2.set_xlabel('R (m)')
        ax2.set_ylabel('Z (m)')

        fig.savefig(figname)
        fig.show()

        return

    # Read the output produced by TOUGH3

    # filepath = '/home/victor/Desktop/toughio/examples/multiphase_fluid_flow_for_natural_gas_extraction/figures/in_cond_test_2/files/OUTPUT_ELEME.csv'
    filepath = 'OUTPUT_ELEME.csv'

    ele_raw = pd.read_csv(filepath, index_col=False, usecols=[0])
    ids_ele1 = ele_raw[ele_raw.values == '             A11 0'].index.values                                                 # 13 empty spaces before element ID, then 5 characters for element ID -> if chenges, find out with len(ele_raw.values[X,0])
    ids_t = ids_ele1 - 1
    times_raw = ele_raw.values[ids_t, 0]
    times_corr = []
    [times_corr.append(float(times_raw[i][11:])) for i in range(len(times_raw))]                                            # in seconds

    # print output times:
    print('Found output at the following times:', )
    [print('TIME [years]: ' + str(int(times_corr[i]/(60*60*24*365.25)))) for i in range(len(times_corr))]

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
        mass_fraction_com2_gas='X_AIR_G',
        mass_fraction_com2_liquid='X_AIR_L'
    )

    # keep the data at specified time moment
    d = data[time_moment_plot]
    # filter out the boundary elements
    d = d[d['ROCK']!=7]
    # reset indices
    d = d.reset_index(drop=True)

    # prepare data for meshgrid and contourplot
    # 2D
    z_array = pd.Series(d['Z'].unique()).sort_values(ascending=False).values
    x_array = pd.Series(d['X'].unique()).sort_values(ascending=True).values
    x_array = x_array[x_array<=x_cut]
    X, Z = np.meshgrid(x_array, z_array)
    image_1 = griddata((d.X.values, d.Z.values), d[my_dict[name_plot_1]].values, (X, Z), method='linear')
    image_2 = griddata((d.X.values, d.Z.values), d[my_dict[name_plot_2]].values, (X, Z), method='linear')

    # plotting
    contourmap_plot(X, Z, image_1, image_2, name_plot_1, name_plot_2, figname='contourplot_coarse.png')

    # refining the mesh (e.g. finer and linear instead of coarser and log)
    # x_fine = np.linspace(-1000.0, 3000.0, 20000)
    # z_fine = np.linspace(-2000.0, -4000.0, 5000) * (-1)
    # X_, Z_ = np.meshgrid(x_fine, z_fine)
    # image_1_ = griddata((d.X.values, d.Z.values), d[my_dict[name_plot_1]].values, (X_, Z_), method='linear')
    # image_2_ = griddata((d.X.values, d.Z.values), d[my_dict[name_plot_2]].values, (X_, Z_), method='linear')

    # plotting
    # Note: if I do the finer mesh I may need to get rid of the levels in plotting (check this)
    # contourmap_plot(X_, Z_, image_1_, image_2_, name_plot_1, name_plot_2, figname='contourplot_refined.png')