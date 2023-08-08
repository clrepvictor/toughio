"""
@ by Victor Clasen Repollés (clvictor@sed.ethz.ch)
"""

########################################################################################
# First, we import :mod:`numpy` and :mod:`pygmsh`.

import numpy as np
import pandas as pd
import pygmsh
import toughio
import itertools
import chardet
import matplotlib.pyplot as plt

########################################################################################

# 1) CREATE MESH #

"""
-> choose 'meshmaker' to create a 2D (radial) axisymmetric grid
-> choose 'pygmsh' to create a 2D plane grid with pygmsh
"""

meshmaker = 'meshmaker'
# meshmaker = 'pygmsh'

if meshmaker=='pygmsh':

    # Variables
    # useful to characterize the geometry of the model

    lc = 100.0                      # characteristic length
    r_min, r_max = 0.0, 10000.0     # radial boundaries (minimum and maximum radial distances of the mesh)
    z_min, z_max = 0.0, -1500.0     # z-axis boundaries (minimum and maximum depth distances of the mesh)

    # inj_x, inj_z = 0.0, -1500.0     # radial position and depth of injection

    bnd_thick = 100.0                # thickness of boundary elements

    # we define the coordinates for the injection point.
    inj_point_coord = [r_min, z_max, 0.0]
    # the motivation behind this is that Gmsh keeps the first node defined in the geometry (which we will define as being
    # the injection point), in case it detects duplicated nodes, such that the mesh is refined near that point.

    # we define the points of the domain: R, Z, Y

    left_down_coord = [r_min, z_max, 0.0]
    right_down_coord = [r_max, z_max, 0.0]
    right_up_coord = [r_max, z_min, 0.0]
    left_up_coord = [r_min, z_min, 0.0]

    ########################################################################################
    # BOUNDARY CONDITIONS #
    # impervious -> not allowing fluid to pass through
    # adiabatic -> process in which heat does not enter or leave the system concerned

    # impervious and adiabatic for bottom and side boundaries
    # atmospheric conditions are fixed along the upper boundary (i.e. surface), which is open to heat and fluid flows

    # Therefore, no-flow boundary condition is imposed on the sides and on the bottom boundaries (default on TOUGH)
    # On the upper boundary we have to impose atmospheric conditions (open to heat and fluid flows)

    bound_up_coord = [
        [r_min, z_min, 0.0],
        [r_max, z_min, 0.0],
        [r_max, z_min - bnd_thick, 0.0],
        [r_min, z_min - bnd_thick, 0.0],
    ]

    # Once all the points have been created, we can now generate the geometry, assign rock types/materials as Gmsh physical properties, and generate the mesh.

    with pygmsh.geo.Geometry() as geo:
        # injection point
        inj_point = geo.add_point(inj_point_coord, mesh_size=0.025*lc)
        # rectangular domain
        # rect_dom = geo.add_rectangle([left_down_coord, right_down_coord, right_up_coord, left_up_coord], mesh_size=2.0*lc)
        rect_dom = geo.add_polygon([left_down_coord, right_down_coord, right_up_coord, left_up_coord], mesh_size=0.75*lc)
        # upper boundary layer
        # bound_up = geo.add_rectangle(bound_up_coord, mesh_size=lc)
        bound_up = geo.add_polygon(bound_up_coord, mesh_size=2*lc)

        # Define materials
        # geo.add_physical(inj_point, 'INJPT') # the injection point is just a point that helps to refine the mesh, we do not include it as a cell
        geo.add_physical(rect_dom, 'HROCK')
        geo.add_physical(bound_up, 'BOUND')

        # Remove duplicate entities
        geo.env.removeAllDuplicates()
        mesh = geo.generate_mesh(dim=2, algorithm=6)

        # Convert cell sets to material
        cell_data = [np.empty(len(c.data), dtype=int) for c in mesh.cells]
        field_data = {}
        for i, (k, v) in enumerate(mesh.cell_sets.items()):
            if k:
                field_data[k] = np.array([i + 1, 3])
                for ii, vv in enumerate(v):
                    cell_data[ii][vv] = i + 1
        mesh.cell_data['material'] = cell_data
        mesh.field_data.update(field_data)
        mesh.cell_sets = {}

        # Remove lower dimensions entities
        idx = [i for i, cell in enumerate(mesh.cells) if cell.type == 'triangle']
        mesh.cells = [mesh.cells[i] for i in idx]
        mesh.cell_data = {k: [v[i] for i in idx] for k, v in mesh.cell_data.items()}

        # Export the mesh for post-processing
        mesh.write('mesh.vtu')

    # The generated mesh can be visualized in Python with :mod: 'pyvista'

    import pyvista
    pyvista.set_plot_theme("document")

    p = pyvista.Plotter(window_size=(800, 800))
    p.add_mesh(mesh=pyvista.from_meshio(mesh),
               scalar_bar_args={"title": "Materials"},
               show_scalar_bar=True,
               show_edges=True,
    )

    p.view_xy()
    p.show()

    del mesh
    mesh = toughio.read_mesh('mesh.vtu')

    # We now add material names:
    mesh.add_material('HROCK', 1)
    mesh.add_material('BOUND', 2)

    # The mesh used in this sample problem is 2D and has been defined in the XY plane, but the points have 3D coordinates
    # (with zeros as 3rd dimension for every cells). To make it 3D in the XY plane, we swap the 2nd and 3rd dimensions and
    # then extrude the mesh by 1 meter along the Y axis (2nd dimension).
    mesh.points[:, [1, 2]] = mesh.points[:, [2, 1]]
    mesh.extrude_to_3d(height=1.0, axis=1)

    # print the number of elements of the computational domain
    print('Number of elements: ' + str(len(mesh.centers)))

    # Check mesh quality:
    # TOUGH does not use any geometrical coordinate system and assumes that the line connecting a cell with its neighbor
    # is orthogonal to their common interface. :mod: 'toughio' provides a mesh property that measures the quality of a cell
    # as the average absolute cosine angle between the line connecting a cell with its neighbor and the normal vector of
    # the common interface.

    import pyvista
    pyvista.set_plot_theme("document")

    p = pyvista.Plotter(window_size=(800, 800))
    p.add_mesh(
        mesh=mesh.to_pyvista(),
        scalars=mesh.qualities,
        scalar_bar_args={"title": "Average cell quality"},
        clim=(0.0, 1.0),
        cmap="RdBu",
        show_scalar_bar=True,
        show_edges=True,
    )
    p.view_xz()
    p.show()

    # Usually, a simple distribution plot is enough to rapidly assess the quality of a mesh.

    # import seaborn
    # ax = seaborn.displot(mesh.qualities[mesh.materials != 'BOUND'], kind='hist')

    # We start now defining the boundary conditions:
    # :mod: 'toughio' recognizes the cell data key 'boundary_condition' and automatically imposes Dirichlet boundary
    # conditions to cells that have any value other than 0 in this cell data array.
    # In case of atmospheric boundary conditions, they can be specified using Dirichlet boundary elements with very large volumes.
    # Also a single atmospheric element can be connected to all elements at the ground surface (this should be tried)
    materials = mesh.materials
    bcond = (materials == 'BOUND').astype(int)
    mesh.add_cell_data('boundary_condition', bcond)

    # Initial conditions can be defined as a cell data array associated to key 'initial_condition' where each column of the
    # array corresponds to a primary variable. Note that :mod: 'toughio' will not write any initial condition value that is
    # lower than the threshold flag -1.0e9
    # single phase conditions: pressure, temperature, CO2 partial pressure
    # two-phase conditions: gas phase pressure, gas saturation, CO2 partial pressure

    # Also note that standard metric (SI) have to be used: meters, seconds, kilograms, °C, Newtons, Joules, Pascal, etc.
    centers = mesh.centers
    incon = np.full((mesh.n_cells, 3), -1.0e9)

    # single phase conditions
    incon[:, 0] = 1.1e5 - 2000.0 * 9.81 * centers[:, 2]                     # gas phase pressure
    incon[:, 1] = 0.85 + centers[:, 2]/(2 *np.max(abs(centers[:, 2])))      # gas saturation
    incon[:, 2] = 1.0e5                                                     # CO2 partial pressure
    mesh.add_cell_data('initial_condition', incon)

    plot_check = 'no'

    if plot_check=='yes':
        # plot to check the initial conditions
        fig = plt.figure(figsize=[20, 15])
        gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0.5, wspace=0.7)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(incon[:, 0], centers[:, 2], color='tab:blue')
        ax1.set_xlabel('gas phase pressure')
        ax1.set_ylabel('depth of cell center points')
        ax1.set_title('gas phase pressure against depth check')

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(incon[:, 1], centers[:, 2], color='tab:red')
        ax2.set_xlabel('gas saturation')
        ax2.set_ylabel('depth of cell center points')
        ax2.set_title('gas saturation against depth check')

        ax3 = fig.add_subplot(gs[2])
        ax3.plot(incon[:, 2], centers[:, 2], color='tab:orange')
        ax3.set_xlabel('CO2 partial pressure')
        ax3.set_ylabel('depth of cell center points')
        ax3.set_title('CO2 partial pressure against depth check')

        fig.savefig('initcond_check.png')

    # We can now write the 'MESH' and 'INCON' files by calling the method :meth: 'toughio.mesh.write_tough'
    # Additionally, we can also pickle the final mesh for later use (reading a pickle file is much faster than reading any
    # mesh format)

    mesh.write_tough('MESH', incon=True)
    mesh.write('mesh.pickle')

elif meshmaker=='meshmaker':

    def get_thicknesses(center_pts, x_init, x_end):
        """
        Function to get the cell thicknesses from the center points for one axis coordinates
        """
        thick_array = np.zeros_like(center_pts)
        for i in range(len(center_pts)):
            ht = center_pts[i] - x_init
            x_cell_max = center_pts[i] + ht
            thick_array[i] = x_cell_max - x_init
            x_init = x_cell_max
        print('------------------------------------------------------------')
        print('The sum of all thicknesses is: ' + str(np.sum(thick_array)), )
        print('The difference to the aimed section length/height is: ' + str(np.abs(np.sum(thick_array) - x_end)), )
        return thick_array

    # # MESH 1.0 #
    # z_thick = np.append(np.abs(np.diff(np.geomspace(0.2, 1500.2, 50) - 0.2)), 1000) # mesh 1
    #
    # # MESH 2.0 #
    # # z_thick = np.append(np.full(50, 30.0), 1000) # mesh 2
    #
    # parameters = {
    #     'meshmaker': {
    #         'type': 'rz2dl',
    #         'parameters': [
    #             {'type': 'radii', 'radii': [0.0]},
    #             {'type': 'equid', 'n_increment': 1, 'size': 0.15},
    #             {'type': 'logar', 'n_increment': 99, 'radius': 10000.0},
    #             {'type': 'layer', 'thicknesses': z_thick},
    #         ]
    #     }
    # }

    # # MESH 3.0 # testing mesh with equally sized cells for the entire length and depth of the domain
    # # r_thick = np.abs(np.diff(np.linspace(0.0, 10000.0, 201)))
    # z_thick = np.append(np.full(50, 30.0), 1000)
    #
    # parameters = {
    #     'meshmaker': {
    #         'type': 'rz2dl',
    #         'parameters': [
    #             {'type': 'radii', 'radii': [0.0]},
    #             {'type': 'equid', 'n_increment': 200, 'size': 50.0},
    #             {'type': 'layer', 'thicknesses': z_thick},
    #         ]
    #     }
    # }

    # MESH 4.0 #

    # with open('mesh_coord_centers.csv', 'r') as f:
    #     raw_data = f.read()
    #     result = chardet.detect(raw_data.encode())
    #     encode_format = result['encoding']

    # df = pd.read_csv('mesh_coord_centers.csv', header=None, index_col=None, skiprows=[0, 1], nrows=0)

    x_centers = np.asarray([12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5, 275.0, 325.0, 375.0, 425.0,
                            475.0, 525.0, 575.0, 625.0, 675.0, 725.0, 787.9, 883.2, 1028.0, 1247.0, 1578.0, 2081.0,
                            2843.0, 3998.0, 5749.0, 8402.0])

    z_centers = np.asarray([2.5, 7.5, 12.5, 17.5, 22.5, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
                            132.5, 147.5, 162.5, 177.5, 192.5, 207.5, 222.5, 237.5, 252.5, 267.5, 282.5, 297.5, 312.5,
                            327.5, 342.5, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 562.0,
                            587.5, 612.5, 637.5, 662.5, 687.5, 712.5, 737.5, 762.5, 787.5, 812.5, 837.5, 862.5, 887.5,
                            912.5, 937.5, 962.5, 987.5, 1013.0, 1038.0, 1060.0, 1080.0, 1100.0, 1120.0, 1140.0, 1160.0,
                            1180.0, 1200.0, 1220.0, 1240.0, 1260.0, 1280.0, 1300.0, 1320.0, 1340.0, 1360.0, 1380.0,
                            1400.0, 1418.0, 1433.0, 1448.0, 1463.0, 1475.0, 1485.0, 1495.0])

    x_thick = get_thicknesses(x_centers, x_init=0.0, x_end=10000.0)
    x_rad = np.insert(np.asarray(list(itertools.accumulate(x_thick))), 0, 0.0)
    z_thick = get_thicknesses(z_centers, x_init=0.0, x_end=1500.0)

    # add cells at the top for boundary conditions
    z_thick = np.insert(z_thick, len(z_thick), 0.1)

    parameters = {
        'meshmaker': {
            'type': 'rz2dl',
            'parameters': [
                {'type': 'radii', 'radii': list(x_rad)},
                {'type': 'layer', 'thicknesses': list(z_thick)}
            ]
        }
    }

    mesh = toughio.meshmaker.from_meshmaker(parameters, material='HROCK')

    # the mesh has been defined in the XY plane, but the points have 3D coordinates (with zeros as 3rd dimension for every cells).
    # to make it 3D in the XZ, we swap the 2nd and 3rd dimension, and then extrude the mesh by 1 meter along the Y axis (2nd dimesion).
    # mesh.points[:, [1, 2]] = mesh.points[:, [2, 1]]
    # mesh.extrude_to_3d(height=1.0, axis=1)

    # Alternatively, without needing to chenge anything in _cylindric_grid.py (keeping origin at 0.0 for z and structured grid in three dimensions)
    # mesh.points[:, 2] = mesh.points[:, 2] + 1000
    mesh.points[:, 2] = mesh.points[:, 2]

    # Now, find out the largest volume elements and distiguish for setting boundary conditions at the top of the mesh
    # ind1 = np.where(mesh.centers[:, 2] < 0.0)[0]
    ind2 = np.where(mesh.centers[:, 2] >= -2.0)[0]
    materials_new = np.ones((len(mesh.materials)))
    materials_new[ind2] = 2.0
    # overwrite existing material names ...
    mesh.add_cell_data('material', materials_new.astype(int))
    # rename materials
    mesh.add_material('HROCK', 1)
    mesh.add_material('BOUND', 2)

    ####################################################################################################################
    # # uncomment block for plotting, comment to save MESH
    # # re-create cell data
    # cell_data = []
    # cell_data.append(mesh.cell_data['material'])
    # mesh.cell_data['material'] = cell_data
    #
    # # Plotting 2D mesh
    # import pyvista
    # pyvista_mesh = pyvista.from_meshio(mesh)
    #
    # # optionally clip the mesh along the x-axis to reduce from 10 km to 1 km ...
    # x_cut = 1000.0
    # clipped_mesh = pyvista_mesh.clip(invert=True, normal=[1, 0, 0], origin=[x_cut, 0, 0])
    # # and cut the large boundary cells from the plot ...
    # z_cut = 0.0
    # clipped_mesh = clipped_mesh.clip(invert=True, normal=[0, 0, 1], origin=[0, 0, z_cut])
    #
    # pyvista.set_plot_theme('dark')
    #
    # p = pyvista.Plotter(window_size=(500, 1200), off_screen=True)
    # p.add_mesh(mesh=clipped_mesh,
    #            scalar_bar_args={'title': 'Materials'},
    #            show_scalar_bar=True,
    #            show_edges=True,
    # )
    # p.show_axes()
    # p.view_xz()
    # p.camera.zoom(1.8)
    # p.screenshot('mesh.png')
    # p.show()
    # p.close()

    ####################################################################################################################

    # print the number of elements of the computational domain
    print('Number of elements: ' + str(len(mesh.centers)))

    # Now we specify the boundary conditions, since toughio can recognize the cell data key 'boundary_condition' and automatically
    # imposes Dirichlet boundary conditions (i.e., large volume assigned to this cells) to cells that have any value other than 0 in this cell data array
    materials = mesh.materials
    bcond = (materials == 'BOUND').astype(int)
    mesh.add_cell_data('boundary_condition', bcond)

    # initial conditions (single-phase)
    # get the last index of the boundary
    bound_idx = np.where(mesh.materials=='BOUND')[0][-1] + 1
    incon = np.full((mesh.n_cells, 3), -1.0e9)
    incon[:bound_idx, 0] = 1.0132e5                                         # atmospheric conditions for pressure
    incon[bound_idx:, 0] = 1.0e5 - 997 * 9.81 * mesh.centers[bound_idx:, 2]       # hydrostatic pore pressure
    # incon[bound_idx:, 0] = 1.0132e5
    incon[:bound_idx, 1] = 21.0                                          # temperature at boundary conditions
    # incon[bound_idx:, 1] = 21.0 - 0.0267 * mesh.centers[bound_idx:, 2]            # some temperature gradient
    incon[bound_idx:, 1] = 21.0 - 0.106 * mesh.centers[bound_idx:, 2]  # some temperature gradient
    # incon[bound_idx:, 1] = 180.0
    incon[:bound_idx, 2] = 35.0                                           # co2 partial pressure at the boundary
    incon[bound_idx:, 2] = 35.0                                           # co2 partial pressure at the rock
    mesh.add_cell_data('initial_condition', incon)

    # initial conditions (same as below)
    # incon = np.full((mesh.n_cells, 3), -1.0e9)
    # incon[:, 0] = 1.0e5
    # incon[:, 1] = 0.0
    # incon[:, 2] = 1.0e4
    # mesh.add_cell_data('initial_condition', incon)

    # mesh.write_tough('MESH', incon=True)
    mesh.write_tough('MESH', incon=False)
    mesh.write('mesh.pickle')

########################################################################################################################

# A :mod: 'toughio' input file is defined as a nested dictionary with meaningful keywords.
# Let's initialize our parameters dictionary by giving the simulation a title and defining the equation-of-state.

parameters = {
    'title': 'Heat and fluid flow simulation from magmatic fluid intrusion at depth',
    'eos': 'eos2',      # eos2 to inject co2 and water
    'n_component': 2,
    'n_phase': 2,
    'start': True,
    'isothermal': False,
}

# Define properties of the different materials (block 'ROCKS').
if meshmaker=='pygmsh':
    parameters['rocks'] = {
        'HROCK': {
            'density': 2000.0,
            'conductivity': 2.8,
            'specific_heat': 1000.0,
            'porosity': 0.2,
            'permeability': 1.0e-14,
            'initial_condition': [1.0132e5, 180.0, 35.0],                   # pressure, temperature, CO2 partial pressure (if zero, only water is present!)
        },
        'BOUND': {                  # atmospheric conditions (air at approx. 300 K)
            'density': 2000.0,
            'conductivity': 2.8,
            'specific_heat': 1.0e55,
            'porosity': 0.2,
            'permeability': 1.0e-13,
            'initial_condition': [1.0132e5, 80.0, 35.0],                    # pressure, temperature, CO2 partial pressure (if zero, only water is present!)
        }
    }
elif meshmaker=='meshmaker':
    # set constant (throughout the whole mesh) initial conditions
    parameters['default'] = {
        'density': 2000.0,
        'porosity': 0.2,
        'conductivity': 2.8,
        'conductivity_dry': 2.8,
        'tortuosity': 0.7,
        # 'relative_permeability': {
        #     'id': 1,
        #     'parameters': [0.0, 0.0, 1.0, 1.0],
        # },
        'relative_permeability': {
            'id': 3,
            'parameters': [0.3, 0.05],
        },
        # 'capillarity': {
        #     'id': 6,
        #     'parameters': [5000.0, 0.4],
        # },
        'capillarity': {
            'id': 1,
            'parameters': [0.0, 0.0, 1.0],
        },
        'initial_condition': [1.0132e5, 180.0, 35.0], # pressure, temperature, CO2 partial pressure (if zero, only water is present!)
    }

    parameters['rocks'] = {
        'HROCK': {
            'permeability': 1.0e-14,
            'specific_heat': 1000.0,
            # 'initial_condition': [1.0132e5, 180.0, 35.0], # pressure, temperature, CO2 partial pressure (if zero, only water is present!)
        },
        'BOUND': {
            'permeability': 1.0e-13,
            'specific_heat': 1.0e55,
            'initial_condition': [1.0132e5, 21.0, 35.0], # pressure, temperature, CO2 partial pressure (if zero, only water is present!)
        }
    }

# We can specify some simulation parameters (block 'PARAM'), options ('MOP') and selections (block 'SELEC')
parameters['options'] = {
    'n_cycle': 9999,                                                            # MCYC: maximum number of time steps to be calculated
    'n_cycle_print': 9999,                                                      # MCYPR: printout for every multiple of this number
    't_ini': 0.0,                                                               # TSTART: starting time of the simulation in seconds
    't_max': 10000.0 * 365.25 * 24.0 * 3600.0,                                    # TIMAX: time in seconds at which simulation stops (here thousand years)
    't_steps': 100.0 * 365.25 * 24.0 * 3600.0,                                   # DELTEN: length of the time steps in seconds (here yearly)
    't_step_max': 1000.0 * 365.25 * 24.0 * 3600.0,                                # DELTMX: upper limit for time step size in seconds
    't_reduce_factor': 4,                                                     # REDLT: factor by which time step is reduced in case of convergence failure or other problems
    'eps1': 1.0e-5,                                                             # DLT(I): Length (in seconds) of time step I
    # 'eps2': 1.0e-2,                                                             # DLT(II): Length (in seconds) of time step II
    'gravity': 9.81,                                                            # GF: magnitude of the gravitational acceleration vector
}
# MOP: select choice of various options, which are documented in printed output from a TOUGH3 run
parameters['extra_options'] = {
    1: 1,                                                                       # short printout for non-convergent iterations
    7: 1,                                                                       # printout of input data provided
    13: 1,                                                                      # writes user-specified initial conditions to file SAVE
    16: 4,                                                                      # time step size will be doubled if convergence occurs within ITER <= MOP(16) Newton-Raphson iterations
    21: 7,                                                                      # 7 for AZTEC, 8 for PETSc parallel iterative solver
}

# SELEC: optional (not used for now)

# We have to define the generators (i.e., the source)
# However, we first need to know the name of the cell in which CO2 and H2O enters the mesh.
# In this case, as the injection extent is of 150 m
# Therefore, we either inject in more than one cell or we constrain the injected into the first cell ...
inj_opt = '2' # 1: for injecting into all cells in the 150 m or 2: for injecting only on one cell

# Now we can add the generators to the parameters by specifying the type and injection rate (block 'GENER')
parameters['generators'] = []

inj_max = 150.0
comp = ['COM1', 'COM2']
com1_inj = 2400.0 * 1000.0 / (24.0 * 60.0 * 60.0)       # injected component 1 (water)
com2_inj = 1000.0 * 1000.0 / (24.0 * 60.0 * 60.0)       # injected component 2 (co2)
enthalpy = [3.06e6, 7.12e5]                             # enthalpies

if inj_opt=='2':
    # We need to look for all cells and distribute the injection rate according to their respective volumetric contributions to the domain.
    # Let's unpickle the mesh back and use the method :meth: 'toughio.Mesh.near' to get the name of the injection element.
    mesh = toughio.read_mesh('mesh.pickle')
    label_1 = mesh.labels[mesh.near((0.0, 0.0, -1500.0))]
    ind_1 = int(np.where(mesh.labels == label_1)[0])
    label_2 = mesh.labels[mesh.near((inj_max, 0.0, -1500.0))]
    ind_2 = int(np.where(mesh.labels == label_2)[0]) + 1  # add one indice more
    # get lists/arrays of all generators
    label_all = mesh.labels[ind_1:ind_2]
    # way 1 #
    vols = mesh.volumes[ind_1:ind_2] / np.sum(mesh.volumes[ind_1:ind_2])
    # introduce an error if sum(vols)!=1.0 #
    # injection rates
    rates_all = [com1_inj * vols, com2_inj * vols]

    for i in range(len(comp)):
        for j in range(len(label_all)):
            parameters['generators'].append(
                {
                    'label': label_all[j],
                    'type': comp[i],
                    'rates': rates_all[i][j],
                    'specific_enthalpy': enthalpy[i],
                }
            )

else:
    mesh = toughio.read_mesh('mesh.pickle')
    label = mesh.labels[mesh.near((0.0, 0.0, -1500.0))]
    V = inj_max**2 * np.pi * z_thick[0]
    rates_all = [com1_inj, com2_inj] / V

    for i in range(len(comp)):
        parameters['generators'].append(
            {
                'label': label,
                'type': comp[i],
                'rates': rates_all[i],
                'specific_enthalpy': enthalpy[i],
            }
        )

# Let's now customize the outputs.
# For this example, we want TOUGH to save the output every 10 years.
# We want to save the volumetric gas fraction and the temperature, as well as the pore pressure at those times
# years_range = [100.0, 1000.0, 10000.0, 25000.0, 50000.0, 100000.0]
years_range = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 7500.0, 10000.0]
# years_range = [1.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0]
# years_range = [0.0, 25.0, 50.0, 75.0, 100.0]
# years_range = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
y = 365.25 * 24.0 * 3600.0
parameters['times'] = [x * y for x in years_range]

# Print only the following listed variables ...
parameters['output'] = {
    'variables': [
        {'name': 'saturation'},
        {'name': 'temperature'},
        {'name': 'pressure'},
        {'name': 'coordinate'},
    ]
}

# Finally, we can export the model parameters input file by using the function :func: 'toughio.write_input'
toughio.write_input('INFILE', parameters)