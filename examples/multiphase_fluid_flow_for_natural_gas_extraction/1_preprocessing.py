"""
Preprocessing of:

"Multiphase fluid flow for natural gas extraction" from Zbinden et al (2017)

"""

import numpy as np
import pygmsh
import toughio
import matplotlib.pyplot as plt

# MESH ################################################################

lc = 100.0                          # Characteristic length of the mesh
xmin, xmax = -1000.0, 10000.0       # X axis boundaries
zmin, zmax = -2000.0, -4000.0       # Z axis boundaries

inj_z = -3000.0                     # Depth of injection
flt_offset = 130.0                  # Offset of fault
flt_core_thick = 2.5                # Thickness of fault core
dz_l_thick = 3.75                   # Thickness of left damage zone
dz_r_thick = 3.75                   # Thickness of right damage zone
tana = np.tan(np.deg2rad(80.0))     # Tangeant of dipping angle of fault

dist = 1500.0 - (0.5 * flt_core_thick + dz_l_thick)     # distance of fault (left damage zone boundary) to the injection point

bnd_thick = 10.0                    # Thickness of boundary elements

# Fault (damage zone + fault core)

# left damage zone
depths_left = [zmin - 150.0, -2800.0, -2950.0, -3050.0, zmax + 150.0]
dz_left = [[xmin + dist + (z - inj_z) / tana, z, 0.0] for z in depths_left]

depths_right = [zmin - 150.0, zmax + 150.0]
dz_right = [[xmin + dist + (z - inj_z) / tana + dz_l_thick, z, 0.0] for z in depths_right]

dz_l_pts = dz_left + dz_right[::-1]

# right damage zone
depths_left = [zmin - 150.0, zmax + 150.0]
dz_left = [[xmin + dist + (z - inj_z) / tana + dz_l_thick + flt_core_thick, z, 0.0] for z in depths_left]

depths_right = [zmin - 150.0, -2800.0 + flt_offset, -2950.0 + flt_offset, -3050.0 + flt_offset, zmax + 150.0]
dz_right = [[xmin + dist + (z -inj_z) / tana + dz_l_thick + flt_core_thick + dz_r_thick, z, 0.0] for z in depths_right]

dz_r_pts = dz_left + dz_right[::-1]

# fault core
depths_left = [zmin - 150.0, zmax + 150.0]
fc_left = [[xmin + dist + (z - inj_z) / tana + dz_l_thick, z, 0.0] for z in depths_left]

depths_right = [zmin - 150.0, zmax + 150.0]
fc_right = [[xmin + dist + (z - inj_z) / tana + dz_l_thick + flt_core_thick, z, 0.0] for z in depths_right]

fc_pts = fc_left + fc_right[::-1]

# Different layers #

# upper aquifer
upper_aquifer_pts = [
    [xmin, zmin, 0.0],
    [xmin, zmin - 800.0, 0.0],
    [xmin + dist + ((zmin - 800.0) - inj_z) / tana, zmin - 800.0, 0.0],
    [xmin + dist + ((zmin - 150.0) - inj_z) / tana, zmin - 150.0, 0.0],
    [xmin + dist + ((zmin - 150.0) - inj_z) / tana + dz_l_thick, zmin - 150.0, 0.0],
    [xmin + dist + ((zmin - 150.0) - inj_z) / tana + dz_l_thick + flt_core_thick, zmin - 150.0, 0.0],
    [xmin + dist + ((zmin - 150.0) - inj_z) / tana + dz_l_thick + flt_core_thick + dz_r_thick, zmin - 150.0, 0.0],
    [xmin + dist + ((zmin - 800.0 + flt_offset) - inj_z) / tana + dz_l_thick + flt_core_thick + dz_r_thick, zmin - 800.0 + flt_offset, 0.0],
    [xmax, zmin - 800.0 + flt_offset, 0.0],
    [xmax, zmin, 0.0],
]
caprock_left_pts = [
    [xmin, zmin - 800.0, 0.0],
    [xmin, zmin - 950.0, 0.0],
    [xmin + dist + ((zmin - 950.0) - inj_z) / tana, zmin - 950.0, 0.0],
    [xmin + dist + ((zmin - 800.0) - inj_z) / tana, zmin - 800.0, 0.0],
]
reservoir_left_pts = [
    [xmin, zmin - 950.0, 0.0],
    [xmin, zmin - 1050.0, 0.0],
    [xmin + dist + ((zmin - 1050.0) - inj_z) / tana, zmin - 1050.0, 0.0],
    [xmin + dist + ((zmin - 950.0) - inj_z) / tana, zmin - 950.0, 0.0],
]
base_rock_pts = [
    [xmin, zmin - 1050.0, 0.0],
    [xmin, zmax, 0.0],
    [xmax, zmax, 0.0],
    [xmax, zmin - 1050.0 + flt_offset, 0.0],
    [xmin + dist + ((zmin - 1050.0 + flt_offset) - inj_z) / tana + dz_l_thick + flt_core_thick + dz_r_thick, zmin - 1050.0 + flt_offset, 0.0],
    [xmin + dist + ((zmax + 150.0) - inj_z) / tana + dz_l_thick + flt_core_thick + dz_r_thick, zmax + 150.0, 0.0],
    [xmin + dist + ((zmax + 150.0) - inj_z) / tana + dz_l_thick + flt_core_thick, zmax + 150.0, 0.0],
    [xmin + dist + ((zmax + 150.0) - inj_z) / tana + dz_l_thick, zmax + 150.0, 0.0],
    [xmin + dist + ((zmax + 150.0) - inj_z) / tana, zmax + 150.0, 0.0],
    [xmin + dist + ((zmin - 1050.0) - inj_z) / tana, zmin - 1050.0, 0.0],
]
reservoir_right_pts = [
    [xmin + dist + ((zmin - 950.0 + flt_offset) - inj_z) / tana + dz_l_thick + flt_core_thick + dz_r_thick, zmin - 950.0 + flt_offset, 0.0],
    [xmin + dist + ((zmin - 1050.0 + flt_offset) - inj_z) / tana + dz_l_thick + flt_core_thick + dz_r_thick, zmin - 1050.0 + flt_offset, 0.0],
    [xmax, zmin - 1050.0 + flt_offset, 0.0],
    [xmax, zmin - 950.0 +flt_offset, 0.0],
]
caprock_right_pts = [
    [xmin + dist + ((zmin - 800.0 + flt_offset) - inj_z) / tana + dz_l_thick + flt_core_thick + dz_r_thick, zmin - 800.0 + flt_offset, 0.0],
    [xmin + dist + ((zmin - 950.0 + flt_offset) - inj_z) / tana + dz_l_thick + flt_core_thick + dz_r_thick, zmin - 950.0 + flt_offset, 0.0],
    [xmax, zmin - 950.0 + flt_offset, 0.0],
    [xmax, zmin - 800.0 + flt_offset, 0.0],
]

# boundaries
bound_top_pts = [
    [xmin, zmin + bnd_thick, 0.0],
    [xmin, zmin, 0.0],
    [xmax, zmin, 0.0],
    [xmax, zmin + bnd_thick, 0.0],
]
bound_right_pts = [
    [xmax, zmin, 0.0],
    [xmax, zmax, 0.0],
    [xmax + bnd_thick, zmax, 0.0],
    [xmax + bnd_thick, zmin, 0.0],
]
bound_bottom_pts = [
    [xmin, zmax, 0.0],
    [xmin, zmax + bnd_thick, 0.0],
    [xmax, zmax + bnd_thick, 0.0],
    [xmax, zmax, 0.0]
]

# create geometry #

with pygmsh.geo.Geometry() as geo:
    # Define polygons
    fault_core = geo.add_polygon(fc_pts, mesh_size=0.1*lc)
    damage_zone_left = geo.add_polygon(dz_l_pts, mesh_size=0.15*lc)
    damage_zone_right = geo.add_polygon(dz_r_pts, mesh_size=0.15*lc)
    reservoir_left = geo.add_polygon(reservoir_left_pts, mesh_size=0.2*lc)
    reservoir_right = geo.add_polygon(reservoir_right_pts, mesh_size=1.0*lc)
    caprock_left = geo.add_polygon(caprock_left_pts, mesh_size=0.3*lc)
    caprock_right = geo.add_polygon(caprock_right_pts, mesh_size=1.0*lc)
    upper_aquifer = geo.add_polygon(upper_aquifer_pts, mesh_size=1.0*lc)
    base_rock = geo.add_polygon(base_rock_pts, mesh_size=1.0*lc)
    bound_top = geo.add_polygon(bound_top_pts, mesh_size=2.0*lc)
    bound_right = geo.add_polygon(bound_right_pts, mesh_size=2.0*lc)
    bound_bottom = geo.add_polygon(bound_bottom_pts, mesh_size=2.0*lc)

    # Define materials
    geo.add_physical(fault_core, 'FACOR')
    geo.add_physical([damage_zone_left, damage_zone_right], 'DAZON')
    geo.add_physical([reservoir_left, reservoir_right], 'RESER')
    geo.add_physical([caprock_left, caprock_right], 'CAPRO')
    geo.add_physical(upper_aquifer, 'UPAQU')
    geo.add_physical(base_rock, 'BROCK')
    geo.add_physical([bound_top, bound_right, bound_bottom], 'BOUND')

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
    mesh.cell_data["material"] = cell_data
    mesh.field_data.update(field_data)
    mesh.cell_sets = {}

    # Remove lower dimension entities
    idx = [i for i, cell in enumerate(mesh.cells) if cell.type == "triangle"]
    mesh.cells = [mesh.cells[i] for i in idx]
    mesh.cell_data = {k: [v[i] for i in idx] for k, v in mesh.cell_data.items()}

    # Export the mesh for post-processing
    mesh.write("mesh.vtu")

# visualize mesh #

# import pyvista
# pyvista.set_plot_theme("document")
#
# p = pyvista.Plotter(window_size=(800, 800))
# p.add_mesh(
#     mesh=pyvista.from_meshio(mesh),
#     scalar_bar_args={"title": "Materials"},
#     show_scalar_bar=True,
#     show_edges=True,
# )
# p.view_xy()
# p.show()

# check goodness of mesh #
del mesh
mesh = toughio.read_mesh('mesh.vtu')

mesh.add_material('FACOR', 1)       # fault core
mesh.add_material('DAZON', 2)       # damage zone
mesh.add_material('RESER', 3)       # reservoir
mesh.add_material('CAPRO', 4)       # caprock
mesh.add_material('UPAQU', 5)       # upper aquifer
mesh.add_material('BROCK', 6)       # basal aquifer
mesh.add_material('BOUND', 7)       # boundary

mesh.points[:, [1, 2]] = mesh.points[:, [2, 1]]
mesh.extrude_to_3d(height=1.0, axis=1)

print('Number of elements: ' + str(len(mesh.centers)))

# import pyvista
# pyvista.set_plot_theme("document")
#
# p = pyvista.Plotter(window_size=(800, 800))
# p.add_mesh(
#     mesh=mesh.to_pyvista(),
#     scalars=mesh.qualities,
#     scalar_bar_args={"title": "Average cell quality"},
#     clim=(0.0, 1.0),
#     cmap="RdBu",
#     show_scalar_bar=True,
#     show_edges=True,
# )
# p.view_xz()
# p.show()

# define boundary conditions #
materials = mesh.materials
bcond = (materials == 'BOUND').astype(int)
mesh.add_cell_data('boundary_condition', bcond)

# define initial conditions #
centers = mesh.centers
incon = np.full((mesh.n_cells, 3), -1.0e9)
incon[:, 0] = 1.0e5 - 9810 * centers[:, 2]                          # gas phase pressure (not known beforehand?) or do we assume hydrostatic pressure gradient?
# gas saturation (initial conditions per material)
material = ['FACOR', 'DAZON', 'RESER', 'CAPRO', 'UPAQU', 'BROCK', 'BOUND']
g_sat = [-1.0e9, -1.0e9, 0.90, 1.0e-3, 1.0e-3, 1.0e-6, 0.0]
for i in range(len(material)):
    idx = np.where(mesh.materials==material[i])[0]
    incon[idx, 1] = g_sat[i]
# incon[:, 1] = 0.05                                                  # gas saturation + 10
incon[:, 2] = 60.0 - 0.025 * centers[:, 2]                          # temperature
mesh.add_cell_data('initial_condition', incon)

mesh.write_tough('MESH', incon=True)
mesh.write('mesh.pickle')

# generating model parameters #
parameters = {
    'title': 'Simulation of multiphase fluid flow for natural gas extraction',
    'eos': 'eos3',          # EOS module for water and air (in this case CO2)
    'n_component': 2,       # components: Water and CO2
    'n_phase': 2,           # phases: liquid and gaseous
    'isothermal': True,     # for all simulation isothermal conditions because the temperature variations are negligible
    'start': True,
}

# Default values shared by the different materials
parameters['default'] = {
    'density': 2260.0,
    'conductivity': 1.8,
    'specific_heat': 1500.0,
    'conductivity_dry': 1.8,
    'tortuosity': 0.7,
    'relative_permeability': {
        'id': 3,
        'parameters': [0.3, 0.05],
    },
    'capillarity': {
        'id': 7,
        'parameters': [0.457, 0.0, 5.025e-5, 5.0e7, 0.99],
    },
}

# Properties of the different materials
parameters['rocks'] = {
    'FACOR': {
        'porosity': 0.1,
        'permeability': 1.0e-19,
    },
    'DAZON': {
        'porosity': 0.1,
        'permeability': 1.0e-17,
    },
    'RESER': {
        'porosity': 0.16,
        'permeability': 1.0e-13,
    },
    'CAPRO': {
        'porosity': 0.01,
        'permeability': 1.0e-21,
        'capillarity': {
            'id': 7,
            'parameters': [0.457, 0.0, 1.61e-6, 5.0e7, 0.99],
        },
    },
    'UPAQU': {
        'porosity': 0.1,
        'permeability': 1.0e-14,
    },
    'BROCK': {
        'porosity': 0.01,
        'permeability': 1.0e-13,
        'capillarity': {
            'id': 7,
            'parameters': [0.457, 0.0, 1.61e-6, 5.0e7, 0.99],
        },
    },
    'BOUND': {
        'specific_heat': 1.0e55,
        'porosity': 0.1,
        'permeability': 1.0e-13,
    },
}

# Simulation parameters
parameters['options'] = {
    'n_cycle': 9999,
    'n_cycle_print': 9999,
    't_ini': 0.0,                               # start at time zero
    # 't_max': 95.0 * 365.25 * 24.0 * 3600.0,     # simulation for a total of 95 years
    't_max': 10.0 * 365.25 * 24.0 * 3600.0,   # simulation for a total of 1000 years
    't_steps': 1.0 * 24.0 * 3600.0,             # length of the time step to 1 day
    # 't_step_max': 1.0 * 365.25 * 24.0 * 3600.0, # maximum length of the time step to 1 year
    't_step_max': 1.0 * 365.25 * 24.0 * 3600.0,     # maximum length of the time step to 10 years
    't_reduce_factor': 10,                      # time step reducing factor
    'eps1': 1.0e-6,
    'eps2': 1.0e-4,
    'gravity': 9.81,
}

# Additional parameters (MOP(I))
parameters['extra_options'] = {
    1: 1,                                       # short printout for non-convergent iterations
    7: 1,                                       # printout of input data provided
    13: 0,                                      # 0 standard content and 1 for writing user-specified initial conditions to file SAVE
    16: 4,                                      # time step size will be doubled if convergence occurs within ITER <= MOP(16) Newton-Raphson iterations
    17: 9,                                      # reducing time step after linear equation solver failure
    19: 1,                                      # by setting MOP(19) = 1, initialization can be made with TOUGH-style variables (P, T, X) for single-phase, (Pg, Sg, T) for two-phase
    21: 0,                                      # Lanczos-type preconditioned bi-conjugate gradient solver
}

# # Define the generator (i.e., the source)
# mesh = toughio.read_mesh('mesh.pickle')
# label = mesh.labels[mesh.near((-1000.0, 0.0, -3000.0))]
#
# # Set the generator to the parameters by specifying the type and injection rate
# parameters['generators'] = [
#     {
#         'label': label,
#         'type': 'COM2',
#         'rates': 3.0e-3,   # constant production rate
#     },
# ]

# Customize the outputs times
# years_range = [3.2, 15.9, 34.9, 50.8]
# years_range = [1.0, 10.0, 100.0, 1000.0]
years_range = [1.0, 10.0]
y = 365.25 * 24.0 * 3600.0
parameters['times'] = [x * y for x in years_range]

# Export the model parameters input file
toughio.write_input('INFILE', parameters)