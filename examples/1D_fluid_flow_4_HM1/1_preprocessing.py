"""
Preprocessing of:

1D fluid flow solution with pressure diffusion

HM1 template with TOUGH3

"""

########################################################################################################################
# Import packages

import numpy as np
import pygmsh
import pickle
import toughio
import matplotlib.pyplot as plt

########################################################################################################################

# 1) CREATE 1-D (CYLINDRICAL) AXISYMMETRIC MESH

parameters = {
    'meshmaker': {
        'type': 'rz2d',
        'parameters': [
            {'type': 'radii', 'radii': [0.0]},
            {'type': 'equid', 'n_increment': 1, 'size': 0.1},
            {'type': 'logar', 'n_increment': 99, 'radius': 1000.0},
            {'type': 'layer', 'thicknesses': [8]}, # this is the interval thickness
        ]
    }
}

mesh = toughio.meshmaker.from_meshmaker(parameters, material='BGRAN')
mesh.write_tough('MESH', incon=False)
mesh.write('mesh.pickle')

# 2) CREATE INPUT FILE

parameters = {
    'title': '1D fluid flow solution - HM1 test for given injection protocol',
    'eos': 'eos1',
    'n_component': 1,       # only one water component
    'n_phase': 2,           # only liquid phase
    'start': True,
    'isothermal': True,     # constant temperature conditions
    'do_diffusion': False,   # enabling molecular diffusion # but NK=2 required, so for this example is False
}

# Bedretto granite properties/parameters
parameters['rocks'] = {
    'BGRAN': {
        'density': 2620.0,              # bulk density (in kg/m³)
        'porosity': 0.1,                # 1.75% porosity on average
        'permeability': 5.0e-15,        # 1.13 micro Darcy (in m²)
        'compressibility': 1.e-12,      # compressibility in 1/Pa # should be equal to the product of porosity and storage
    },
}

# set up initial conditions
parameters['default'] = {
    'initial_condition': [5.6 * (1.e6), 22.0],            # total pressure, temperature (single-phase)
}
# total pressure = hydrostatic pressure, only if fully saturated with water
# Kai's ARMA paper stated from 2.0 MPa to 5.6 MPa, which is below the hydrostatic pressure,
# due to the strong tunnel drainage and pressure drawdown ...
# initial interval temperature: 21.5°C-22.0°C

parameters['options'] = {
    'n_cycle': 9999,                    # maximum number of time steps
    'n_cycle_print': 4,              # printout for every multiple of this number
    't_ini': 0.0,                       # starting time of the simulation
    't_max': 1230.0 * 60.0,             # time in seconds at which the simulation stops
    't_steps': 0.5 * 60.0,              # length of the time step in seconds
    't_step_max': 1.0 * 60.0,           # upper limit for the time step size in seconds
    't_reduce_factor': 2,               # time step reducing factor
    'eps1': 1.0e-8,                     # length in seconds of first time step
    'eps2': 1.0e-6,                     # length in seconds of second time step
    'gravity': 9.81,
}

parameters['extra_options'] = {
    1: 1,
    7: 1,
    12: 2,                              # 2: step rate capability for time dependent generation data
    13: 1,
    16: 4,
    21: 8,
}

# 3) DEFINE THE GENERATORS
mesh = toughio.read_mesh('mesh.pickle')
label = mesh.labels[mesh.near((0.0, 0.0, 0.0))]

uc1 = 60 # unit corrector from minutes to seconds
# uc2 = 1 / (1000 * 60) # unit corrector from L/min to m³/s # this is wrong since the input has to be in kg/s ...
uc2 = 1 / 60

parameters['generators'] = [
    {
        'label': label,
        'type': 'COM1',
        'times': [0.0 * uc1, 120.0 * uc1, 240.0 * uc1, 360.0 * uc1, 480.0 * uc1, 600.0 * uc1, 720.0 * uc1, 1230.0 * uc1], # optionally the last time can be infinite, as long as I stop the simulation at the right time, otherwise it runs forever
        'rates': [(1.e-1) * uc2, 15.0 * uc2, 30.0 * uc2, 45.0 * uc2, 60.0 * uc2, 75.0 * uc2, (1.e-1) * uc2, 0.0 * uc2],
        # 'times': [0.0 * uc1, 120.0 * uc1, 720.0 * uc1, 721.0 * uc1, 1230.0 * uc1],
        # 'rates': [(1.e-1) * uc2, (1.e-1) * uc2, 75.0 * uc2, (1.e-1) * uc2, (1.e-1) * uc2]
    },
]

# save injection protocol for plotting later in postprocessing
path_folder = '/home/victor/Desktop/toughio/examples/1D_fluid_flow_4_HM1/'
filename = 'inj_protocol.pkl'
with open(path_folder + filename, 'wb') as f:
    pickle.dump([parameters['generators'][0]['times'], parameters['generators'][0]['rates']], f)

# 4) CUSTOMIZE THE OUPUT
# generate pressure solution output every x minutes
# times_array = np.linspace(0.0, 1230.0, 616)
# y = 60.0 # from minutes to seconds
# parameters['times'] = [x * y for x in times_array]

# variables to print
parameters['output'] = {
    'variables': [
        {'name': 'coordinate'},
        {'name': 'generation'},
        {'name': 'pressure'},
        {'name': 'absolute', 'options': 1},
    ],
}

# 5) WRITE THE INFILE
toughio.write_input('INFILE', parameters)