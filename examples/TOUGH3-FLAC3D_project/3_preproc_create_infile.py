"""
Script to create both INFILES for the steady state and injection (mode to choose below)
@ by Victor Clasen Repollés (victor.clasen@sed.ethz.ch)
"""
import datetime

# Import the required libraries
import toughio
import numpy as np
import pandas as pd

import utils

#%% get CELL_IDs for injection cell and pressure monitoring points (sensors at MB2, MB5, and MB8)

PS_MB2 = 123.6          # sensor depth at MB2
PS_MB4 = 97.8           # no sensor but position closest to the fault core (roughly)
PS_MB5 = 106.218        # sensor depth at MB5
PS_MB8 = 123.85         # sensor depth at MB8

project_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/'

# injection cell (as above, repeat if commented)
# mesh_dict = toughio.read_mesh(project_path + '0_preprocessing_OUTPUT/MESH_bound') # coarser mesh
mesh_dict = toughio.read_mesh(project_path + '0_preprocessing_OUTPUT/fine_mesh_3110/MESH') # finer mesh

cell_ids = np.asarray(list(mesh_dict['elements'].keys()))

vals = []
for i in range(len(cell_ids)):
    vals.append(np.sum(np.square(mesh_dict['elements'][cell_ids[i]]['center'])))
inj_ind = np.asarray(vals).argmin()
inj_cell = cell_ids[inj_ind]

print('Injection cell ----> ' + str(inj_cell) + ': ' + str(mesh_dict['elements'][inj_cell]['center']))

# MB2
MB2_df = pd.read_csv(project_path + '0_boreholes/finer_mesh_3.0/' + 'MB2_borehole_information_edited.csv', index_col=0)
idx1 = np.argmin(np.abs(np.array(MB2_df['Depth (m)']) - PS_MB2))
MB2_cell_id = MB2_df['cell_id'][idx1]

print('Monitoring cell (along MB2) ----> ' + str(MB2_cell_id) + ': ' + str(mesh_dict['elements'][MB2_cell_id]['center']))

if MB2_df['domain'][idx1]=='ROCKS':
    print('!!!! MB2 monitoring cell outside fault core and damage zone ... getting cell at fault core (FZONE) !!!!')
    idx11 = np.where(MB2_df['domain']=='FZONE')[0][0]
    MB2_cell_id = MB2_df['cell_id'][idx11]
    print('NEW monitoring cell (along MB2) ----> ' + str(MB2_cell_id) + ': ' + str(mesh_dict['elements'][MB2_cell_id]['center']))

# MB5
MB5_df = pd.read_csv(project_path + '0_boreholes/finer_mesh_3.0/' + 'MB5_borehole_information_edited.csv', index_col=0)
idx2 = np.argmin(np.abs(np.array(MB2_df['Depth (m)']) - PS_MB5))
MB5_cell_id = MB5_df['cell_id'][idx2]

print('Monitoring cell (along MB5) ----> ' + str(MB5_cell_id) + ': ' + str(mesh_dict['elements'][MB5_cell_id]['center']))

if MB5_df['domain'][idx2]=='ROCKS':
    print('!!!! MB5 monitoring cell outside fault core and damage zone ... getting cell at fault core (FZONE) !!!!')
    idx22 = np.where(MB5_df['domain']=='FZONE')[0][0]
    MB5_cell_id = MB5_df['cell_id'][idx22]
    print('NEW monitoring cell (along MB5) ----> ' + str(MB5_cell_id) + ': ' + str(mesh_dict['elements'][MB5_cell_id]['center']))

# MB8
MB8_df = pd.read_csv(project_path + '0_boreholes/finer_mesh_3.0/' + 'MB8_borehole_information_edited.csv', index_col=0)
idx3 = np.argmin(np.abs(np.array(MB8_df['Depth (m)']) - PS_MB8))
MB8_cell_id = MB8_df['cell_id'][idx3]

print('Monitoring cell (along MB8) ----> ' + str(MB8_cell_id) + ': ' + str(mesh_dict['elements'][MB8_cell_id]['center']))

if MB8_df['domain'][idx3]=='ROCKS':
    print('!!!! MB8 monitoring cell outside fault core and damage zone ... getting cell at fault core (FZONE) !!!!')
    idx33 = np.where(MB8_df['domain']=='FZONE')[0][0]
    MB8_cell_id = MB8_df['cell_id'][idx33]
    print('NEW monitoring cell (along MB8) ----> ' + str(MB8_cell_id) + ': ' + str(mesh_dict['elements'][MB8_cell_id]['center']))

# MB4
MB4_df = pd.read_csv(project_path + '0_boreholes/finer_mesh_3.0/' + 'MB4_borehole_information_edited.csv', index_col=0)
idx4 = np.argmin(np.abs(np.array(MB4_df['Depth (m)']) - PS_MB4))
MB4_cell_id = MB4_df['cell_id'][idx4]

print('Monitoring cell (along MB4) ----> ' + str(MB4_cell_id) + ': ' + str(mesh_dict['elements'][MB4_cell_id]['center']))

if MB4_df['domain'][idx4]=='ROCKS':
    print('!!!! MB4 monitoring cell outside fault core and damage zone ... getting cell at fault core (FZONE) !!!!')
    idx44 = np.where(MB4_df['domain']=='FZONE')[0][0]
    MB4_cell_id = MB4_df['cell_id'][idx44]
    print('NEW monitoring cell (along MB4) ----> ' + str(MB4_cell_id) + ': ' + str(mesh_dict['elements'][MB4_cell_id]['center']))

#%% Write INFILE

# Initialize parameters dictionary

parameters = {
    'title': 'TOUGH3-FLAC3D VALTER ST1 interval 13 stimulation model for FEAR',
    'eos': 'eos3',          # EOS3 for water and air
    'n_component': 2,       # two components: water and air
    'n_phase': 2,           # one phase, only liquid (but here set to 2 because otherwise it does not work)
    'start': True,
    'isothermal': False,    # temperature changes during the simulation
}

# specify if steady-state (initial conditions) or injection
# mode = 'steady_state'
# mode = 'enthalpy_check'
mode = 'injection'
mesh_type = 'finer'
coupled = False

# ROCKS #

# add domain-specific parameters
parameters['rocks'] = {
    'ROCKS': {
        'permeability': 1.e-17,     # permeability
        'specific_heat': 790.0,     # specific heat
        'permeability_model': {     # permeability model (FLAC)
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {   # equivalent pore pressure (FLAC)
            'id': 3,
            'parameters': [0.0],
        },
    },
    'FUPPE': {
        'permeability': 1.e-19,
        'specific_heat': 790.0,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
    'FZONE': {
        'porosity': 0.03,                # add different porosity to the fault core zone
        'permeability': 5.0e-15,
        'specific_heat': 790.0,
        'compressibility': 4.0e-10,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
    'INJPO': {                          # new domain for the injection point only (HAS TO BE MODIFIED ACCORDINGLY INSIDE THE MESH!)
        'density': 1000.0,
        'porosity': 0.99,
        'permeability': 1.0e-12,
        'conductivity': 0.6,
        'specific_heat': 4186.0,
        'compressibility': 0.0,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
    # 'INPS1': {
    #     'density': 2620.0,
    #     'porosity': 0.25,
    #     'permeability': 5.0e-13,
    #     'specific_heat': 790.0,
    #     'compressibility': 1.0e-11,
    #     'permeability_model': {
    #         'id': 1,
    #         'parameters': [0.0],
    #     },
    #     'equivalent_pore_pressure': {
    #         'id': 1,
    #         'parameters': [0.0],
    #     },
    # },
    # 'INPS2': {
    #     'density': 2620.0,
    #     'porosity': 0.1,
    #     'permeability': 1.0e-13,
    #     'specific_heat': 790.0,
    #     'compressibility': 1.0e-10,
    #     'permeability_model': {
    #         'id': 1,
    #         'parameters': [0.0],
    #     },
    #     'equivalent_pore_pressure': {
    #         'id': 1,
    #         'parameters': [0.0],
    #     },
    # },
    'FBOTT': {
        'permeability': 1.e-19,
        'specific_heat': 790.0,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
    'BFRON': {
        'permeability': 1.e-16,
        'specific_heat': 1.e55,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
    'BBACK': {
        'permeability': 1.e-16,
        'specific_heat': 1.e55,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
    'BLEFT': {
        'permeability': 1.e-16,
        'specific_heat': 1.e55,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
    'BRIGH': {
        'permeability': 1.e-16,
        'specific_heat': 1.e55,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
    'BUPPE': {
        'permeability': 1.e-16,
        'specific_heat': 1.e55,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
    'BBOTT': {
        'permeability': 1.e-16,
        'specific_heat': 1.e55,
        'permeability_model': {
            'id': 1,
            'parameters': [0.0],
        },
        'equivalent_pore_pressure': {
            'id': 3,
            'parameters': [0.0],
        },
    },
}

# if mesh_type=='finer':
#     parameters['rocks'].update(
#         {
#             'ZONET': {
#                 'porosity': 0.01,
#                 'permeability': 1.0e-14,
#                 'specific_heat': 790.0,
#                 'compressibility': 6.0e-10,
#                 'permeability_model': {
#                     'id': 1,
#                     'parameters': [0.0],
#                 },
#                 'equivalent_pore_pressure': {
#                     'id': 3,
#                     'parameters': [0.0],
#                 },
#             },
#             'ZONEB': {
#                 'porosity': 0.01,
#                 'permeability': 1.0e-14,
#                 'specific_heat': 790.0,
#                 'compressibility': 6.0e-10,
#                 'permeability_model': {
#                     'id': 1,
#                     'parameters': [0.0],
#                 },
#                 'equivalent_pore_pressure': {
#                     'id': 3,
#                     'parameters': [0.0],
#                 },
#             },
#         }
#     )

if mode=='steady_state':

    # Add initial conditions for upper and lower boundary faces
    parameters['rocks']['BBOTT']['initial_condition'] = [3.4e6, 1.0e-50, 24.0]
    parameters['rocks']['BUPPE']['initial_condition'] = [3.0e6, 1.0e-50, 15.0]

    # set default parameters (valid for all materials)
    parameters['default'] = {
        'density': 2620.0,  # density
        'porosity': 1.75e-2,  # porosity
        'compressibility': 1.e-9,  # pore compressibility
        'conductivity': 4.0,  # thermal conductivity in saturated conditions
        'conductivity_dry': 3.2,  # thermal conductivity in dry conditions
        # now we define the choice of relative permeability and capillary functions
        # 'relative_permeability': {
        #     'id': 3,  # Corey's curves
        #     'parameters': [0.3, 0.05],  # 1) residual liquid saturation 2) residual gas saturation
        # },
        # 'capillarity': {
        #     'id': 1,  # Van Genuchten function
        #     'parameters': [0.0, 0.25, 1.0],
        #     # 1) max. negative capillary pressure 2) liquid saturation lower limit 3) liquid saturation higher limit
        # },
        'initial_condition': [3.3e6, 1.0e-50, 19.5]
    }

    # PARAM #

    parameters['options'] = {
        'n_cycle': 9999,  # maximum number of time steps
        't_ini': 0.0,  # starting time of the simulation
        'n_cycle_print': 9999,                                      # printout for every multiple of this number (if 9999, printout as specified in parameters times (see cell bottom)
        't_max': 100000.0 * 365.25 * 24.0 * 3600.0,                               # years of steady state calculation, in seconds
        't_steps': 50.0 * 365.25 * 24.0 * 3600.0,                                    # time step size (years)
        't_step_max': 1000.0 * 365.25 * 24.0 * 3600.0,                               # maximum time step size (years)
        'eps1': 1.0e-5,                                                 # convergence criterion for relative error (default)
        'gravity': 9.81,
        't_reduce_factor': 2,                                           # factor by which the time step is reduced in case of convergence failure or other problems
    }

    # MOP #

    parameters['extra_options'] = {
        1: 1,                                                           # printout for non-convergent iterations
        7: 1,                                                           # printout of input data provided
        10: 0,                                                          # chooses the interpolation formula for heat conductivity as a function of liquid saturation
        11: 4,                                                          # determines evaluation of mobility and permeability at interfaces: 3) mobilities are averaged between adjacent elements, permeability is harmonic weighted
        13: 1,                                                          # writes user-specific initial conditions to file SAVE
        16: 4,                                                          # time step size will be doubled if convergence occurs within ITER <= 4 Newton-Raphson iterations
        18: 0,                                                          # performs upstream weighting for interface density
        21: 3,                                                          # DSLUCS as linear equation solver
    }

    # TIMES # customize the output times
    # days_range = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # d = 24.0 * 3600.0
    # parameters['times'] = [x * d for x in days_range]

    # OUTPU #

    # WRITE THE INFILE #
    toughio.write_input('INFILE_' + mode, parameters)

elif mode=='injection':

    # GENER #

    input_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/DSS_DTS_py_project/Valter_phase1_interval13/'

    hydraulics = pd.read_hdf(input_path + 'Hydraulics_filtered_int13_alt4.h5', key='df', mode='r')

    # injection_mode = 'single_point'
    injection_mode = 'multiple_points'

    if injection_mode=='single_point':
        parameters['generators'] = [
            {
                'label': inj_cell,
                'type': 'COM1',     # injection
                'times': list(hydraulics.time),
                'rates': list(hydraulics.flowrate),
                'specific_enthalpy': list(np.full(len(hydraulics.time), 7.867e4)),  # specific enthalpy at 3.3 MPa and 18 degrees of injected water temperature
                # 'specific_enthalpy': list(np.full(len(hydraulics.time), 5.248e4)),  # specific enthalpy at the pressure at the injection point and at 12.0 °C
                # 'specific_enthalpy': list(np.full(len(hydraulics.time), -2.818e4)), # T = 6.0 °C
            }
        ]
    else:
        # inj_cells_ids = ['A1938', 'ACO90', 'ACO75', 'ACO76', 'ACO77', 'ACO62', 'A1R22'] # finer_mesh_1.0 (introduced 19.09.23)
        # inj_cells_ids = ['AO348', 'AO349', 'AO350', 'AO351', 'AO352', 'AO353', 'AO293', 'AO299'] # finer_mesh_2.0 (introduced 24.10.23)
        inj_cells_ids = ['ASU61', 'ASU62', 'ASU63', 'ASU64', 'ASU65'] # finer_mesh_3.0.2 (introduced 31.10.23)
        # inj_cells_ids = ['AI340', 'AI341', 'AI342', 'AI343', 'AI344'] # finer_mesh_3.0.1
        # re-open mesh_dict (BECAUSE: the new included INJPO cells do not have center coordinates, could not be used in previous cell!)
        mesh_dict = toughio.read_mesh('/home/victor/Desktop/toughio/examples/TOUGH3-FLAC3D_project/finer_mesh_3.0.4/MESH_injection')
        hydraulics_new = utils.inj_cells_distributing_1(hydraulics, mesh_dict, inj_cells_ids)
        # or
        # inj_cells_ids = [['AO348', 'AO349', 'AO350', 'AO351', 'AO352', 'AO353'], # A
        #                 ['AO288', 'AO289', 'AO290', 'AO291', 'AO292', 'AO293', # B
        #                  'AO282', 'AO283', 'AO284', 'AO285', 'AO286', 'AO287', # C
        #                  'AO342', 'AO343', 'AO344', 'AO345', 'AO346', 'AO347', # D
        #                  'AO4 2', 'AO4 3', 'AO4 4', 'AO4 5', 'AO4 6', 'AO4 7', # E
        #                  'AO4 8', 'AO4 9', 'AO410', 'AO411', 'AO412', 'AO413', # F
        #                  'AO414', 'AO415', 'AO416', 'AO417', 'AO418', 'AO419', # G
        #                  'AO354', 'AO355', 'AO356', 'AO357', 'AO358', 'AO359', # H
        #                  'AO294', 'AO295', 'AO296', 'AO297', 'AO298', 'AO299'], # I
        #                 ['AO168', 'AO169', 'AO170', 'AO171', 'AO172', 'AO173', # J
        #                  'AO162', 'AO163', 'AO164', 'AO165', 'AO166', 'AO167', # K
        #                  'AO156', 'AO157', 'AO158', 'AO159', 'AO160', 'AO161', # L
        #                  'AO276', 'AO277', 'AO278', 'AO279', 'AO280', 'AO281', # M
        #                  'AO336', 'AO337', 'AO338', 'AO339', 'AO340', 'AO341', # N
        #                  'AO396', 'AO397', 'AO398', 'AO399', 'AO4 0', 'AO4 2', # O
        #                  'AO456', 'AO457', 'AO458', 'AO459', 'AO460', 'AO461', # P
        #                  'AO462', 'AO463', 'AO464', 'AO465', 'AO466', 'AO467', # Q
        #                  'AO468', 'AO469', 'AO470', 'AO471', 'AO472', 'AO473', # R
        #                  'AO474', 'AO475', 'AO476', 'AO477', 'AO478', 'AO479', # S
        #                  'AO480', 'AO481', 'AO482', 'AO483', 'AO484', 'AO485', # T
        #                  'AO420', 'AO421', 'AO422', 'AO423', 'AO424', 'AO425', # V
        #                  'AO360', 'AO361', 'AO362', 'AO363', 'AO364', 'AO365', # W
        #                  'AO3 0', 'AO3 1', 'AO3 2', 'AO3 3', 'AO3 4', 'AO3 5', # X
        #                  'AO180', 'AO181', 'AO182', 'AO183', 'AO184', 'AO185', # Y
        #                  'AO174', 'AO175', 'AO176', 'AO177', 'AO178', 'AO179']] # Z
        # weights = [0.6, 0.3, 0.1] # sum must be equal to 1
        # hydraulics_new = utils.inj_cells_distributing_2(hydraulics, mesh_dict, inj_cells_ids, weights)
        #
        # if isinstance(inj_cells_ids[0], list)==True:
        #     # reorder list of cell ids
        #     inj_cell_ids_flattened = []
        #     for sublist in inj_cells_ids:
        #         inj_cell_ids_flattened.extend(sublist)
        #     inj_cells_ids = inj_cell_ids_flattened
        #     # reorder the hydraulics
        #     # TODO: Program this better (should adapt to len(weights)!)
        #     hydraulics_new_concatenated = np.concatenate((hydraulics_new[0], hydraulics_new[1], hydraulics_new[2]), axis=0)
        #     hydraulics_new = hydraulics_new_concatenated

        parameters['generators'] = []
        for i in range(len(hydraulics_new[:, 0])):
            parameters['generators'].append(
                {
                    'label': inj_cells_ids[i],
                    'type': 'COM1',
                    'times': list(hydraulics.time),
                    'rates': list(hydraulics_new[i, :]),
                    'specific_enthalpy': list(np.full(len(hydraulics.time), 7.869e4)), # at 18°C injected water for ASU63 pressure conditions
                }
            )

    # set default parameters (valid for all materials)
    parameters['default'] = {
        'density': 2620.0,  # density
        'porosity': 1.75e-2,  # porosity
        'compressibility': 1.e-9,  # pore compressibility
        'conductivity': 4.0,  # thermal conductivity in saturated conditions
        'conductivity_dry': 3.2,  # thermal conductivity in dry conditions
        # now we define the choice of relative permeability and capillary functions
        # 'relative_permeability': {
        #     'id': 3,  # Corey's curves
        #     'parameters': [0.3, 0.05],  # 1) residual liquid saturation 2) residual gas saturation
        # },
        # 'capillarity': {
        #     'id': 1,  # Van Genuchten function
        #     'parameters': [0.0, 0.25, 1.0],
        #     # 1) max. negative capillary pressure 2) liquid saturation lower limit 3) liquid saturation higher limit
        # },
        # 'initial_condition': [3.3e6, 0.1 + 10, 20.0]
    }

    # PARAM #

    parameters['options'] = {
        'n_cycle': 9999,                                            # maximum number of time steps
        # 't_ini': 0.0,                                               # starting time of the simulation
        't_ini': 10.0 * 3600.0,
        'n_cycle_print': 9999,                                         # printout for every multiple of this number (every 30 min, since time step is 1 min)
        # 't_max': list(hydraulics.time)[-1],                         # maximum injection time
        't_max': 20.0 * 3600.0,
        't_steps': 1.0 * 60.0,                                      # time step size (minutes)
        't_step_max': 10.0 * 60.0,                                  # maximum time step size (minutes)
        'eps1': 1.0e-5,                                             # convergence criterion for relative error (default)
        'gravity': 9.81,
        't_reduce_factor': 4,                                       # factor by which the time step is reduced in case of convergence failure or other problems
    }

    # MOP #

    parameters['extra_options'] = {
        1: 1,                                                           # printout for non-convergent iterations
        7: 1,                                                           # printout of input data provided
        10: 0,                                                          # chooses the interpolation formula for heat conductivity as a function of liquid saturation
        11: 3,                                                          # determines evaluation of mobility and permeability at interfaces: 3) mobilities are averaged between adjacent elements, permeability is harmonic weighted
        12: 0,                                                          # triple linear interpolation for time dependent sink/source data
        16: 4,                                                          # time step size will be doubled if convergence occurs within ITER <= 4 Newton-Raphson iterations
        18: 0,                                                          # performs upstream weighting for interface density
        21: 8,                                                          # DSLUCS as linear equation solver # for coupled simulation with FLAC, change to 8, PETSc # Also PETSc works for parallel computing, while DSLUCS does not
    }

    # MOMOP #

    parameters['more_options'] = {
        14: 1,                                                          # reprint input file at the end of the TOUGH3 output file
        15: 1,                                                          # Porosity of block ROCKS is used for calculation of rock energy content
        17: 1,                                                          # if > 0, then prints variables according to OUTPU block for the specified cell in FOFT
    }

    # FOFT # element history (specifies grid blocks for which time series data is desired)
    parameters['element_history'] = [
        inj_cell,                           # injection cell id
        MB2_cell_id,                        # monitoring point cell id
        MB4_cell_id,                        # monitoring point cell id
        MB5_cell_id,                        # monitoring point cell id
        MB8_cell_id                         # monitoring point cell id
    ]
    # or

    # FOR FINER_MESH_1.0 #
    # if mesh_type=='finer':
    #     for l in ['ACO46', 'ACO60', 'ACO61', 'ACO62', 'AAM59', 'A1943', 'ACO75', 'ACO77', 'A1R27', 'A6427', 'ACO90', 'ACO91', 'ACO92', 'ACP 6']:
    #         parameters['element_history'].append(l)
    #
    #     # COFT # connection history (specifies connections for which time series data are desired)
    #     parameters['connection_history'] = [
    #         'ACO90ACO91',
    #         'ACO91ACO92',
    #         'ACO91ACP 6',
    #         'ACO46ACO61',
    #         'ACO60ACO61',
    #         'ACO61ACO62',
    #         'A1943AAM59',
    #         'ACO60ACO75',
    #         'A1943ACO75',
    #         'ACO75ACO90',
    #         'ACO75ACO76',
    #         'ACO61ACO76',
    #         'ACO76ACO91',
    #         'ACO76ACO77',
    #         'ACO62ACO77',
    #         'ACO77ACO92',
    #         'A1R27ACO77',
    #         'A1R27A6427'
    #     ]
    # else:
    #     for l in ['A9P78', 'A9P83', 'A9P63', 'A9P68', 'A9P71', 'A9P72', 'A9P74', 'A9P75', 'A9P67', 'A9P69', 'A9P77', 'A9P79']:
    #         parameters['element_history'].append(l)
    #
    #     # COFT # connection history (specifies connections for which time series data are desired)
    #     parameters['connection_history'] = [
    #         'A9P73A9P78',
    #         'A9P73A9P74',
    #         'A9P73A9P72',
    #         'A9P73A9P68',
    #         'A9P78A9P83',
    #         'A9P71A9P72',
    #         'A9P63A9P68',
    #         'A9P78A9P83',
    #         'A9P78A9P79',
    #         'A9P77A9P78',
    #     ]

    # # FOR FINER_MESH_2.0 #
    # parameters['element_history'] = []
    # for l in ['AO1 8', 'AO1 9', 'AO110', 'AO111', 'AO112', 'AO113',
    #           'AO168', 'AO169', 'AO170', 'AO171', 'AO172', 'AO173',
    #           'AO288', 'AO289', 'AO290', 'AO291', 'AO292', 'AO293',
    #           'AO348', 'AO349', 'AO350', 'AO351', 'AO352', 'AO353',
    #           'AO4 0', 'AO4 9', 'AO410', 'AO411', 'AO412', 'AO413',
    #           'AO468', 'AO469', 'AO470', 'AO471', 'AO472', 'AO473',
    #           'AO528', 'AO529', 'AO530', 'AO531', 'AO532', 'AO533']:
    #     parameters['element_history'].append(l)

    # # COFT # connection history (specifies connections for which time series data are desired)
    # parameters['connection_history'] = []

    # TIMES #
    times_all_1 = np.arange(10.0, 20.0, 0.5) * 3600.0
    times_all_2 = np.arange(24.0, list(hydraulics.time)[-1] / 3600.0, 4.0) * 3600.0
    parameters['times'] = list(np.concatenate((times_all_1, times_all_2), axis=0))

    # OUTPUT #
    parameters['output'] = {
        'variables': [
            {'name': 'pressure'},
            {'name': 'temperature'},
            # {'name': 'heat flow'},
            {'name': 'coordinate'},
            {'name': 'index'},
        ]
    }

    # Instead of OUTPUT do a time series of th injected cell to first compare with the pressure and temperature -> put MCYPR to 9999 then!

    if coupled==True:
        parameters['flac'] = {
            'creep': False,             # False: default mode, True: creep mode
            'porosity_model': 1,        # 0: no mechanical induced porosity change, 1: mechanical-induced porosity change, 2: porosity change as a function of volumetric strain
            'version': 7,
        }

        # The option 'FLAC' automatically adds the modes permeability model and equivalent pore pressure in 'ROCKS'

        # Change the linear equation solver (as mentioned above)
        parameters['extra_options'][21] = 8

        # WRITE THE INFILE #
        toughio.write_input('INFILE_coupled_' + mode, parameters)

    else:
        # WRITE THE INFILE #
        toughio.write_input('INFILE_' + mode, parameters)

else:
    raise TypeError('no such condition')