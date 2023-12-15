"""
Preprocessing script to manipulate MESH plain text file only (cell 1)
-> Applies open boundary conditions to material labels starting with "B"
-> rewrites ELEME block using toughio routines and copies CONNE block
Manipulates MESH using the textfile and the toughio object exported and transformed from FLAC (cell 2)
-> reason for combination is that the material ids are not exported in the object format, for applying the boundary conditions
-> Also, the INCON file can be easily created and exported with the new MESH
@ by Victor Clasen Repollés (victor.clasen@sed.ethz.ch)
"""

# Import the required libraries
import toughio
import numpy as np

from toughio._mesh.tough._tough import _write_eleme as write_eleme
# from toughio._mesh.tough._tough import _write_conne as write_conne
from toughio._common import open_file

#%% MANIPULATE MESH ONLY WITH TEXTFILE

# input_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/preprocessing_OUTPUT/'
#
# mesh_dict = toughio.read_mesh(input_path + 'MESH')
#
# def bound_correction(last_material):
#     if last_material.startswith('B'):
#         return int(1)
#     else:
#         return int(0)
#
# def data_rearrange(mesh_dict, apply_bcond=False):
#     # pre-calculations
#     n_cells = len(mesh_dict['elements'])                            # get the number of cells in the mesh
#     n_coord = len(mesh_dict['elements']['A11 0']['center'])         # get the spatial dimension (number of coordinates)
#     # get array of cell names
#     labels = np.asarray(list(mesh_dict['elements'].keys()))
#     # get array of cell center coordinates
#     nodes = [[0] * n_coord for _ in range(n_cells)]
#     # get materials labels for all cells
#     materials = []
#     # get volumes for all cells
#     volumes = []
#     # set boundary conditions
#     boundary_conditions = np.zeros(n_cells, dtype=int)
#     for i in range(n_cells):
#         if n_cells!=len(labels):
#             raise ValueError
#         materials.append(mesh_dict['elements'][labels[i]]['material'])
#         volumes.append(mesh_dict['elements'][labels[i]]['volume'])
#         if apply_bcond==True:
#             boundary_conditions[i] = bound_correction(materials[-1])
#         for j in range(n_coord):
#             nodes[i][j] = float(mesh_dict['elements'][labels[i]]['center'][j])
#     """
#     nodes and volumes have to be converted to arrays, while material_name is an empty dictionary
#     and material_end an empty list
#     """
#     return labels, np.asarray(nodes), materials, np.asarray(volumes), boundary_conditions, dict(), []
#
# def copy_lines(source_file_1, source_file_2, destination_file='MESH_FINAL'):
#     with open(source_file_1, 'r') as f_source_1:
#         lines_to_copy_1_ = f_source_1.readlines()
#     with open(source_file_2, 'r') as f_source_2:
#         lines_to_copy_2 = f_source_2.readlines()
#
#     lines_to_copy_1 = lines_to_copy_1_[len(lines_to_copy_2):len(lines_to_copy_1_)]
#
#     lines_to_copy_all = lines_to_copy_2 + lines_to_copy_1
#
#     with open(destination_file, 'w') as f_dest:
#         f_dest.writelines(lines_to_copy_all)
#
# # data preparation
# data = data_rearrange(mesh_dict, apply_bcond=True)
#
# # rewrite textfile
# with open_file('MESH_eleme', 'w') as f:
#     # ELEME
#     write_eleme(
#         f,
#         data[0],    # labels (cell ids)
#         data[1],    # nodes (cell center points)
#         data[2],    # materials
#         data[3],    # volumes
#         data[4],    # boundary conditions criteria
#         data[5],    # material name (empty)
#         data[6],    # material end (empty)
#     )
#     # CONNE
#     """
#     CONNE block hast to be copied from the old file because:
#     -> read toughio object does not correspond to MESH
#     -> Therefore, no existing information on: edge points, faces, face normals, etc.
#     -> Cannot use write_conne()
#     """
#
# # now copy and paste the CONNE part from the old file
# copy_lines('MESH', 'MESH_eleme', destination_file='MESH_bound')

#%% MANIPULATE MESH WITH TOUGHIO OBJECT FILE

import meshio

# input_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/0_preprocessing_OUTPUT/25.08/'
input_path = '/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/0_preprocessing_OUTPUT/fine_mesh_3110/'

# mesh_dict = toughio.read_mesh('/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/' + 'MESH')
mesh_dict = toughio.read_mesh('/home/victor/ownCloud - Victor Clasen Repolles (victor.clasen@sed.ethz.ch)@polybox.ethz.ch/Shared/TOUGH-FLAC_project/0_preprocessing_OUTPUT/fine_mesh_3110/' + 'MESH')

mesh_toughio_obj = toughio.from_meshio(meshio.flac3d.read(input_path + 'OUTPUT1_mesh.f3grid'))

# def add_value_to_list_of_lists(list_of_lists, value):
#     for sub_list in list_of_lists:
#         for i in range(len(sub_list)):
#             sub_list[i] += value

def combine_meshes(mesh_dict, mesh_toughio_obj, apply_to='all'):
    cell_ids = list(mesh_dict['elements'].keys())
    material_list = []
    for i in range(len(cell_ids)):
        material_list.append(mesh_dict['elements'][cell_ids[i]]['material'])
    material_list_array = np.asarray(material_list)
    material_list_unique = np.unique(material_list_array)
    # get indices of unique materials
    ind_all = []
    int_array = np.arange(len(material_list_unique))
    for j in range(len(material_list_unique)):
        ind_all.append([index for index, value in enumerate(material_list_array) if value == material_list_unique[j]])
        mesh_toughio_obj.materials[ind_all[-1]] = int_array[j]
    # filter material labels to boundary only
    if apply_to=='all':
        bound_material =[item for item in material_list_unique if isinstance(item, str) and item.startswith('B')]
    elif apply_to=='upper_lower':
        bound_material = ['BBOTT', 'BUPPE']
    bcond_list = []
    for j in range(len(material_list_unique)):
        mesh_toughio_obj.add_material(material_list_unique[j], int_array[j])
        if material_list_unique[j] in bound_material:
            # set boundary condition
            bcond_list.append((material_list_array == material_list_unique[j]).astype(int))
    mesh_toughio_obj.add_cell_data('boundary_condition', np.sum(np.asarray(bcond_list), axis=0))
    return mesh_toughio_obj

mesh_combined = combine_meshes(mesh_dict, mesh_toughio_obj, apply_to='upper_lower')

# INCON #

centers = mesh_combined.centers
incon = np.full((mesh_combined.n_cells, 3), -1.0e9)
incon[:, 0] = 3.3e6 - 13900.85 * centers[:, 2]               # linear pressure gradient from sensor readings at ST1 (with pressure at the injection point just before start of stimulation)
incon[:, 1] = 1.0e-50
incon[:, 2] = 19.4 - 0.020855 * centers[:, 2]               # linear temperature gradient from DTS data (with temp at the injection point)
# incon[:, 2] = 19.5 - 0.034 * centers[:, 2]
mesh_combined.add_cell_data('initial_condition', incon)

mesh_combined.write_tough(input_path + 'MESH_steady_state', incon=True)
# mesh_combined.write('mesh.pickle')