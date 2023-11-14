from converter.input import *
from converter.get_coord_mat import *
from converter.get_adj_mat import *
from converter.drop_nodes import *
from converter.output import *

import numpy as np

def process_files(input_filepath: str, output_dirpath: str, 
                  fixed_edges: str, is_directed: str, 
                  is_full_to_selected_nodes: str, get_coords: str) -> None:
    """
    Read the NC file that contains a spatiotemporal grid, and output the 
    converted graph in NPY files.
    :param input_filepath: the path to the input NC file
    :param output_dirpath: the path to the output directory
    :param fixed_edges: the fixed number of edges for all nodes
    :param is_directed: if the generated graph is directed
    :param is_full_to_selected_nodes: if all edges are added to the selected nodes
    :param get_coords: if the node coordinates are output
    """
    # Generate and save the node feature tensor.
    grid = input_grid(input_filepath)
    
    # Get the coordinate tensor.
    coordinate_grid = get_coord_mat(input_filepath)
    grid_transposed = grid.transpose(1,2,0)
    grid_flattened = grid_transposed.reshape(grid_transposed.shape[0] * 
                                             grid_transposed.shape[1], 
                                             grid_transposed.shape[2])
                                             
    # Remove nodes with NAs.
    node_feats = drop_rows_with_nas(grid_flattened)
    output(node_feats, output_dirpath, 'node_feats')
    
    # Temporary: Test the program with a smaller node feature matrix.
    # Comment the line below when normally running the program.
    #node_feats = node_feats[:200]
    
    # Output the coordinate tensor (slow).
    if str(get_coords) == 'yes':
        # Also remove the corresponding nodes in the coorindate tensor.
        land_indices = np.setdiff1d(np.arange(grid_flattened.shape[0]), np.where(
                       (grid_flattened[:, np.newaxis] == node_feats).all(-1))[0])
        coordinates_ocean = np.delete(coordinate_grid, land_indices, axis=0)
        output(coordinates_ocean, output_dirpath, 'coords')
  
    # Convert the string into the numerical.
    fixed_edges = int(str(fixed_edges))
    
    # Convert the string into the boolean.
    is_directed_bool = True if str(is_directed) is 'yes' else False
    is_directed_printed = '_directed' if is_directed_bool else ''
    is_full_to_selected_nodes_bool = True if str(is_full_to_selected_nodes) is 'yes' else False
    is_full_to_selected_nodes_printed = '_12' if is_full_to_selected_nodes_bool else ''
    adj_mat = get_adj_mat(node_feats, fixed_edges, is_directed_bool, is_full_to_selected_nodes)
    output(adj_mat, output_dirpath, 'adj_mat' + '_' + str(fixed_edges) + 
           is_full_to_selected_nodes_printed + is_directed_printed)