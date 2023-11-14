from pathlib import Path
import argparse

from converter.process_files import *

# Argument parser
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('input_filepath', type=str,
                        help='Input File Pathname')
arg_parser.add_argument('output_dirpath', type=str,
                        help='Output Directory Pathname')
arg_parser.add_argument('fixed_edges', type=str,
                        help='Node Feature Correlation Threshold')
arg_parser.add_argument('is_directed', type=str,
                        help="If Directed Graphs: 'yes' or 'no'")
arg_parser.add_argument('is_full_to_selected_nodes', type=str,
                        help="If All Edges to Selected Nodes: 'yes' or 'no'")
arg_parser.add_argument('get_coords', type=str,
                        help="If Output Coordinates: 'yes' or 'no'")
args = arg_parser.parse_args()

input_filepath = Path(args.input_filepath)
output_dirpath = Path(args.output_dirpath)
fixed_edges = Path(args.fixed_edges)
is_directed = Path(args.is_directed)
is_full_to_selected_nodes = Path(args.is_full_to_selected_nodes)
get_coords = Path(args.get_coords)

# Read the input, process the input, and write the output.
process_files(input_filepath, output_dirpath, fixed_edges, is_directed, 
              is_full_to_selected_nodes, get_coords)