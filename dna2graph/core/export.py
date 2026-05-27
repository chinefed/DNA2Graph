import os
import string
import itertools

import cv2
import h5py
import numpy as np
import pandas as pd
import networkx as nx
from roifile import ROI_TYPE, ImagejRoi, roiwrite

from dna2graph.core.classification import GROUP_NAME_MAP
from dna2graph.core.classification import classify_components
from dna2graph.core.graph_utils import (
    graph_to_mask,
    get_graph_bounding_box,
    linear_walk_decomposition
)


GRAPH2MASK_DILATION_SIZE = 5


def generate_alphabetic_id():
    '''
    A generator that produces an infinite sequence of letter combinations, starting from 
    single letters ('a' to 'z'), then moving to two-letter combinations ('aa', 'ab', 'ac', ...), 
    and continuing with longer combinations indefinitely (e.g., 'aaa', 'aab', 'aac', ...).
    '''
    for length in itertools.count(1):  # Start from length 1, and keep increasing
        for combo in itertools.product(string.ascii_lowercase, repeat=length):
            yield ''.join(combo)


def assign_component_ids(G):
    '''
    Assigns a unique identifier to each connected component in the input graph.

    The identifier is constructed from the component's group name and its sequential
    index within that group, and is stored in the `component_id` attribute of all
    nodes in the component.
    This function assumes that each node in the input graph carries a `component_group`
    attribute specifying its numerical group identifier. Numerical group identifiers are
    mapped to group names using the `GROUP_NAME_MAP` dictionary.
    '''
    G = G.copy()
    index_tracker = {}

    for component in nx.connected_components(G):
        v = next(iter(component)) # Get an arbitrary node from the component
        component_group = G.nodes[v]['component_group'] # Get the component group

        # Get the sequential index of the component within its group
        index = index_tracker.get(component_group, 0)
        index_tracker[component_group] = index + 1 # Update index tracker

        # Generate the component identifier
        group_name = GROUP_NAME_MAP[component_group]
        component_id = f'{group_name}_{index}'

        # Assign the component ID to each node in the component
        nx.set_node_attributes(G, {u: component_id for u in component}, 'component_id')

    return G


def extract_components_info(G):
    '''
    Extract detailed information for each connected component of the input graph, including:
    - Component ID (if nodes do not have 'component_id' attribute, None)
    - Component group (if nodes do not have 'component_group' attribute, None)
    - Bounding box
    - Linear walk decomposition
    '''
    components_info = []

    for component in nx.connected_components(G):
        v = next(iter(component)) # Get an arbitrary node from the component
        component_id = G.nodes[v].get('component_id', None) # Get the component ID
        component_group = G.nodes[v].get('component_group', None) # Get the component group

        # Get bounding box and perform linear walk decomposition
        G_component = G.subgraph(component)
        component_bbox = get_graph_bounding_box(G_component)
        component_walks = linear_walk_decomposition(G_component)

        # Store the component information
        components_info.append({
            'id': component_id,
            'group': component_group,
            'bbox': component_bbox,
            'walks': component_walks
        })

    return components_info


def linear_walks_to_multi_coords(walks):
    '''
    Encodes a list of linear walks into the internal ImageJ format for composite ROIs.
    All walks are combined into a single composite ROI.
     
    The format is a flat array of opcodes and coordinates. The coordinates are in pixels,
    with the origin at the top-left corner of the image.
    The format is as follows:
    - 0, x, y: move the cursor to point (y, x)
    - 1, x, y: draw a line to point (y, x)
    Here, we never use the opcode 4 as we are interested in lines and not closed shapes.
    '''
    multi_coordinates = []

    for walk in walks:
        encoded_walk = [0]  # Start with a MOVETO opcode (0)

        for node in walk:
            y, x = node
            encoded_walk += [x, y, 1] # Add LINETO opcode (1) after each coordinate pair
        encoded_walk = encoded_walk[:-1] # Remove the last LINETO opcode

        multi_coordinates += encoded_walk

    return np.array(multi_coordinates, dtype=np.float32)


class Exporter:
    '''
    A class for exporting analysis outputs, including:
    - Graph representations
    - Binary segmentation masks
    - CSV reports
    - ImageJ ROIs for segmentations, bounding boxes, and linear walk decompositions
    '''
    def __init__(self,
                 output_root_dir,
                 save_graph=True,
                 save_mask=True,
                 save_report=True,
                 save_segmentation_rois=True,
                 save_bbox_rois=True,
                 save_lin_decomp_rois=True
        ):
        self.output_root_dir = output_root_dir
        self.save_graph = save_graph
        self.save_mask = save_mask
        self.save_report = save_report
        self.save_segmentation_rois = save_segmentation_rois
        self.save_bbox_rois = save_bbox_rois
        self.save_lin_decomp_rois = save_lin_decomp_rois

        self.G = None # Placeholder for the graph to be exported
        self.img_name = None # Placeholder for the image name
        self.output_dir = None # Placeholder for the output directory
        self.components_info = None # Placeholder for components information

    def load_graph(self, G, img_name):
        '''
        Load the graph and image name for exporting.
        '''
        self.G = G
        self.img_name = img_name

        # Create output directory for the current image
        self.output_dir = os.path.join(self.output_root_dir, self.img_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Classify components and assign IDs
        self.G = classify_components(self.G)
        self.G = assign_component_ids(self.G)

        # Extract components information
        self.components_info = extract_components_info(self.G)

    def save(self):
        '''
        Save the analysis output based on user preferences.
        '''
        # === Data Exports ===

        # Save the graph to the output directory
        if self.save_graph:
            self._save_graph()

        # Save the mask to the output directory
        if self.save_mask:
            self._save_mask()

        # Save the report to the output directory
        if self.save_report:
            self._save_report()

        # === ROI Exports (ImageJ) ===

        # Save the segmentation ROIs to the output directory
        if self.save_segmentation_rois:
            self._save_segmentation_rois()

        # Save the bounding box ROIs to the output directory
        if self.save_bbox_rois:
            self._save_bbox_rois()

        # Save the linear walk decomposition ROIs to the output directory
        if self.save_lin_decomp_rois:
            self._save_lin_decomp_rois()

    def _check_loaded(self):
        '''
        Check if a graph has been loaded for exporting.
        '''
        if self.G is None or self.img_name is None or self.output_dir is None or self.components_info is None:
            raise ValueError(
                "No graph loaded. Please load a graph using 'load_graph()' before exporting."
            )

    def _save_graph(self):
        '''
        Save the graph representation.
        '''
        # Construct the output file path where the graph will be saved
        output_path = os.path.join(self.output_dir, f'{self.img_name}_graph.h5')

        # Open the HDF5 file for writing
        with h5py.File(output_path, 'w') as f:
            # Loop through each component in the graph
            for component in self.components_info:
                component_id = component['id']
                component_group = component['group']
                component_walks = component['walks']

                # Create a new group in the HDF5 file to store the component's data
                grp = f.create_group(component_id)

                # Store the component's label as an attribute
                grp.attrs['group'] = component_group

                # Stack the component's walks into a single 2D array
                coords = np.vstack(component_walks, dtype=np.uint16, casting='unsafe')

                # The offsets array keeps track of where each walk begins in the 'coords' array.
                offsets = np.cumsum([0] + [len(w) for w in component_walks])
                
                # Store the datasets
                grp.create_dataset('coords', data=coords, dtype=np.uint16, compression='lzf')
                grp.create_dataset('offsets', data=offsets, dtype=np.uint32)

    def _save_mask(self):
        '''
        Convert the graph to a binary mask and save it.
        '''
        self._check_loaded()
        mask = graph_to_mask(self.G, GRAPH2MASK_DILATION_SIZE) # Convert graph to binary mask
        cv2.imwrite(os.path.join(self.output_dir, f'{self.img_name}_mask.png'), mask)

    def _save_report(self):
        '''
        Save a CSV report with information about graph components.
        '''
        self._check_loaded()
        records = []

        for component in self.components_info:
            component_id = component['id']
            component_group = component['group']
            component_bbox = component['bbox']
            component_walks = component['walks']

            # Unpack bounding box
            top, left, bottom, right = component_bbox

            # Count the number of linear walks in the component
            n_linear_walks = len(component_walks) 

            # Initialize record for this component
            record = {
                'id': component_id,
                'group': component_group,
                'top': top,
                'left': left,
                'bottom': bottom,
                'right': right,
                'n_linear_walks': n_linear_walks
            }
            # Generate alphabetic walk IDs (e.g., a, b, ..., z, aa, ab, ...)
            walk_id_generator = generate_alphabetic_id()

            for walk in component_walks:
                walk_id = next(walk_id_generator) # Get walk ID
                
                # Measure the length of the walk
                walk_length = np.linalg.norm(np.diff(walk, axis=0), axis=1).sum()
                record[walk_id] = walk_length # Store walk length in the record

            records.append(record) # Add to the list of records

        # Convert records to Pandas DataFrame
        df = pd.DataFrame.from_records(records)
        # Insert a column with the total length of all walks in a component
        df.insert(7, 'total_length', df.iloc[:, 7:].sum(axis=1, skipna=True))

        # Save DataFrame to CSV
        df.to_csv(
            os.path.join(self.output_dir, f'{self.img_name}_report.csv'), 
            index=False, 
            float_format='%.2f'
        )

    def _save_segmentation_rois(self):
        '''
        Save segmentation ROIs in ImageJ format.
        '''
        self._check_loaded()
        rois = []
        
        for component in self.components_info:
            component_id = component['id']
            component_group = component['group']
            component_bbox = component['bbox']
            component_walks = component['walks']

            # Create ImageJ ROI
            roi = ImagejRoi()
            roi.version = 228
            roi.roitype = ROI_TYPE.RECT
            roi.name = component_id
            roi.group = component_group
            roi.top, roi.left, roi.bottom, roi.right = component_bbox
            roi.multi_coordinates = linear_walks_to_multi_coords(component_walks)
            roi.shape_roi_size = len(roi.multi_coordinates)

            rois.append(roi)

        roiwrite(
            os.path.join(self.output_dir, f'{self.img_name}_segmentation_rois.zip'),
            rois,
            mode='w'
        )

    def _save_bbox_rois(self):
        '''
        Save bounding box ROIs in ImageJ format.
        '''
        self._check_loaded()
        rois = []

        for component in self.components_info:
            component_id = component['id']
            component_group = component['group']
            component_bbox = component['bbox']

            # Create ImageJ ROI
            roi = ImagejRoi()
            roi.version = 228
            roi.roitype = ROI_TYPE.RECT
            roi.name = component_id
            roi.group = component_group
            roi.top, roi.left, roi.bottom, roi.right = component_bbox

            rois.append(roi)

        roiwrite(
            os.path.join(self.output_dir, f'{self.img_name}_bbox_rois.zip'),
            rois,
            mode='w'
        )

    def _save_lin_decomp_rois(self):
        '''
        Save linear walk decomposition ROIs in ImageJ format.
        '''
        self._check_loaded()
        rois = []
        for component in self.components_info:
            component_id = component['id']
            component_group = component['group']
            component_walks = component['walks']

            # Generate alphabetic walk IDs (e.g., a, b, ..., z, aa, ab, ...)
            walk_id_generator = generate_alphabetic_id()

            for walk in component_walks:
                walk_id = next(walk_id_generator) # Get walk ID

                # Swap the coordinate order
                walk = [(x, y) for y, x in walk]

                # Create ImageJ ROI
                roi = ImagejRoi()
                roi.version = 228
                roi.roitype = ROI_TYPE.POLYLINE
                roi.name = f"{component_id}_{walk_id}"
                roi.group = component_group
                roi.integer_coordinates = np.array(walk, dtype=np.int32)
                roi.n_coordinates = len(roi.integer_coordinates)

                rois.append(roi)

        roiwrite(
            os.path.join(self.output_dir, f'{self.img_name}_lin_decomp_rois.zip'),
            rois,
            mode='w'
        )