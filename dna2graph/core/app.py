import os
import pickle

import cv2
import numpy as np

from dna2graph.constants import TRAINED_SUFFIX
from dna2graph.core.segmentation import Segmenter
from dna2graph.core.correction import Corrector
from dna2graph.core.export import Exporter
from dna2graph.core.image_utils import (
    convert_16bit_to_8bit,
    detect_uniform_regions,
    enhance_contrast
)

CONTRAST_CLIP_PERCENT = 0.035

CACHE_DIR = '.cache'
CACHED_GRAPH = '.cached_graph.pkl'


def pickle_object(object, filename):
    '''
    Save an object to a pickle file.
    '''
    with open(filename, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)


def depickle_object(object, filename):
    '''
    Load an object from a pickle file.
    '''
    with open(filename, 'rb') as f:
        object = pickle.load(f)

    return object


def validate_img(img, img_path):
    '''
    Validate the image to ensure it is grayscale and has a valid data type.
    '''
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    if img.ndim != 2:
        raise ValueError(f"Image is not grayscale: {img_path}")
    
    if img.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Unsupported data type for image at {img_path}.")
    
    if img.shape[0] > 65535 or img.shape[1] > 65535:
        raise ValueError(f"Image is too large: {img_path}")
    
    return True


def validate_mask(mask, img_path):
    '''
    Validate the mask to ensure it is not empty.
    '''
    if not np.any(mask):
        raise ValueError(f"Empty mask for image: {img_path}")
    
    return True


class DNA2Graph:
    def __init__(
        self,
        output_root_dir,
        config,
        trained_seg,
        save_graph,
        save_mask,
        save_report,
        save_segmentation_rois,
        save_bbox_rois,
        save_lin_decomp_rois,
        clean_cache,
        stop_event=None
    ):
        self.output_root_dir = output_root_dir
        self.config = config

        suffix = TRAINED_SUFFIX if trained_seg else ''
        self.segmenter = Segmenter(
            trained_seg,
            **config[f'segmenter{suffix}'],
            stop_event=stop_event
        )
        self.corrector = Corrector(
            trained_seg,
            **config[f'corrector{suffix}'],
            stop_event=stop_event
        )
        self.exporter = Exporter(
            output_root_dir=output_root_dir,
            save_graph=save_graph,
            save_mask=save_mask,
            save_report=save_report,
            save_segmentation_rois=save_segmentation_rois,
            save_bbox_rois=save_bbox_rois,
            save_lin_decomp_rois=save_lin_decomp_rois
        )

        self.clean_cache = clean_cache

    def forward(self, img_path):
        '''
        Main application logic.
        '''
        # Get the image file name
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0]

        # Create output directory for the image
        output_dir = os.path.join(self.output_root_dir, img_name)
        os.makedirs(output_dir, exist_ok=True)

        # Create cache directory for the image
        cache_dir = os.path.join(output_dir, CACHE_DIR)
        os.makedirs(cache_dir, exist_ok=True)

        # If clean_cache is True, remove the cached graph if it exists
        cached_graph = os.path.join(cache_dir, CACHED_GRAPH)
        if self.clean_cache and os.path.exists(cached_graph):
            os.remove(cached_graph)

        # Load cached graph if available, else process and cache
        try:
            with open(cached_graph, 'rb') as f:
                # Load the graph from cache
                G = pickle.load(f)
        except Exception:
            G = self._process_img(img_path)
            with open(cached_graph, 'wb') as f:
                # Save the graph to cache
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Prune the graph to remove small components, enhance connectivity, and label components
        G = self.corrector.finalize(G)

        # Save the analysis results
        self.exporter.load_graph(G, img_name)
        self.exporter.save()

    def _process_img(self, img_path):
        '''
        Process an image to generate a graph representation.
        '''
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        validate_img(img, img_path)

        # Compute the validity mask to detect uniform regions
        validity_mask = detect_uniform_regions(
            img,
            **self.config['validity_mask']
        )

        # Enhance the contrast of the image
        img = enhance_contrast(
            img,
            clip_percent=CONTRAST_CLIP_PERCENT,
            validity_mask=validity_mask
        )

        # If 16-bit image, convert to 8-bit
        if img.dtype == np.uint16:
            img = convert_16bit_to_8bit(img)

        # Compute raw segmentation output
        mask = self.segmenter.segment(img, validity_mask)
        validate_mask(mask, img_path)

        # Correct in the graph domain
        G = self.corrector.correct(mask, validity_mask)

        return G