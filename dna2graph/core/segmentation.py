import warnings

# Filter MONAI compatibility warning
warnings.filterwarnings(
    'ignore',
    message='Using a non-tuple sequence for multidimensional indexing'
)

import cv2
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from monai.inferers import sliding_window_inference

from dna2graph.utils import get_asset_path
from dna2graph.core.filters import ParallelFilter
from dna2graph.core.image_utils import (
    get_histogram_percentiles,
    match_histogram,
    high_boost,
    hysteresis_thresholding,
    filter_mask
)
from dna2graph.core.model import Model


TARGET_CDF_FILENAME = 'target_cdf.npy'


class PreprocessingFilter(ParallelFilter):
    '''
    Preprocessing pipeline for image segmentation.
    '''
    def __init__(
        self,
        patch_size,
        p,
        bilateral_diameter=None,
        bilateral_sigma_color=None,
        bilateral_sigma_space=None,
        high_boost_radius=None,
        high_boost_amount=None,
        hysteresis_low_pct=None,
        hysteresis_high_pct=None,
        stop_event=None
    ):
        super().__init__(patch_size, p, stop_event=stop_event)
        self.bilateral_diameter = bilateral_diameter
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.high_boost_radius = high_boost_radius
        self.high_boost_amount = high_boost_amount
        self.hysteresis_low_pct = hysteresis_low_pct
        self.hysteresis_high_pct = hysteresis_high_pct

        # Load the target CDF for histogram matching
        self.target_cdf = np.load(get_asset_path(TARGET_CDF_FILENAME))

    def forward(self, img_validity_mask):
        '''
        Apply preprocessing to the image.
        '''
        # Extract the image and validity mask from the stacked array
        img = img_validity_mask[..., 0]
        validity_mask = img_validity_mask[..., 1]

        # If the validity mask is empty, return an empty mask
        if np.all(validity_mask == 0): 
            return np.zeros_like(img_validity_mask)

        # Match histogram to the target CDF
        img = match_histogram(img, self.target_cdf, validity_mask)

        # Invert the image
        np.subtract(255, img, out=img)

        # Apply bilateral filter
        if not (
            self.bilateral_diameter is None or
            self.bilateral_sigma_color is None or
            self.bilateral_sigma_space is None
        ):
            img = cv2.bilateralFilter(
                img,
                d=self.bilateral_diameter,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space
            )

        # High boost filtering
        if not (
            self.high_boost_radius is None or
            self.high_boost_amount is None
        ):
            img = high_boost(
                img,
                self.high_boost_radius,
                self.high_boost_amount
            )

        # Set unvalid pixels to 0
        img[validity_mask == 0] = 0

        # Hysteresis thresholding
        if not (
            self.hysteresis_low_pct is None or
            self.hysteresis_high_pct is None
        ):
            low, high = get_histogram_percentiles(
                img,
                [self.hysteresis_low_pct, self.hysteresis_high_pct],
                validity_mask
            )
            img  = hysteresis_thresholding(img, low, high)

        # Stack the processed image with the validity mask
        img_validity_mask = np.stack((img, validity_mask), axis=-1)

        return img_validity_mask
    

class Segmenter:
    '''
    A class to segment images.
    '''
    def __init__(
        self,
        trained_seg,
        patch_size,
        p,
        bilateral_diameter=None,
        bilateral_sigma_color=None,
        bilateral_sigma_space=None,
        high_boost_radius=None,
        high_boost_amount=None,
        hysteresis_low_pct=None,
        hysteresis_high_pct=None,
        min_area_1=None,
        closure_kernel_size=None,
        min_area_2=None,
        batch_size=None,
        overlap=None,
        threshold=None,
        stop_event=None
    ):
        self.trained_seg = trained_seg
        self.preprocessing_filter = PreprocessingFilter(
            patch_size,
            p,
            bilateral_diameter,
            bilateral_sigma_color,
            bilateral_sigma_space,
            high_boost_radius,
            high_boost_amount,
            hysteresis_low_pct,
            hysteresis_high_pct,
            stop_event=stop_event
        )

        if trained_seg:
            self.patch_size = patch_size
            self.batch_size = batch_size
            self.overlap = overlap
            self.threshold = threshold

            # Set the device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
                
            # Download/cached model from Hugging Face
            try:
                model_path = hf_hub_download(
                    repo_id='chinefed/dna2graph-models',
                    filename='segmentation/dna2graph_segmenter_v1.pth'
                )
            except:
                raise RuntimeError(
                    "Unable to download trained segmentation model."
                )

            # Load the pre-trained model
            self.model = Model(return_logits=False).to(self.device)
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(checkpoint)
            self.model.eval()

        else:
            self.min_area_1 = min_area_1
            self.min_area_2 = min_area_2
            self.closure_kernel = np.ones(
                (closure_kernel_size, closure_kernel_size),
                np.uint8
            )

    def segment(self, img, validity_mask):
        '''
        Segment the image.
        '''
        # Stack the image and validity mask
        img_validity_mask = np.stack((img, validity_mask), axis=-1)

        # Apply preprocessing
        img_validity_mask = self.preprocessing_filter.forward_parallel(
            img_validity_mask
        )

        # Extract the image from the stacked array
        img = img_validity_mask[..., 0]

        if self.trained_seg:
            # Normalize the image
            img = img / 255.0

            # Convert to torch and send to device
            img = torch.from_numpy(img).to(self.device, dtype=torch.float32)
            img = img.unsqueeze(dim=0) # add channel dimension
            img = img.unsqueeze(dim=0) # add batch dimension

            # Perform inference
            with torch.no_grad():
                img = sliding_window_inference(
                    img,
                    predictor=self.model,
                    roi_size=self.patch_size,
                    sw_batch_size=self.batch_size,
                    overlap=self.overlap
                )

            # Convert batck to numpy
            img = img.squeeze().cpu().numpy()

            # Binarize the probability map
            img = ((img > self.threshold) * 255).astype(np.uint8)

        else:
            # Filter the mask based on minimum area 1
            img = filter_mask(img, self.min_area_1)

            # Apply morphological closing
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.closure_kernel)

            # Filter the mask again based on minimum area 2
            img = filter_mask(img, self.min_area_2)

        return img


