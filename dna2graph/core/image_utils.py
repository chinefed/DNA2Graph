import cv2
import numpy as np
from skimage.morphology import skeletonize


def convert_16bit_to_8bit(img):
    '''
    Convert a 16-bit image to an 8-bit image by scaling.
    '''
    img = img.astype(np.float32)
    scale = np.float32(255.0 / 65535.0)

    np.multiply(img, scale, out=img)
    np.clip(img, 0, 255.0, out=img)

    return img.astype(np.uint8)


def get_histogram_percentiles(img, q_list, validity_mask=None):
    '''
    Computes percentiles of the histogram of an image.
    '''
    if img.ndim != 2:
        raise ValueError("Image must be 2D (grayscale).")
    
    dtype = img.dtype
    try:
        max_value = np.iinfo(dtype).max
    except:
        raise ValueError(f"Unsupported image dtype: {dtype}")
    
    for q in q_list:
        if not (0 <= q <= 100):
            raise ValueError(f"Quantile {q} is not in [0, 100].")
        
    q_list = np.array(q_list) / 100.0  # Convert to [0, 1] range
        
    if validity_mask is not None:
        if validity_mask.shape != img.shape:
            raise ValueError("Mask must match image shape.")
        if np.count_nonzero(validity_mask) == 0:
            raise ValueError("Mask excludes all pixels.")

    # Compute CDF the image
    hist = cv2.calcHist(
        [img],
        [0],
        validity_mask,
        [max_value + 1],
        [0, max_value + 1]
    ).flatten()
    hist = hist / hist.sum()
    cdf = np.cumsum(hist)

    # Percentiles
    percentiles = np.searchsorted(cdf, q_list)

    return percentiles


def enhance_contrast(img, clip_percent, validity_mask=None):
    '''
    Enhance the contrast of an image using percentile normalization.
    '''
    dtype = img.dtype
    try:
        max_value = np.iinfo(dtype).max
    except:
        raise ValueError(f"Unsupported image dtype: {dtype}")
    
    threshold = clip_percent / 2
    low, high = get_histogram_percentiles(
        img,
        [threshold, 100 - threshold],
        validity_mask
    )

    img = img.astype(np.float32)
    low = np.float32(low)
    scale = np.float32(max_value / (high - low))

    np.subtract(img, low, out=img)
    np.multiply(img, scale, out=img)
    np.clip(img, 0, max_value, out=img)

    return img.astype(dtype)


def match_histogram(img, target_cdf, validity_mask=None):
    '''
    Match the histogram of an image to a target histogram.
    '''
    # Compute CDF the image
    hist = cv2.calcHist([img], [0], validity_mask, [256], [0,256]).flatten()
    hist = hist / hist.sum()
    cdf = np.cumsum(hist)

    # Create a lookup table to map pixel values
    lookup_table = np.zeros(256, dtype=np.uint8)
    for pixel in range(256):
        # Find the target pixel value with the closest CDF
        lookup_table[pixel] = min(np.searchsorted(target_cdf, cdf[pixel]), 255)

    # Apply the mapping
    img = cv2.LUT(img, lookup_table)

    return img


def detect_uniform_regions(img, block_size, tolerance, safety_margin):
    '''
    Detects uniform regions in an image, which should be discarded during segmentation
    '''
    # Compute the threshold based on the image range and tolerance
    img_range = img.max() - img.min()
    threshold = img_range * tolerance
    
    # At each pixel, compute the abs diff between the max and min pixel values
    # in a block of size block_size x block_size centered at that pixel.
    kernel = np.ones((block_size, block_size), img.dtype)
    max_img = cv2.dilate(img, kernel)
    min_img = cv2.erode(img, kernel)
    diff = max_img - min_img

    # Create a mask where the difference exceeds the threshold.
    # Pixels for which the difference is lower than the threshold
    # are considered in a uniform region and should be discarded.
    validity_mask = (diff >= threshold).astype(np.uint8) * 255

    # Erode the mask with a structring element of size 
    # block_size * safety_level to add a safety margin.
    k = safety_margin * 2 + 1
    validity_mask = cv2.erode(validity_mask, np.ones((k, k), np.uint8))

    return validity_mask


def high_boost(img, radius, amount):
    '''
    Apply high boost filtering to an image.
    '''
    img = img.astype(np.float32)
    alpha = np.float32(1 + amount)
    beta = np.float32(-amount)

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius)

    np.multiply(img, alpha, out=img)
    np.multiply(blurred, beta, out=blurred)
    np.add(img, blurred, out=img)
    np.clip(img, 0, 255.0, out=img)
    
    return img.astype(np.uint8)


def hysteresis_thresholding(img, low, high):
    '''
    Apply hysteresis thresholding to an image.
    '''
    _, weak = cv2.threshold(img, low, 255, cv2.THRESH_BINARY)
    _, mask = cv2.threshold(img, high, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    while True:
        dilated = cv2.bitwise_and(cv2.dilate(mask, kernel), weak)
        if cv2.countNonZero(cv2.bitwise_xor(mask, dilated)) == 0:
            break
        mask = dilated

    return mask


def filter_mask(mask, min_area):
    '''
    Filter connected components in a binary mask based on minimum area.
    '''
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    valid_labels = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] > min_area) + 1
    filtered_mask = np.isin(labels, valid_labels).astype(np.uint8) * 255

    return filtered_mask


def get_distance_weighted_skeleton(mask):
    '''
    Get a distance-weighted skeleton from a binary mask.
    '''
    # Get weighted skeleton with distance weights
    skeleton = skeletonize(mask > 0).astype(np.uint8)

    # Compute the distance transform to weight the skeleton
    # We round up the distance values to the nearest integer
    # (so that every non-zero pixel in the skeleton has a non-zero weight),
    # clip the values to [0, 255] and convert to uint8.
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    np.ceil(distance, out=distance)
    np.clip(distance, 0, 255, out=distance)
    distance = distance.astype(np.uint8)

    # Multiply the skeleton by the distance transform to weight the skeleton
    skeleton *= distance

    return skeleton


def get_boundary_skeleton_pixels(skeleton, validity_mask, tolerance):
    '''
    Identify skeleton pixels that are on the boundary of valid regions.
    '''
    # Ensure the skeleton is binary and uint8
    skeleton = (skeleton > 0).astype(np.uint8) * 255
    # Ensure the validity_mask is binary and uint8
    validity_mask = (validity_mask > 0).astype(np.uint8) * 255

    # Kernel size for erosion depends on the boundary tolerance
    k = tolerance * 2 + 1

    # Define a square kernel for morphological erosion
    kernel = np.ones((k, k), np.uint8)  

    # Erode the validity mask to shrink valid regions inward,
    # effectively removing the boundary pixels from the valid area
    eroded = cv2.erode(
        validity_mask,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # XOR between original mask and eroded mask gives the boundary region
    # Then intersect (AND) with skeleton to keep only skeleton boundary pixels
    boundary_pixels = cv2.bitwise_and(
        cv2.bitwise_xor(validity_mask, eroded),
        skeleton
    )

    # Extract coordinates of boundary pixels
    coords = [(int(y), int(x)) for y, x in np.argwhere(boundary_pixels)]

    return coords
