import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

from dna2graph.utils import UserCancelledError


def init_worker(filter):
    global global_filter
    global_filter = filter


def worker(
        shm_img_name,
        shm_output_name,
        shape,
        dtype,
        y,
        x,
        patch_size,
        p
    ):
    '''
    Worker function to process a patch of the image in parallel.
    '''
    global global_filter

    # Attach to the shared memory blocks
    shm_img = shared_memory.SharedMemory(name=shm_img_name)
    shm_output = shared_memory.SharedMemory(name=shm_output_name)
    shared_img = np.ndarray(shape, dtype=dtype, buffer=shm_img.buf)
    shared_output = np.ndarray(shape, dtype=dtype, buffer=shm_output.buf)

    try:
        # We reserve p pixels on each side for padding
        h, w = shape[:2]
        h_crop = h - 2 * p
        w_crop = w - 2 * p

        # Extract the patch
        y_min_in = y - p
        y_max_in = y + patch_size + p
        x_min_in = x - p
        x_max_in = x + patch_size + p
        patch = shared_img[y_min_in:y_max_in, x_min_in:x_max_in, ...]

        # Process the patch
        patch = global_filter.forward(patch)

        # Remove the padding
        patch = patch[p:p + patch_size, p:p + patch_size, ...]

        # Cut the patch if it's a reminder patch
        # -h_crop % patch_size is equivalent to:
        # (patch_size - h_crop % patch_size) % patch_size
        is_last_y_patch = (y == h - p - patch_size)
        is_last_x_patch = (x == w - p - patch_size)
        cut_y = -h_crop % patch_size if is_last_y_patch else 0
        cut_x = -w_crop % patch_size if is_last_x_patch else 0
        patch = patch[cut_y:, cut_x:, ...]

        # Copy the result to the output array
        y_min_out = y + cut_y
        y_max_out = y + patch_size
        x_min_out = x + cut_x
        x_max_out = x + patch_size
        shared_output[y_min_out:y_max_out, x_min_out:x_max_out, ...] = patch

    finally:
        shm_img.close()
        shm_output.close()


class ParallelFilter:
    '''
    A class to apply a filter to an image.
    '''

    def __init__(self, patch_size, p, stop_event=None):
        self.patch_size = patch_size
        self.p = p
        self.stop_event = stop_event

    def forward(self, img):
        '''
        Apply the filter to the image.
        This method should be implemented by subclasses.
        '''
        raise NotImplementedError("Subclasses must implement the forward method.")

    def forward_parallel(self, img):
        '''
        Apply the filter to the image in parallel using patches.
        '''
        patch_size = self.patch_size
        p = self.p

        # We reserve p pixels on each side for padding
        h, w = img.shape[:2]
        h_crop = h - 2 * p
        w_crop = w - 2 * p

        # Compute patch grid
        y_coords = list(range(p, h - p - patch_size + 1, patch_size))
        x_coords = list(range(p, w - p - patch_size + 1, patch_size))

        # Handle reminder if h or w is not a multiple of patch size.
        # We want every region in the image to be covered by patches.
        if h_crop % patch_size > 0:
            y_coords.append(h - p - patch_size)
        if w_crop % patch_size > 0:
            x_coords.append(w - p - patch_size)

        # Create shared memory for the input
        shm_img = shared_memory.SharedMemory(create=True, size=img.nbytes)
        shared_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shm_img.buf)
        np.copyto(shared_img, img)

        # Create shared memory for the output
        shm_output = shared_memory.SharedMemory(create=True, size=img.nbytes)
        shared_output = np.ndarray(img.shape, dtype=img.dtype, buffer=shm_output.buf)
        shared_output[...] = 0 # Empty buffer

        # Build argument list
        coords = [(y, x) for y in y_coords for x in x_coords]
        args = [
            (
                shm_img.name,
                shm_output.name,
                img.shape,
                img.dtype,
                y,
                x,
                patch_size,
                p
            )
            for y, x in coords
        ]

        try:
            pool = mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(self,))
            result = pool.starmap_async(worker, args)

            while not result.ready():
                if self.stop_event and self.stop_event.is_set():
                    pool.terminate()
                    raise UserCancelledError
                time.sleep(0.2)

            try:
                result.get()
            except Exception as e:
                raise RuntimeError(
                    f"Worker raised an exception: {e}"
                )

            # Recover the output
            output = np.ndarray(img.shape, dtype=img.dtype, buffer=shm_output.buf).copy()

        finally:
            pool.close()
            pool.join()
            shm_img.close()
            shm_img.unlink()
            shm_output.close()
            shm_output.unlink()

        return output


