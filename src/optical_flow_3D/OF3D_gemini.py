import math
from typing import Optional # Keep for older Python compatibility if needed, but | None is preferred

import cupy as cp
import numpy as np
from tqdm import tqdm

# Use modern, specific type hints for arrays
from numpy.typing import NDArray
from cupy.typing import NDArray as CPArray
# Use the stable ndimage API
from cupy import ndimage as cp_ndimage

# (Assuming these functions are in another file and have also been updated)
from .helpers.farneback_functions import make_abc_fast, update_matrices, update_flow, calculate_confidence
from .helpers.helpers import gaussian_pyramid_3d, imresize_3d

def get_positions(start_point: tuple[int, int, int],
                  total_vol: tuple[int, int, int],
                  vol: tuple[int, int, int],
                  shape: tuple[int, int, int],
                  overlap: tuple[int, int, int],
                  axis: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """ Calculates the starting positions of the subvolumes

    This breaks a large volume that is too large to be analysed in one go into smaller subvolumes.
    If any one of the values in overlap is greater than 0, the subvolumes will include regions outside the region of
    interest. This helps to minimize any edge effects. When the subvolumes are merged back these extra regions will
    be removed.

    Args:
        start_point (tuple[int, int, int]): starting position of the region of interest in the image volume
        total_vol (tuple[int, int, int]): total size of the region of interest
        vol (tuple[int, int, int]): maximum volume size that can be analysed at one go
        shape (tuple[int, int, int]): size of the image volume
        overlap (tuple[int, int, int]): amount of overlap between adjacent subvolumes
        axis (int): axis to calculate the positions from

    Returns:
        position (list[tuple[int, int]]): starting and ending position along the specified axis to generate the subvolume
        valid_position (list[tuple[int, int]]): starting and ending position of the final displacement field that should be merged
        valid_vol (list[tuple[int, int]]): starting and ending position of the subvolume displacement field that should be merged
    """
    q, r = divmod(total_vol[axis], vol[axis] - overlap[axis])
    position, valid_vol, valid_position = [], [], []

    count = q + (1 if r > 0 else 0)
    end, valid_end = 0, 0  # Initialize for loop
    for i in range(count):
        if i == 0:
            start = start_point[axis] - overlap[axis] // 2
            valid_start = 0
        else:
            start = end - overlap[axis]
            valid_start = valid_end
        end = start + vol[axis]

        _start = max(start, 0)
        start_diff = start - _start
        start_valid = overlap[axis] // 2 + start_diff

        _end = min(end, shape[axis], start_point[axis] + total_vol[axis] + overlap[axis] // 2)
        valid_end = min(end - overlap[axis] // 2 - start_point[axis], total_vol[axis])

        end_valid = valid_end - valid_start + start_valid

        position.append((_start, _end))
        valid_position.append((valid_start, valid_end))
        valid_vol.append((start_valid, end_valid))

    return position, valid_position, valid_vol

def farneback_3d(image1: CPArray, image2: CPArray, iters: int, num_levels: int,
                 scale: float = 0.5, spatial_size: int = 9, sigma_k: float = 0.15,
                 filter_type: str = "box", filter_size: int = 5,
                 presmoothing: float | None = None,
                 threadsperblock: tuple[int, int, int] = (8, 8, 8)
                ) -> tuple[CPArray, CPArray, CPArray, CPArray]:
    """ Estimates the displacement across image1 and image2 using the 3D Farneback two frame algorithm

    Args:
        image1 (cuda array): first image
        image2 (cuda array): second image
        iters (int): number of iterations
        num_levels (int): number of pyramid levels
        scale (float): Scaling factor used to generate the pyramid levels. Defaults to 0.5
        spatial_size (int): size of the support used in the calculation of the standard deviation of the Gaussian
            applicability. Defaults to 9.
        sigma_k (float): scaling factor used to calculate the standard deviation of the Gaussian applicability. The
            formula to calculate sigma is sigma_k*(spatial_size - 1). Defaults to 0.15.
        filter_type (int): Defines the type of filter used to average the calculated matrices. Defaults to "box"
        filter_size (int): Size of the filter used to average the matrices. Defaults to 5
        presmoothing (int): Standard deviation used to perform Gaussian smoothing of the images. Defaults to None
        threadsperblock (tuple[int, int, int]): Defines the number of cuda threads. Defaults to (8, 8, 8)

    Returns:
        vx (cuda array): array containing the displacements in the x direction
        vy (cuda array): array containing the displacements in the y direction
        vz (cuda array): array containing the displacements in the z direction
        confidence (cuda array): array containing the calculated confidence of the Farneback algorithm
    """
    assert filter_type.lower() in ["gaussian", "box"]
    if filter_type.lower() == "gaussian":
        filter_fn = lambda x: cp_ndimage.gaussian_filter(x, sigma=filter_size / 2 * 0.3)
    else:  # "box"
        filter_fn = lambda x: cp_ndimage.uniform_filter(x, size=filter_size)

    image1 = cp.asarray(image1, dtype=cp.float32)
    image2 = cp.asarray(image2, dtype=cp.float32)
    if presmoothing is not None:
        image1 = cp_ndimage.gaussian_filter(image1, presmoothing)
        image2 = cp_ndimage.gaussian_filter(image2, presmoothing)

    # initialize gaussian pyramid
    gauss_pyramid_1 = [image1]
    gauss_pyramid_2 = [image2]
    true_scales = []
    for _ in range(1, num_levels):
        im1, scale1 = gaussian_pyramid_3d(gauss_pyramid_1[-1], sigma=1, scale=scale)
        im2, _ = gaussian_pyramid_3d(gauss_pyramid_2[-1], sigma=1, scale=scale)
        gauss_pyramid_1.append(im1)
        gauss_pyramid_2.append(im2)
        true_scales.append(scale1)

    if not isinstance(iters, list):
        iters = [iters] * num_levels

    vx, vy, vz, confidence = None, None, None, None

    # Pyr code
    for lvl in range(num_levels - 1, -1, -1):
        # print("Currently working on pyramid level: {}".format(lvl))
        lvl_image_1 = gauss_pyramid_1[lvl]
        lvl_image_2 = gauss_pyramid_2[lvl]

        if lvl == num_levels - 1:
            # initialize velocities
            vx = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
            vy = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
            vz = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
        else:
            # Upsample and scale flow from previous level
            scale_z, scale_y, scale_x = true_scales[lvl]
            vx = imresize_3d(vx, scale=(scale_z, scale_y, scale_x)) / scale_x
            vy = imresize_3d(vy, scale=(scale_z, scale_y, scale_x)) / scale_y
            vz = imresize_3d(vz, scale=(scale_z, scale_y, scale_x)) / scale_z
            
            # Handle potential NaNs from upsampling
            cp.nan_to_num(vx, copy=False)
            cp.nan_to_num(vy, copy=False)
            cp.nan_to_num(vz, copy=False)

        b1_params = make_abc_fast(lvl_image_1, spatial_size, sigma_k=sigma_k)
        b2_params = make_abc_fast(lvl_image_2, spatial_size, sigma_k=sigma_k)

        border = cp.asarray([0.14, 0.14, 0.4472, 0.4472, 0.4472, 1], dtype=cp.float32)
        shape = vx.shape
        h = [cp.zeros(shape, dtype=cp.float32) for _ in range(3)]
        g = [cp.zeros(shape, dtype=cp.float32) for _ in range(6)]

        for i in range(iters[lvl]):
            blockspergrid = (math.ceil(shape[0] / threadsperblock[0]),
                             math.ceil(shape[1] / threadsperblock[1]),
                             math.ceil(shape[2] / threadsperblock[2]))

            update_matrices[blockspergrid, threadsperblock](*b1_params, *b2_params, vx, vy, vz, border, *h, *g)

            h = [filter_fn(arr) for arr in h]
            g = [filter_fn(arr) for arr in g]

            update_flow[blockspergrid, threadsperblock](*h, *g, vx, vy, vz)

    # Final confidence calculation at the finest level
    confidence = cp.zeros(vx.shape, dtype=cp.float32)
    calculate_confidence[blockspergrid, threadsperblock](*h, *g, vx, vy, vz, confidence)

    return vx, vy, vz, confidence


class Farneback3D:
    """Farneback3D class used to instantiate the algorithm with its parameters.

    Args:
        iters (int): number of iterations. Defaults to 5
        num_levels (int): number of pyramid levels. Defaults to 5
        scale (float): Scaling factor used to generate the pyramid levels. Defaults to 0.5
        spatial_size (int): size of the support used in the calculation of the standard deviation of the Gaussian
            applicability. Defaults to 9.
        sigma_k (float): scaling factor used to calculate the standard deviation of the Gaussian applicability. The
            formula to calculate sigma is sigma_k*(spatial_size - 1). Defaults to 0.15.
        filter_type (str): Defines the type of filter used to average the calculated matrices. Defaults to "box"
        filter_size (int): Size of the filter used to average the matrices. Defaults to 21
        presmoothing (int): Standard deviation used to perform Gaussian smoothing of the images. Defaults to None
        device_id (int): Device id of the GPU. Defaults to 0
    """

    def __init__(self,
                 iters: int = 5,
                 num_levels: int = 5,
                 scale: float = 0.5,
                 spatial_size: int = 9,
                 sigma_k: float = 0.15,
                 filter_type: str = "box",
                 filter_size: int = 21,
                 presmoothing: float | None = None,
                 device_id: int = 0):
        self.iters = iters
        self.num_levels = num_levels
        self.scale = scale
        self.spatial_size = spatial_size
        self.presmoothing = presmoothing
        self.sigma_k = sigma_k
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.device_id = device_id

    def calculate_flow(self,
                       image1: NDArray,
                       image2: NDArray,
                       start_point: tuple[int, int, int] = (0, 0, 0),
                       total_vol: Optional[tuple[int, int, int]] = None,
                       sub_volume: tuple[int, int, int] = (256, 256, 256),
                       overlap: tuple[int, int, int] = (64, 64, 64),
                       threadsperblock: tuple[int, int, int] = (8, 8, 8)
                      ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """ Calculates the displacement across image1 and image2 using the 3D Farneback two frame algorithm

        Args:
            image1 (np.ndarray): first image
            image2 (np.ndarray): second image
            start_point (tuple[int, int, int]): starting position of the region of interest in the image volume.
                Defaults to (0, 0, 0)
            total_vol (Optional[tuple[int, int, int]]): total size of the region of interest. Defaults to None
            sub_volume (tuple[int, int, int]): maximum volume size that can be analysed at one go.
                Defaults to (256, 256, 256)
            overlap (tuple[int, int, int]): amount of overlap between adjacent subvolumes.
                Defaults to (64, 64, 64)
            threadsperblock (tuple[int, int, int]): Defines the number of cuda threads.
                Defaults to (8, 8, 8)

        Returns:
            output_vx (np.ndarray): array containing the displacements in the x direction
            output_vy (np.ndarray): array containing the displacements in the y direction
            output_vz (np.ndarray): array containing the displacements in the z direction
            output_confidence (np.ndarray): array containing the calculated confidence of the Farneback algorithm
        """
        if total_vol is None:
            total_vol = tuple(s - sp for s, sp in zip(image1.shape, start_point))

        print("Running 3D Farneback optical flow with the following parameters:")
        print(
            f"Iters: {self.iters} | Levels: {self.num_levels} | Scale: {self.scale} | "
            f"Kernel: {self.spatial_size} | Filter: {self.filter_type}-{self.filter_size} | "
            f"Presmoothing: {self.presmoothing}",
            flush=True)
        
        # Check if chunking is necessary
        if np.any(np.array(total_vol) > np.array(sub_volume)):
            shape = image1.shape
            z_pos, z_valid_pos, z_valid_vol = get_positions(start_point, total_vol, sub_volume, shape, overlap, 0)
            y_pos, y_valid_pos, y_valid_vol = get_positions(start_point, total_vol, sub_volume, shape, overlap, 1)
            x_pos, x_valid_pos, x_valid_vol = get_positions(start_point, total_vol, sub_volume, shape, overlap, 2)
            
            output_vx = np.zeros(total_vol, dtype=np.float32)
            output_vy = np.zeros(total_vol, dtype=np.float32)
            output_vz = np.zeros(total_vol, dtype=np.float32)
            output_confidence = np.zeros(total_vol, dtype=np.float32)
            
            num_chunks = len(z_pos) * len(y_pos)

            for z_i in range(len(z_pos)):
                for y_i in range(len(y_pos)):
                    for x_i in tqdm(range(len(x_pos)), desc=f"Processing chunk row {z_i * len(y_pos) + y_i + 1}/{num_chunks}"):
                        in_slice = np.s_[z_pos[z_i][0]:z_pos[z_i][1], y_pos[y_i][0]:y_pos[y_i][1], x_pos[x_i][0]:x_pos[x_i][1]]
                        out_slice = np.s_[z_valid_pos[z_i][0]:z_valid_pos[z_i][1], y_valid_pos[y_i][0]:y_valid_pos[y_i][1], x_valid_pos[x_i][0]:x_valid_pos[x_i][1]]
                        valid_vol_slice = np.s_[z_valid_vol[z_i][0]:z_valid_vol[z_i][1], y_valid_vol[y_i][0]:y_valid_vol[y_i][1], x_valid_vol[x_i][0]:x_valid_vol[x_i][1]]
                        
                        sub_image1 = image1[in_slice]
                        sub_image2 = image2[in_slice]

                        with cp.cuda.Device(self.device_id):
                            vx, vy, vz, confidence = farneback_3d(
                                sub_image1, sub_image2, self.iters, self.num_levels,
                                scale=self.scale, spatial_size=self.spatial_size,
                                sigma_k=self.sigma_k, filter_type=self.filter_type,
                                filter_size=self.filter_size, presmoothing=self.presmoothing,
                                threadsperblock=threadsperblock
                            )
                            # Stitch results back into the main arrays
                            output_vx[out_slice] = cp.asnumpy(vx[valid_vol_slice])
                            output_vy[out_slice] = cp.asnumpy(vy[valid_vol_slice])
                            output_vz[out_slice] = cp.asnumpy(vz[valid_vol_slice])
                            output_confidence[out_slice] = cp.asnumpy(confidence[valid_vol_slice])

                            # Free memory after each chunk
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()
        else:
            with cp.cuda.Device(self.device_id):
                vx, vy, vz, confidence = farneback_3d(
                    image1, image2, iters=self.iters, num_levels=self.num_levels,
                    scale=self.scale, spatial_size=self.spatial_size, sigma_k=self.sigma_k,
                    filter_type=self.filter_type, filter_size=self.filter_size,
                    presmoothing=self.presmoothing, threadsperblock=threadsperblock
                )
                output_vx = cp.asnumpy(vx)
                output_vy = cp.asnumpy(vy)
                output_vz = cp.asnumpy(vz)
                output_confidence = cp.asnumpy(confidence)

        # CRITICAL FIX: Return in vx, vy, vz order, not vz, vy, vx
        return output_vx, output_vy, output_vz, output_confidence
