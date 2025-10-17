import typing
import cupy as cp
import numpy as np
from tqdm import tqdm

def get_positions(start_point: typing.Tuple[int, int, int],
                  total_vol: typing.Tuple[int, int, int],
                  vol: typing.Tuple[int, int, int],
                  shape: typing.Tuple[int, int, int],
                  overlap: typing.Tuple[int, int, int],
                  axis: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """ Calculates the starting positions of the subvolumes

    This breaks a large volume that is too large to be analysed in one go into smaller subvolumes.
    If any one of the values in overlap is greater than 0, the subvolumes will include regions outside the region of
    interest. This helps to minimize any edge effects. When the subvolumes are merged back these extra regions will
    be removed.

    Args:
        start_point (typing.Tuple[int, int, int]): starting position of the region of interest in the image volume
        total_vol (typing.Tuple[int, int, int]): total size of the region of interest
        vol (typing.Tuple[int, int, int]): maximum volume size that can be analysed at one go
        shape (typing.Tuple[int, int, int]): size of the image volume
        overlap (typing.Tuple[int, int, int]): amount of overlap between adjacent subvolumes
        axis (int): axis to calculate the positions from

    Returns:
        position (List[Tuple[int, int]]): starting and ending position along the specified axis to generate the subvolume
        valid_position (List[Tuple[int, int]]): starting and ending position of the final displacement field that should be merged
        valid_vol (List[Tuple[int, int]]): starting and ending position of the subvolume displacement field that should be merged
    """
    q, r = divmod(total_vol[axis], vol[axis] - overlap[axis])
    position = []
    valid_vol = []
    valid_position = []

    count = q + (r != 0)
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

        _end = min((end, shape[axis], start_point[axis] + total_vol[axis] + overlap[axis] // 2))
        valid_end = min((end - overlap[axis] // 2 - start_point[axis], total_vol[axis]))

        end_valid = valid_end - valid_start + start_valid

        position.append((_start, _end))
        valid_position.append((valid_start, valid_end))
        valid_vol.append((start_valid, end_valid))

    return position, valid_position, valid_vol

def farneback_3d(image1, image2, iters: int, num_levels: int,
                 scale: float = 0.5, spatial_size: int = 9, sigma_k: float = 0.15,
                 filter_type: str = "box", filter_size: int = 5,
                 presmoothing: int = None, threadsperblock: typing.Tuple[int, int, int] = (8, 8, 8)):
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
        threadsperblock (typing.Tuple[int, int, int]): Defines the number of cuda threads. Defaults to (8, 8, 8)

    Returns:
        vx (cuda array): array containing the displacements in the x direction
        vy (cuda array): array containing the displacements in the y direction
        vz (cuda array): array containing the displacements in the z direction
        confidence (cuda array): array containing the calculated confidence of the Farneback algorithm
    """
    assert filter_type.lower() in ["gaussian", "box"]
    if filter_type.lower() == "gaussian":
        def filter_fn(x):
            return cupyx.scipy.ndimage.gaussian_filter(x, filter_size / 2 * 0.3)
    elif filter_type.lower() == "box":
        def filter_fn(x):
            return cupyx.scipy.ndimage.uniform_filter(x, size=filter_size)

    image1 = cp.asarray(image1, dtype=cp.float32)
    image2 = cp.asarray(image2, dtype=cp.float32)
    if presmoothing is not None:
        image1 = cupyx.scipy.ndimage.gaussian_filter(image1, presmoothing)
        image2 = cupyx.scipy.ndimage.gaussian_filter(image2, presmoothing)

    # initialize gaussian pyramid
    gauss_pyramid_1 = {1: image1}
    gauss_pyramid_2 = {1: image2}
    true_scale_dict = {}
    for pyr_lvl in range(1, num_levels + 1):
        if pyr_lvl == 1:
            gauss_pyramid_1 = {pyr_lvl: image1}
            gauss_pyramid_2 = {pyr_lvl: image2}
        else:
            gauss_pyramid_1[pyr_lvl], true_scale_dict[pyr_lvl] = gaussian_pyramid_3d(gauss_pyramid_1[pyr_lvl - 1],
                                                                                     sigma=1, scale=scale)
            gauss_pyramid_2[pyr_lvl], _ = gaussian_pyramid_3d(gauss_pyramid_2[pyr_lvl - 1], sigma=1, scale=scale)

    if type(iters) != list:
        iters = [iters] * num_levels

    # Pyr code
    for lvl in range(num_levels, 0, -1):
        # print("Currently working on pyramid level: {}".format(lvl))
        lvl_image_1 = gauss_pyramid_1[lvl]
        lvl_image_2 = gauss_pyramid_2[lvl]

        if lvl == num_levels:
            # initialize velocities
            vx = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
            vy = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
            vz = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
        else:
            # check if nan values are present
            vx[cp.isnan(vx)] = 0
            vy[cp.isnan(vy)] = 0
            vz[cp.isnan(vz)] = 0

            vx[cp.abs(confidence) > 1] = 0
            vy[cp.abs(confidence) > 1] = 0
            vz[cp.abs(confidence) > 1] = 0
            del confidence

            vx = 1 / true_scale_dict[lvl + 1][2] * imresize_3d(vx, scale=true_scale_dict[lvl + 1])
            vy = 1 / true_scale_dict[lvl + 1][1] * imresize_3d(vy, scale=true_scale_dict[lvl + 1])
            vz = 1 / true_scale_dict[lvl + 1][0] * imresize_3d(vz, scale=true_scale_dict[lvl + 1])

        b1_0, b1_1, b1_2, a1_00, a1_01, a1_02, a1_11, a1_12, a1_22 = make_abc_fast(lvl_image_1, spatial_size,
                                                                                   sigma_k=sigma_k)
        b2_0, b2_1, b2_2, a2_00, a2_01, a2_02, a2_11, a2_12, a2_22 = make_abc_fast(lvl_image_2, spatial_size,
                                                                                   sigma_k=sigma_k)

        border = cp.asarray([0.14, 0.14, 0.4472, 0.4472, 0.4472, 1], dtype=cp.float32)
        shape = vx.shape
        h0 = cp.zeros(shape, dtype=cp.float32)
        h1 = cp.zeros(shape, dtype=cp.float32)
        h2 = cp.zeros(shape, dtype=cp.float32)
        g00 = cp.zeros(shape, dtype=cp.float32)
        g01 = cp.zeros(shape, dtype=cp.float32)
        g02 = cp.zeros(shape, dtype=cp.float32)
        g11 = cp.zeros(shape, dtype=cp.float32)
        g12 = cp.zeros(shape, dtype=cp.float32)
        g22 = cp.zeros(shape, dtype=cp.float32)

        for i in range(iters[lvl - 1]):
            blockspergrid_z = math.ceil(shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(shape[1] / threadsperblock[1])
            blockspergrid_x = math.ceil(shape[2] / threadsperblock[2])
            blockspergrid = (blockspergrid_z, blockspergrid_y, blockspergrid_x)

            update_matrices[blockspergrid, threadsperblock](b1_0, b1_1, b1_2, a1_00, a1_01, a1_02, a1_11, a1_12, a1_22,
                                                            b2_0, b2_1, b2_2, a2_00, a2_01, a2_02, a2_11, a2_12, a2_22,
                                                            vx, vy, vz, border,
                                                            h0, h1, h2, g00, g01, g02, g11, g12, g22)
            cp.cuda.Stream.null.synchronize()

            h0 = filter_fn(h0)
            h1 = filter_fn(h1)
            h2 = filter_fn(h2)
            g00 = filter_fn(g00)
            g01 = filter_fn(g01)
            g02 = filter_fn(g02)
            g11 = filter_fn(g11)
            g12 = filter_fn(g12)
            g22 = filter_fn(g22)
            cp.cuda.Stream.null.synchronize()

            update_flow[blockspergrid, threadsperblock](h0, h1, h2, g00, g01, g02, g11, g12, g22, vx, vy, vz)
            cp.cuda.Stream.null.synchronize()

            if i == iters[lvl - 1] - 1:
                confidence = cp.zeros(vx.shape, dtype=cp.float32)
                calculate_confidence[blockspergrid, threadsperblock](h0, h1, h2, g00, g01, g02, g11, g12, g22, vx, vy, vz,
                                                                     confidence)

            cp.cuda.Stream.null.synchronize()
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
                 presmoothing: int = None,
                 device_id: int = 0,
                 ):
        self.iters = iters
        self.num_levels = num_levels
        self.scale = scale
        self.spatial_size = spatial_size

        self.presmoothing = presmoothing
        self.sigma_k = sigma_k
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.device_id = device_id

    def calculate_flow(self, image1: np.ndarray, image2: np.ndarray,
                       start_point: typing.Tuple[int, int, int] = (0, 0, 0),
                       total_vol: typing.Optional[typing.Tuple[int, int, int]] = None,
                       sub_volume: typing.Tuple[int, int, int] = (256, 256, 256),
                       overlap: typing.Tuple[int, int, int] = (64, 64, 64),
                       threadsperblock: typing.Tuple[int, int, int] = (8, 8, 8),
                       ):
        """ Calculates the displacement across image1 and image2 using the 3D Farneback two frame algorithm

        Args:
            image1 (np.ndarray): first image
            image2 (np.ndarray): second image
            start_point (typing.Tuple[int, int, int]): starting position of the region of interest in the image volume.
                Defaults to (0, 0, 0)
            total_vol (typing.Optional[typing.Tuple[int, int, int]]): total size of the region of interest. Defaults to None
            sub_volume (typing.Tuple[int, int, int]): maximum volume size that can be analysed at one go.
                Defaults to (256, 256, 256)
            overlap (typing.Tuple[int, int, int]): amount of overlap between adjacent subvolumes.
                Defaults to (64, 64, 64)
            threadsperblock (typing.Tuple[int, int, int]): Defines the number of cuda threads.
                Defaults to (8, 8, 8)

        Returns:
            output_vz (np.ndarray): array containing the displacements in the x direction
            output_vy (np.ndarray): array containing the displacements in the y direction
            output_vx (np.ndarray): array containing the displacements in the z direction
            output_confidence (np.ndarray): array containing the calculated confidence of the Farneback algorithm
        """
        if total_vol is None:
            total_vol = image1.shape - np.array(start_point)

        print("Running 3D Farneback optical flow with the following parameters:")
        print(
            f"Iters: {self.iters} | Levels: {self.num_levels} | Scale: {self.scale} | Kernel: {self.spatial_size} | Filter: {self.filter_type}-{self.filter_size} | Presmoothing: {self.presmoothing}",
            flush=True)

        output_vx = np.zeros(total_vol, dtype=np.float32)
        output_vy = np.zeros(total_vol, dtype=np.float32)
        output_vz = np.zeros(total_vol, dtype=np.float32)
        output_confidence = np.zeros(total_vol, dtype=np.float32)

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        if np.any(total_vol > sub_volume):

            shape = image1.shape
            z_position, z_valid_pos, z_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 0)
            y_position, y_valid_pos, y_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 1)
            x_position, x_valid_pos, x_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 2)

            for z_i in range(len(z_position)):
                for y_i in range(len(y_position)):
                    for x_i in tqdm(range(len(x_position)),
                                    desc=f"Item: {z_i * len(y_position) + y_i + 1}/{len(z_position) * len(y_position)}"):
                        input_image_vol_1 = image1[z_position[z_i][0]:z_position[z_i][1],
                                                   y_position[y_i][0]:y_position[y_i][1],
                                                   x_position[x_i][0]:x_position[x_i][1]].astype(np.float32)
                        input_image_vol_2 = image2[z_position[z_i][0]:z_position[z_i][1],
                                                   y_position[y_i][0]:y_position[y_i][1],
                                                   x_position[x_i][0]:x_position[x_i][1]].astype(np.float32)

                        with cp.cuda.Device(self.device_id):
                            vx, vy, vz, confidence = farneback_3d(input_image_vol_1, input_image_vol_2, self.iters,
                                                                  self.num_levels,
                                                                  scale=self.scale, spatial_size=self.spatial_size,
                                                                  sigma_k=self.sigma_k, filter_type=self.filter_type,
                                                                  filter_size=self.filter_size,
                                                                  presmoothing=self.presmoothing,
                                                                  threadsperblock=threadsperblock)

                        cp.cuda.Stream.null.synchronize()

                        vx_cpu = vx.get()
                        vy_cpu = vy.get()
                        vz_cpu = vz.get()
                        confidence_cpu = confidence.get()

                        output_vx[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                                  y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                                  x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = vx_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                                     y_valid[y_i][0]: y_valid[y_i][1],
                                                                                     x_valid[x_i][0]: x_valid[x_i][1]]
                        output_vy[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                                  y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                                  x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = vy_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                                     y_valid[y_i][0]: y_valid[y_i][1],
                                                                                     x_valid[x_i][0]: x_valid[x_i][1]]
                        output_vz[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                                  y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                                  x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = vz_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                                     y_valid[y_i][0]: y_valid[y_i][1],
                                                                                     x_valid[x_i][0]: x_valid[x_i][1]]

                        output_confidence[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                                          y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                                          x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = confidence_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                                                     y_valid[y_i][0]: y_valid[y_i][1],
                                                                                                     x_valid[x_i][0]: x_valid[x_i][1]]

                        del vx, vy, vz, confidence
                        del vx_cpu, vy_cpu, vz_cpu, confidence_cpu

                        mempool.free_all_blocks()
                        pinned_mempool.free_all_blocks()
        else:
            with cp.cuda.Device(self.device_id):
                vx, vy, vz, confidence = farneback_3d(image1, image2,
                                                      iters=self.iters,
                                                      num_levels=self.num_levels,
                                                      scale=self.scale, spatial_size=self.spatial_size,
                                                      sigma_k=self.sigma_k,
                                                      filter_type=self.filter_type, filter_size=self.filter_size,
                                                      presmoothing=self.presmoothing,
                                                      threadsperblock=threadsperblock)
            cp.cuda.Stream.null.synchronize()

            output_vx = vx.get()
            output_vy = vy.get()
            output_vz = vz.get()
            output_confidence = confidence.get()

        return output_vz, output_vy, output_vx, output_confidence
