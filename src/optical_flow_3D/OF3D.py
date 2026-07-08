from __future__ import annotations
from typing import Tuple, List, Optional
import math
import cupy as cp
import numpy as np
from tqdm import tqdm
import cupyx.scipy.ndimage  # for filters
from numba import cuda
import numpy.typing as npt
from numba import njit, prange
import scipy

@njit(parallel=True)
def inverse(xmap, ymap, zmap, xmin=0, ymin=0, zmin=0, dist_threshold=1, eps=1e-12):
    shape = xmap.shape
    inverse_x = np.zeros_like(xmap)
    inverse_y = np.zeros_like(xmap)
    inverse_z = np.zeros_like(xmap)
    distance_total = np.zeros_like(xmap)

    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                idz = np.int32(np.round(i + zmap[i, j, k]))
                idy = np.int32(np.round(j + ymap[i, j, k]))
                idx = np.int32(np.round(k + xmap[i, j, k]))

                for zval in range(max(idz - dist_threshold, zmin), min(idz + dist_threshold, zmin + shape[0])):
                    for yval in range(max(idy - dist_threshold, ymin), min(idy + dist_threshold, ymin + shape[1])):
                        for xval in range(max(idx - dist_threshold, xmin),
                                          min(idx + dist_threshold, xmin + shape[2])):
                            distance = (zval - (i + zmap[i, j, k])) ** 2 + (yval - (j + ymap[i, j, k])) ** 2 + (
                                        xval - (k + xmap[i, j, k])) ** 2
                            inverse_distance = 1 / (distance + eps)

                            inverse_z[zval, yval, xval] += inverse_distance * i
                            inverse_y[zval, yval, xval] += inverse_distance * j
                            inverse_x[zval, yval, xval] += inverse_distance * k
                            distance_total[zval, yval, xval] += inverse_distance

    return inverse_x, inverse_y, inverse_z, distance_total

def generate_inverse_image(image, vx, vy, vz, use_gpu: bool = True, device_id: int = 0) -> np.ndarray:
    """ Uses the displacements to transform the image

    This transformed image can then be overlaid over the actual image to verify the quality of the displacements.

    Args:
        image (np.ndarray): File path to save the displacements. The extension type should be *.npz
        vx (np.ndarray): Array containing the displacements in the x direction
        vy (np.ndarray): Array containing the displacements in the y direction
        vz (np.ndarray): Array containing the displacements in the z direction
        use_gpu (bool): Option to run some part of the procedure on the gpu

    Returns:
        inverse_image (np.ndarray): transformed image using the displacemennt field
    """
    # image should be the first image that is used for the optical flow calculations
    map_x_inverse, map_y_inverse, map_z_inverse, distance_total = inverse(vx, vy, vz)

    map_x_inverse = map_x_inverse / (distance_total + 1e-12)
    map_y_inverse = map_y_inverse / (distance_total + 1e-12)
    map_z_inverse = map_z_inverse / (distance_total + 1e-12)

    if use_gpu:
        with cp.cuda.Device(device_id):
            inverse_image_gpu = cupyx.scipy.ndimage.map_coordinates(
                cp.asarray(image),
                cp.array([map_z_inverse, map_y_inverse, map_x_inverse]),
                mode="mirror")
            inverse_image = inverse_image_gpu.get()
    else:
        inverse_image = scipy.ndimage.map_coordinates(image,
                                                      np.array([map_z_inverse, map_y_inverse, map_x_inverse]),
                                                      mode="mirror")

    return inverse_image

def gaussian_kernel_1d(sigma: float, radius: int = None) -> npt.ArrayLike:
    """ Generates a 1d kernel that can be used to perform Gaussian smoothing

    Args:
        sigma (float): Standard deviation of the Gaussian kernel
        radius (int): Size of the Gaussian kernel. Final size is equal to 2*radius+1. Defaults to None

    Returns:
        output_kernel (ndarray): 1d Guassian kernel
    """
    if radius is None:
        radius = math.ceil(2 * sigma)

    output_kernel = np.mgrid[-radius:radius + 1]
    output_kernel = np.exp((-(1 / 2) * (output_kernel ** 2)) / (sigma ** 2))
    output_kernel = output_kernel / np.sum(output_kernel)

    return output_kernel

@cuda.jit
def calculate_confidence(h0, h1, h2, g00, g01, g02, g11, g12, g22,
                         vx, vy, vz, confidence):
    """ Calculates the confidence of the Farneback algorithm. Smaller values indicate that the algorithm is more confident.

        Matrices are in the format [[g00, g01, g02], and [[h0],
                                    [g01, g11, g12],      [h1],
                                    [g02, g12, g22]]      [h2]]

    Args:
        h0 (cuda array): array containing the first value of the h matrix
        h1 (cuda array): array containing the second value of the h matrix
        h2 (cuda array): array containing the third value of the h matrix
        g00 (cuda array): array containing the values of the g matrix
        g01 (cuda array): array containing the values of the g matrix
        g02 (cuda array): array containing the values of the g matrix
        g11 (cuda array): array containing the values of the g matrix
        g12 (cuda array): array containing the values of the g matrix
        g22 (cuda array): array containing the values of the g matrix

        vx (cuda array): array containing the displacements in the x direction
        vy (cuda array): array containing the displacements in the y direction
        vz (cuda array): array containing the displacements in the z direction
        confidence (cuda array): array containing the calculated confidence of the Farneback algorithm

    Returns:
        None
    """
    z, y, x = cuda.grid(3)

    depth, length, width = vx.shape

    if z < depth and y < length and x < width:
        confidence[z, y, x] = (h0[z, y, x] ** 2 + h1[z, y, x] ** 2 + h2[z, y, x] ** 2) - \
                              (vx[z, y, x] * (g00[z, y, x] * h0[z, y, x] + g01[z, y, x] * h1[z, y, x] + g02[z, y, x] * h2[
                             z, y, x]) +
                          vy[z, y, x] * (g01[z, y, x] * h0[z, y, x] + g11[z, y, x] * h1[z, y, x] + g12[z, y, x] * h2[
                                     z, y, x]) +
                          vz[z, y, x] * (g02[z, y, x] * h0[z, y, x] + g12[z, y, x] * h1[z, y, x] + g22[z, y, x] * h2[
                                     z, y, x]))

@cuda.jit
def update_flow(h0, h1, h2, g00, g01, g02, g11, g12, g22,
                vx, vy, vz):
    """ Updates the displacements using the calculated matrices.

        Matrices are in the format [[g00, g01, g02], and [[h0],
                                    [g01, g11, g12],      [h1],
                                    [g02, g12, g22]]      [h2]]

    Args:
        h0 (cuda array): array containing the first value of the h matrix
        h1 (cuda array): array containing the second value of the h matrix
        h2 (cuda array): array containing the third value of the h matrix
        g00 (cuda array): array containing the values of the g matrix
        g01 (cuda array): array containing the values of the g matrix
        g02 (cuda array): array containing the values of the g matrix
        g11 (cuda array): array containing the values of the g matrix
        g12 (cuda array): array containing the values of the g matrix
        g22 (cuda array): array containing the values of the g matrix

        vx (cuda array): array containing the displacements in the x direction
        vy (cuda array): array containing the displacements in the y direction
        vz (cuda array): array containing the displacements in the z direction

    Returns:
        None
    """
    z, y, x = cuda.grid(3)

    depth, length, width = vx.shape

    if z < depth and y < length and x < width:
        det = g00[z, y, x] * (g11[z, y, x] * g22[z, y, x] - g12[z, y, x] * g12[z, y, x]) - \
              g01[z, y, x] * (g01[z, y, x] * g22[z, y, x] - g02[z, y, x] * g12[z, y, x]) + \
              g02[z, y, x] * (g01[z, y, x] * g12[z, y, x] - g02[z, y, x] * g11[z, y, x])

        vx[z, y, x] = (h0[z, y, x] * (g11[z, y, x] * g22[z, y, x] - g12[z, y, x] * g12[z, y, x]) -
                       g01[z, y, x] * (h1[z, y, x] * g22[z, y, x] - h2[z, y, x] * g12[z, y, x]) +
                       g02[z, y, x] * (h1[z, y, x] * g12[z, y, x] - h2[z, y, x] * g11[z, y, x])) / det
        vy[z, y, x] = (g00[z, y, x] * (h1[z, y, x] * g22[z, y, x] - h2[z, y, x] * g12[z, y, x]) -
                       h0[z, y, x] * (g01[z, y, x] * g22[z, y, x] - g02[z, y, x] * g12[z, y, x]) +
                       g02[z, y, x] * (g01[z, y, x] * h2[z, y, x] - g02[z, y, x] * h1[z, y, x])) / det
        vz[z, y, x] = (g00[z, y, x] * (g11[z, y, x] * h2[z, y, x] - g12[z, y, x] * h1[z, y, x]) -
                       g01[z, y, x] * (g01[z, y, x] * h2[z, y, x] - g02[z, y, x] * h1[z, y, x]) +
                       h0[z, y, x] * (g01[z, y, x] * g12[z, y, x] - g02[z, y, x] * g11[z, y, x])) / det

def make_abc_fast(signal,
                  spatial_size: int = 9,
                  sigma_k: float = 0.15):
    """Calculates the polynomial expansion coefficients

    Args:
        signal: array containing the pixel values of the 3D image.
        spatial_size (int): size of the support used in the calculation of the standard deviation of the Gaussian
            applicability. Defaults to 9.
        sigma_k (float): scaling factor used to calculate the standard deviation of the Gaussian applicability. The
            formula to calculate sigma is sigma_k*(spatial_size - 1). Defaults to 0.15.

    Returns:
        Returns the A array in this format as it is symmetrical. This saves memory space.
        a = [[a_00, a_01, a_02],
             [a_01, a_11, a_12],
             [a_02, a_12, a_22]]

        Returns the B array in this format.
        b = [[b_0],
             [b_1],
             [b_2]]

        b_0 (cuda array): array containing the first value of the B array
        b_1 (cuda array): array containing the second value of the B array
        b_2 (cuda array): array containing the third value of the B array
        a_00 (cuda array): array containing the values of the A array
        a_01 (cuda array): array containing the values of the A array
        a_02 (cuda array): array containing the values of the A array
        a_11 (cuda array): array containing the values of the A array
        a_12 (cuda array): array containing the values of the A array
        a_22 (cuda array): array containing the values of the A array

    Raises:
        AssertionError: signal must be array with 3 dimensions
    """
    # spatial_size = spatial_size | 1 # ensure the value is odd
    if signal.ndim != 3:
        raise AssertionError("signal must be array with 3 dimensions")

    sigma = sigma_k * (spatial_size - 1)

    n = int((spatial_size - 1) / 2)
    a = np.exp(-(np.arange(-n, n + 1, dtype=np.float32) ** 2) / (2 * sigma ** 2))

    # Set up applicability and basis functions
    applicability = np.multiply.outer(np.multiply.outer(a, a), a)
    z, y, x = np.mgrid[-n:n + 1, -n:n + 1, -n:n + 1]

    basis = np.stack((np.ones(x.shape), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z), axis=3)
    nb = basis.shape[3]

    # Compute the inverse metric
    # can be shortened by only calculating those values that matter
    q = np.zeros((nb, nb), dtype=np.float32)
    for i in range(nb):
        for j in range(i, nb):
            q[i, j] = np.sum(basis[..., i] * applicability * basis[..., j])
            q[j, i] = q[i, j]

    del basis, applicability, x, y, z
    qinv = np.linalg.inv(q)

    # convolutions in z
    kernel_0 = cp.array(a)
    kernel_1 = cp.array(np.arange(-n, n + 1, dtype=np.float32) * a)
    kernel_2 = cp.array(np.arange(-n, n + 1, dtype=np.float32) ** 2 * a)

    conv_z0 = cupyx.scipy.ndimage.correlate1d(signal, kernel_0, axis=0)
    conv_z1 = cupyx.scipy.ndimage.correlate1d(signal, kernel_1, axis=0)
    conv_z2 = cupyx.scipy.ndimage.correlate1d(signal, kernel_2, axis=0)

    # convolutions in y
    conv_z0y0 = cupyx.scipy.ndimage.correlate1d(conv_z0, kernel_0, axis=1)
    conv_z0y1 = cupyx.scipy.ndimage.correlate1d(conv_z0, kernel_1, axis=1)
    conv_z0y2 = cupyx.scipy.ndimage.correlate1d(conv_z0, kernel_2, axis=1)
    del conv_z0

    conv_z1y0 = cupyx.scipy.ndimage.correlate1d(conv_z1, kernel_0, axis=1)
    conv_z1y1 = cupyx.scipy.ndimage.correlate1d(conv_z1, kernel_1, axis=1)
    del conv_z1

    conv_z2y0 = cupyx.scipy.ndimage.correlate1d(conv_z2, kernel_0, axis=1)
    del conv_z2

    # convolutions in x
    conv_z0y0x0 = cupyx.scipy.ndimage.correlate1d(conv_z0y0, kernel_0, axis=2)
    b_0 = qinv[1, 1] * cupyx.scipy.ndimage.correlate1d(conv_z0y0, kernel_1, axis=2)
    a_00 = qinv[4, 4] * cupyx.scipy.ndimage.correlate1d(conv_z0y0, kernel_2, axis=2) + qinv[4, 0] * conv_z0y0x0
    del conv_z0y0

    b_1 = qinv[2, 2] * cupyx.scipy.ndimage.correlate1d(conv_z0y1, kernel_0, axis=2)
    a_01 = qinv[7, 7] * cupyx.scipy.ndimage.correlate1d(conv_z0y1, kernel_1, axis=2) / 2
    del conv_z0y1

    a_11 = qinv[5, 5] * cupyx.scipy.ndimage.correlate1d(conv_z0y2, kernel_0, axis=2) + qinv[5, 0] * conv_z0y0x0
    del conv_z0y2

    b_2 = qinv[3, 3] * cupyx.scipy.ndimage.correlate1d(conv_z1y0, kernel_0, axis=2)
    a_02 = qinv[8, 8] * cupyx.scipy.ndimage.correlate1d(conv_z1y0, kernel_1, axis=2) / 2
    del conv_z1y0

    a_12 = qinv[9, 9] * cupyx.scipy.ndimage.correlate1d(conv_z1y1, kernel_0, axis=2) / 2
    del conv_z1y1

    a_22 = qinv[6, 6] * cupyx.scipy.ndimage.correlate1d(conv_z2y0, kernel_0, axis=2) + qinv[6, 0] * conv_z0y0x0
    del conv_z2y0, conv_z0y0x0

    return b_0, b_1, b_2, a_00, a_01, a_02, a_11, a_12, a_22


@cuda.jit
def update_matrices(b1_0, b1_1, b1_2, a1_00, a1_01, a1_02, a1_11, a1_12, a1_22,
                    b2_0, b2_1, b2_2, a2_00, a2_01, a2_02, a2_11, a2_12, a2_22,
                    vx, vy, vz, border,
                    h0, h1, h2, g00, g01, g02, g11, g12, g22):
    """Sets up the matrices that can be used to solve for the velocities

        Matrices are in the format [[g00, g01, g02], and [[h0],
                                    [g01, g11, g12],      [h1],
                                    [g02, g12, g22]]      [h2]]
        Matrices are updated in place.

    Args:
        b1_0 (cuda array): array containing the first value of the B array from the first image
        b1_1 (cuda array): array containing the second value of the B array from the first image
        b1_2 (cuda array): array containing the third value of the B array from the first image
        a1_00 (cuda array): array containing the values of the A array from the first image
        a1_01 (cuda array): array containing the values of the A array from the first image
        a1_02 (cuda array): array containing the values of the A array from the first image
        a1_11 (cuda array): array containing the values of the A array from the first image
        a1_12 (cuda array): array containing the values of the A array from the first image
        a1_22 (cuda array): array containing the values of the A array from the first image

        b2_0 (cuda array): array containing the first value of the B array from the second image
        b2_1 (cuda array): array containing the second value of the B array from the second image
        b2_2 (cuda array): array containing the third value of the B array from the second image
        a2_00 (cuda array): array containing the values of the A array from the second image
        a2_01 (cuda array): array containing the values of the A array from the second image
        a2_02 (cuda array): array containing the values of the A array from the second image
        a2_11 (cuda array): array containing the values of the A array from the second image
        a2_12 (cuda array): array containing the values of the A array from the second image
        a2_22 (cuda array): array containing the values of the A array from the second image

        vx (cuda array): array containing the displacements in the x direction
        vy (cuda array): array containing the displacements in the y direction
        vz (cuda array): array containing the displacements in the z direction
        border (cuda array): array containing the weighting factor to use for calculations near the border

        h0 (cuda array): array containing the first value of the h matrix
        h1 (cuda array): array containing the second value of the h matrix
        h2 (cuda array): array containing the third value of the h matrix
        g00 (cuda array): array containing the values of the g matrix
        g01 (cuda array): array containing the values of the g matrix
        g02 (cuda array): array containing the values of the g matrix
        g11 (cuda array): array containing the values of the g matrix
        g12 (cuda array): array containing the values of the g matrix
        g22 (cuda array): array containing the values of the g matrix

    Returns:
        None
    """
    z, y, x = cuda.grid(3)

    r = cuda.local.array(shape=(9,), dtype=np.float32)
    for j in range(9):
        r[j] = 0.0

    border_size = len(border) - 1
    depth, length, width = vx.shape

    if z < depth and y < length and x < width:
        dx = vx[z, y, x]
        dy = vy[z, y, x]
        dz = vz[z, y, x]

        fx = x + dx
        fy = y + dy
        fz = z + dz

        x1 = int(math.floor(fx))
        y1 = int(math.floor(fy))
        z1 = int(math.floor(fz))

        fx -= x1
        fy -= y1
        fz -= z1

        ## interpolate values
        if 0 <= x1 and 0 <= y1 and 0 <= z1 and x1 < (width - 1) and y1 < (length - 1) and z1 < (depth - 1):
            a000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
            a001 = fx * (1.0 - fy) * (1.0 - fz)
            a010 = (1.0 - fx) * fy * (1.0 - fz)
            a100 = (1.0 - fx) * (1.0 - fy) * fz
            a011 = fx * fy * (1.0 - fz)
            a101 = fx * (1.0 - fy) * fz
            a110 = (1.0 - fx) * fy * fz
            a111 = fx * fy * fz

            r[0] = a000 * b2_0[z1, y1, x1] + \
                   a001 * b2_0[z1, y1, x1 + 1] + \
                   a010 * b2_0[z1, y1 + 1, x1] + \
                   a100 * b2_0[z1 + 1, y1, x1] + \
                   a011 * b2_0[z1, y1 + 1, x1 + 1] + \
                   a101 * b2_0[z1 + 1, y1, x1 + 1] + \
                   a110 * b2_0[z1 + 1, y1 + 1, x1] + \
                   a111 * b2_0[z1 + 1, y1 + 1, x1 + 1]

            r[1] = a000 * b2_1[z1, y1, x1] + \
                   a001 * b2_1[z1, y1, x1 + 1] + \
                   a010 * b2_1[z1, y1 + 1, x1] + \
                   a100 * b2_1[z1 + 1, y1, x1] + \
                   a011 * b2_1[z1, y1 + 1, x1 + 1] + \
                   a101 * b2_1[z1 + 1, y1, x1 + 1] + \
                   a110 * b2_1[z1 + 1, y1 + 1, x1] + \
                   a111 * b2_1[z1 + 1, y1 + 1, x1 + 1]

            r[2] = a000 * b2_2[z1, y1, x1] + \
                   a001 * b2_2[z1, y1, x1 + 1] + \
                   a010 * b2_2[z1, y1 + 1, x1] + \
                   a100 * b2_2[z1 + 1, y1, x1] + \
                   a011 * b2_2[z1, y1 + 1, x1 + 1] + \
                   a101 * b2_2[z1 + 1, y1, x1 + 1] + \
                   a110 * b2_2[z1 + 1, y1 + 1, x1] + \
                   a111 * b2_2[z1 + 1, y1 + 1, x1 + 1]

            r[3] = a000 * a2_00[z1, y1, x1] + \
                   a001 * a2_00[z1, y1, x1 + 1] + \
                   a010 * a2_00[z1, y1 + 1, x1] + \
                   a100 * a2_00[z1 + 1, y1, x1] + \
                   a011 * a2_00[z1, y1 + 1, x1 + 1] + \
                   a101 * a2_00[z1 + 1, y1, x1 + 1] + \
                   a110 * a2_00[z1 + 1, y1 + 1, x1] + \
                   a111 * a2_00[z1 + 1, y1 + 1, x1 + 1]

            r[4] = a000 * a2_01[z1, y1, x1] + \
                   a001 * a2_01[z1, y1, x1 + 1] + \
                   a010 * a2_01[z1, y1 + 1, x1] + \
                   a100 * a2_01[z1 + 1, y1, x1] + \
                   a011 * a2_01[z1, y1 + 1, x1 + 1] + \
                   a101 * a2_01[z1 + 1, y1, x1 + 1] + \
                   a110 * a2_01[z1 + 1, y1 + 1, x1] + \
                   a111 * a2_01[z1 + 1, y1 + 1, x1 + 1]

            r[5] = a000 * a2_02[z1, y1, x1] + \
                   a001 * a2_02[z1, y1, x1 + 1] + \
                   a010 * a2_02[z1, y1 + 1, x1] + \
                   a100 * a2_02[z1 + 1, y1, x1] + \
                   a011 * a2_02[z1, y1 + 1, x1 + 1] + \
                   a101 * a2_02[z1 + 1, y1, x1 + 1] + \
                   a110 * a2_02[z1 + 1, y1 + 1, x1] + \
                   a111 * a2_02[z1 + 1, y1 + 1, x1 + 1]

            r[6] = a000 * a2_11[z1, y1, x1] + \
                   a001 * a2_11[z1, y1, x1 + 1] + \
                   a010 * a2_11[z1, y1 + 1, x1] + \
                   a100 * a2_11[z1 + 1, y1, x1] + \
                   a011 * a2_11[z1, y1 + 1, x1 + 1] + \
                   a101 * a2_11[z1 + 1, y1, x1 + 1] + \
                   a110 * a2_11[z1 + 1, y1 + 1, x1] + \
                   a111 * a2_11[z1 + 1, y1 + 1, x1 + 1]

            r[7] = a000 * a2_12[z1, y1, x1] + \
                   a001 * a2_12[z1, y1, x1 + 1] + \
                   a010 * a2_12[z1, y1 + 1, x1] + \
                   a100 * a2_12[z1 + 1, y1, x1] + \
                   a011 * a2_12[z1, y1 + 1, x1 + 1] + \
                   a101 * a2_12[z1 + 1, y1, x1 + 1] + \
                   a110 * a2_12[z1 + 1, y1 + 1, x1] + \
                   a111 * a2_12[z1 + 1, y1 + 1, x1 + 1]

            r[8] = a000 * a2_22[z1, y1, x1] + \
                   a001 * a2_22[z1, y1, x1 + 1] + \
                   a010 * a2_22[z1, y1 + 1, x1] + \
                   a100 * a2_22[z1 + 1, y1, x1] + \
                   a011 * a2_22[z1, y1 + 1, x1 + 1] + \
                   a101 * a2_22[z1 + 1, y1, x1 + 1] + \
                   a110 * a2_22[z1 + 1, y1 + 1, x1] + \
                   a111 * a2_22[z1 + 1, y1 + 1, x1 + 1]

            r[3] = (a1_00[z, y, x] + r[3]) * 0.5
            r[4] = (a1_01[z, y, x] + r[4]) * 0.25
            r[5] = (a1_02[z, y, x] + r[5]) * 0.25
            r[6] = (a1_11[z, y, x] + r[6]) * 0.5
            r[7] = (a1_12[z, y, x] + r[7]) * 0.25
            r[8] = (a1_22[z, y, x] + r[8]) * 0.5
        else:
            r[3] = a1_00[z, y, x]
            r[4] = a1_01[z, y, x] * 0.5
            r[5] = a1_02[z, y, x] * 0.5
            r[6] = a1_11[z, y, x]
            r[7] = a1_12[z, y, x] * 0.5
            r[8] = a1_22[z, y, x]

        r[0] = ((b1_0[z, y, x] - r[0]) * 0.5) + (r[3] * dx + r[4] * dy + r[5] * dz)

        r[1] = ((b1_1[z, y, x] - r[1]) * 0.5) + (r[4] * dx + r[6] * dy + r[7] * dz)

        r[2] = (b1_2[z, y, x] - r[2]) * 0.5 + (r[5] * dx + r[7] * dy + r[8] * dz)

        scale = border[min(x, border_size)] * \
                border[min(y, border_size)] * \
                border[min(z, border_size)] * \
                border[min(width - x - 1, border_size)] * \
                border[min(length - y - 1, border_size)] * \
                border[min(depth - z - 1, border_size)]

        for j in range(9):
            r[j] = r[j] * scale

        g00[z, y, x] = r[3] * r[3] + r[4] * r[4] + r[5] * r[5]
        g01[z, y, x] = r[3] * r[4] + r[4] * r[6] + r[5] * r[7]
        g02[z, y, x] = r[3] * r[5] + r[4] * r[7] + r[5] * r[8]
        g11[z, y, x] = r[4] * r[4] + r[6] * r[6] + r[7] * r[7]
        g12[z, y, x] = r[4] * r[5] + r[6] * r[7] + r[7] * r[8]
        g22[z, y, x] = r[5] * r[5] + r[7] * r[7] + r[8] * r[8]

        h0[z, y, x] = r[3] * r[0] + r[4] * r[1] + r[5] * r[2]
        h1[z, y, x] = r[4] * r[0] + r[6] * r[1] + r[7] * r[2]
        h2[z, y, x] = r[5] * r[0] + r[7] * r[1] + r[8] * r[2]

def gaussian_pyramid_3d(image, sigma: float = 1, scale: float = 0.5) -> typing.Tuple[np.ndarray, npt.ArrayLike]:
    """ Downscales the image for use in a Gaussian pyramid

    Args:
        image (cuda array): Image to generate pyramids from
        sigma (float): Standard deviation of the Gaussian kernel used for downscaling. Defaults to 1
        scale (float): Scale factor used to downscale the image. Defaults to 0.5.

    Returns:
        resized_image (ndarray): Downscaled image
        true_scale (typing.Tuple[float, float, float]): Actual scaling factor used. This differs slightly from the input
            scaling factor in cases when the factor used causes the size of the image to not be an integer.
    """
    kernel = cp.asarray(gaussian_kernel_1d(sigma), dtype=cp.float32)
    radius = math.ceil(2 * sigma)

    # gaussian smoothing
    image = cupyx.scipy.ndimage.convolve(image, cp.reshape(kernel, (2 * radius + 1, 1, 1)), mode="reflect")
    image = cupyx.scipy.ndimage.convolve(image, cp.reshape(kernel, (1, 2 * radius + 1, 1)), mode="reflect")
    image = cupyx.scipy.ndimage.convolve(image, cp.reshape(kernel, (1, 1, 2 * radius + 1)), mode="reflect")

    shape = image.shape
    true_scale = [int(round(shape[0] * scale)) / shape[0],
                  int(round(shape[1] * scale)) / shape[1],
                  int(round(shape[2] * scale)) / shape[2]]
    resized_image = cp.empty((int(round(shape[0] * scale)),
                              int(round(shape[1] * scale)),
                              int(round(shape[2] * scale))), dtype=cp.float32)
    cupyx.scipy.ndimage.zoom(image, (scale, scale, scale), output=resized_image, mode="reflect")

    return resized_image, true_scale

def imresize_3d(image, scale: typing.Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> np.ndarray:
    """ Upscales the image by the specified factor

    Args:
        image (cuda array): image to generate pyramids from
        scale (typing.Tuple[float, float, float]): Scale factor used to downscale the image. Actual factor used is 1/scale.
            Defaults to (0.5, 0.5, 0.5).

    Returns:
        image (ndarray): Upscaled image
    """
    image = cupyx.scipy.ndimage.zoom(image, (1 / scale[0], 1 / scale[1], 1 / scale[2]))

    return image

def get_positions(
    start_point: Tuple[int, int, int],
    total_vol: Tuple[int, int, int],
    vol: Tuple[int, int, int],
    shape: Tuple[int, int, int],
    overlap: Tuple[int, int, int],
    axis: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
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
    position: List[Tuple[int, int]] = []
    valid_vol: List[Tuple[int, int]] = []
    valid_position: List[Tuple[int, int]] = []

    count = q + (1 if r != 0 else 0)
    for i in range(count):
        if i == 0:
            start = start_point[axis] - overlap[axis] // 2
            valid_start = 0
        else:
            start = end - overlap[axis]
            valid_start = valid_end

        end = start + vol[axis]

        # clamp to volume bounds
        _start = max(start, 0)
        start_diff = start - _start
        start_valid = overlap[axis] // 2 + start_diff

        # careful: min of multiple things; original had tuple — fix to chain min calls
        _end = min(end, shape[axis], start_point[axis] + total_vol[axis] + overlap[axis] // 2)
        valid_end = min(end - overlap[axis] // 2 - start_point[axis], total_vol[axis])

        end_valid = valid_end - valid_start + start_valid

        position.append((_start, _end))
        valid_position.append((valid_start, valid_end))
        valid_vol.append((start_valid, end_valid))

    return position, valid_position, valid_vol


def farneback_3d(
    image1,
    image2,
    iters: int,
    num_levels: int,
    scale: float = 0.5,
    spatial_size: int = 9,
    sigma_k: float = 0.15,
    filter_type: str = "box",
    filter_size: int = 5,
    presmoothing: Optional[int] = None,
    threadsperblock: Tuple[int, int, int] = (8, 8, 8),
):
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
    filter_type_low = filter_type.lower()
    assert filter_type_low in {"gaussian", "box"}
    if filter_type_low == "gaussian":
        def filter_fn(x):
            return cupyx.scipy.ndimage.gaussian_filter(x, filter_size / 2 * 0.3)
    else:  # box
        def filter_fn(x):
            return cupyx.scipy.ndimage.uniform_filter(x, size=filter_size)

    image1 = cp.asarray(image1, dtype=cp.float32)
    image2 = cp.asarray(image2, dtype=cp.float32)
    if presmoothing is not None:
        image1 = cupyx.scipy.ndimage.gaussian_filter(image1, presmoothing)
        image2 = cupyx.scipy.ndimage.gaussian_filter(image2, presmoothing)

    # initialize gaussian pyramid
    gauss_pyramid_1: dict[int, cp.ndarray] = {1: image1}
    gauss_pyramid_2: dict[int, cp.ndarray] = {1: image2}
    true_scale_dict: dict[int, Tuple[int, int, int]] = {}

    for pyr_lvl in range(2, num_levels + 1):
        prev1 = gauss_pyramid_1[pyr_lvl - 1]
        prev2 = gauss_pyramid_2[pyr_lvl - 1]
        g1, scale_info = gaussian_pyramid_3d(prev1, sigma=1, scale=scale)
        g2, _ = gaussian_pyramid_3d(prev2, sigma=1, scale=scale)
        gauss_pyramid_1[pyr_lvl] = g1
        gauss_pyramid_2[pyr_lvl] = g2
        true_scale_dict[pyr_lvl] = scale_info

    # If single int, broadcast to list per level
    if not isinstance(iters, list):
        iters = [iters] * num_levels

    vx = vy = vz = confidence = None  # type: ignore

    # Pyramid (coarse → fine)
    for lvl in range(num_levels, 0, -1):
        lvl_image_1 = gauss_pyramid_1[lvl]
        lvl_image_2 = gauss_pyramid_2[lvl]

        if lvl == num_levels:
            # initialize velocities
            vx = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
            vy = cp.zeros_like(vx)
            vz = cp.zeros_like(vx)
        else:
            # sanitize previous flow fields
            assert vx is not None and vy is not None and vz is not None
            vx = cp.nan_to_num(vx)
            vy = cp.nan_to_num(vy)
            vz = cp.nan_to_num(vz)

            # zero-out flows beyond confidence (if exists)
            if confidence is not None:
                mask = cp.abs(confidence) > 1
                vx = cp.where(mask, 0, vx)
                vy = cp.where(mask, 0, vy)
                vz = cp.where(mask, 0, vz)

            # upsample flows to current resolution
            # Note: ensure you define imresize_3d correctly for your use-case
            scale_info = true_scale_dict[lvl + 1]
            vx = (1 / scale_info[2]) * imresize_3d(vx, scale=scale_info)
            vy = (1 / scale_info[1]) * imresize_3d(vy, scale=scale_info)
            vz = (1 / scale_info[0]) * imresize_3d(vz, scale=scale_info)

        # make the A, B matrices etc
        b1_0, b1_1, b1_2, a1_00, a1_01, a1_02, a1_11, a1_12, a1_22 = make_abc_fast(
            lvl_image_1, spatial_size, sigma_k=sigma_k
        )
        b2_0, b2_1, b2_2, a2_00, a2_01, a2_02, a2_11, a2_12, a2_22 = make_abc_fast(
            lvl_image_2, spatial_size, sigma_k=sigma_k
        )

        border = cp.asarray([0.14, 0.14, 0.4472, 0.4472, 0.4472, 1], dtype=cp.float32)
        shape0 = vx.shape
        h0 = cp.zeros(shape0, dtype=cp.float32)
        h1 = cp.zeros(shape0, dtype=cp.float32)
        h2 = cp.zeros(shape0, dtype=cp.float32)
        g00 = cp.zeros(shape0, dtype=cp.float32)
        g01 = cp.zeros(shape0, dtype=cp.float32)
        g02 = cp.zeros(shape0, dtype=cp.float32)
        g11 = cp.zeros(shape0, dtype=cp.float32)
        g12 = cp.zeros(shape0, dtype=cp.float32)
        g22 = cp.zeros(shape0, dtype=cp.float32)

        for i in range(iters[lvl - 1]):
            blockspergrid = (
                math.ceil(shape0[0] / threadsperblock[0]),
                math.ceil(shape0[1] / threadsperblock[1]),
                math.ceil(shape0[2] / threadsperblock[2]),
            )

            update_matrices[blockspergrid, threadsperblock](
                b1_0, b1_1, b1_2, a1_00, a1_01, a1_02, a1_11, a1_12, a1_22,
                b2_0, b2_1, b2_2, a2_00, a2_01, a2_02, a2_11, a2_12, a2_22,
                vx, vy, vz, border,
                h0, h1, h2, g00, g01, g02, g11, g12, g22
            )
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

            update_flow[blockspergrid, threadsperblock](
                h0, h1, h2, g00, g01, g02, g11, g12, g22, vx, vy, vz
            )
            cp.cuda.Stream.null.synchronize()

            if i == iters[lvl - 1] - 1:
                confidence = cp.zeros(vx.shape, dtype=cp.float32)
                calculate_confidence[blockspergrid, threadsperblock](
                    h0, h1, h2, g00, g01, g02, g11, g12, g22, vx, vy, vz,
                    confidence
                )

            cp.cuda.Stream.null.synchronize()

    assert vx is not None and vy is not None and vz is not None and confidence is not None
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

    def __init__(
        self,
        iters: int = 5,
        num_levels: int = 5,
        scale: float = 0.5,
        spatial_size: int = 9,
        sigma_k: float = 0.15,
        filter_type: str = "box",
        filter_size: int = 21,
        presmoothing: Optional[int] = None,
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

    def calculate_flow(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        start_point: Tuple[int, int, int] = (0, 0, 0),
        total_vol: Optional[Tuple[int, int, int]] = None,
        sub_volume: Tuple[int, int, int] = (256, 256, 256),
        overlap: Tuple[int, int, int] = (64, 64, 64),
        threadsperblock: Tuple[int, int, int] = (8, 8, 8),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            # Using subtraction of np.ndarray and tuple: convert
            total_vol = tuple(np.array(image1.shape) - np.array(start_point))  # type: ignore

        print("Running 3D Farneback optical flow with the following parameters:")
        print(
            f"Iters: {self.iters} | Levels: {self.num_levels} | Scale: {self.scale} | "
            f"Kernel: {self.spatial_size} | Filter: {self.filter_type}-{self.filter_size} | "
            f"Presmoothing: {self.presmoothing}",
            flush=True,
        )

        output_vx = np.zeros(total_vol, dtype=np.float32)
        output_vy = np.zeros(total_vol, dtype=np.float32)
        output_vz = np.zeros(total_vol, dtype=np.float32)
        output_confidence = np.zeros(total_vol, dtype=np.float32)

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        if any(tv > sv for tv, sv in zip(total_vol, sub_volume)):
            shape = image1.shape
            z_position, z_valid_pos, z_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 0)
            y_position, y_valid_pos, y_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 1)
            x_position, x_valid_pos, x_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 2)

            total_tiles = len(z_position) * len(y_position)
            for z_i in range(len(z_position)):
                for y_i in range(len(y_position)):
                    with tqdm(
                        range(len(x_position)),
                        desc=f"Tile {z_i * len(y_position) + y_i + 1}/{total_tiles}"
                    ) as x_iter:
                        for x_i in x_iter:
                            slc_z = slice(z_position[z_i][0], z_position[z_i][1])
                            slc_y = slice(y_position[y_i][0], y_position[y_i][1])
                            slc_x = slice(x_position[x_i][0], x_position[x_i][1])
                            input_image_vol_1 = image1[slc_z, slc_y, slc_x].astype(np.float32)
                            input_image_vol_2 = image2[slc_z, slc_y, slc_x].astype(np.float32)

                            with cp.cuda.Device(self.device_id):
                                vx, vy, vz, confidence = farneback_3d(
                                    input_image_vol_1,
                                    input_image_vol_2,
                                    self.iters,
                                    self.num_levels,
                                    scale=self.scale,
                                    spatial_size=self.spatial_size,
                                    sigma_k=self.sigma_k,
                                    filter_type=self.filter_type,
                                    filter_size=self.filter_size,
                                    presmoothing=self.presmoothing,
                                    threadsperblock=threadsperblock,
                                )
                                cp.cuda.Stream.null.synchronize()

                            vx_cpu = vx.get()
                            vy_cpu = vy.get()
                            vz_cpu = vz.get()
                            confidence_cpu = confidence.get()

                            out_slc_z = slice(z_valid_pos[z_i][0], z_valid_pos[z_i][1])
                            out_slc_y = slice(y_valid_pos[y_i][0], y_valid_pos[y_i][1])
                            out_slc_x = slice(x_valid_pos[x_i][0], x_valid_pos[x_i][1])

                            in_slc_z = slice(z_valid[z_i][0], z_valid[z_i][1])
                            in_slc_y = slice(y_valid[y_i][0], y_valid[y_i][1])
                            in_slc_x = slice(x_valid[x_i][0], x_valid[x_i][1])

                            output_vx[out_slc_z, out_slc_y, out_slc_x] = vx_cpu[in_slc_z, in_slc_y, in_slc_x]
                            output_vy[out_slc_z, out_slc_y, out_slc_x] = vy_cpu[in_slc_z, in_slc_y, in_slc_x]
                            output_vz[out_slc_z, out_slc_y, out_slc_x] = vz_cpu[in_slc_z, in_slc_y, in_slc_x]
                            output_confidence[out_slc_z, out_slc_y, out_slc_x] = confidence_cpu[in_slc_z, in_slc_y, in_slc_x]

                            del vx, vy, vz, confidence
                            del vx_cpu, vy_cpu, vz_cpu, confidence_cpu
                            mempool.free_all_blocks()
                            pinned_mempool.free_all_blocks()
        else:
            with cp.cuda.Device(self.device_id):
                vx, vy, vz, confidence = farneback_3d(
                    image1,
                    image2,
                    iters=self.iters,
                    num_levels=self.num_levels,
                    scale=self.scale,
                    spatial_size=self.spatial_size,
                    sigma_k=self.sigma_k,
                    filter_type=self.filter_type,
                    filter_size=self.filter_size,
                    presmoothing=self.presmoothing,
                    threadsperblock=threadsperblock,
                )
                cp.cuda.Stream.null.synchronize()

            output_vx = vx.get()
            output_vy = vy.get()
            output_vz = vz.get()
            output_confidence = confidence.get()

        return output_vz, output_vy, output_vx, output_confidence
