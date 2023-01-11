import numpy as np


############### This is a module to calculate outcome of SSIM Method  ###############


def ssim_block(block1, block2):
    """
    ssim index of two image patches
    :param block1: ndarray
    image patch 1
    :param block2: ndarray
    image patch 2
    :return: double
    ssim index
    """
    k = np.array([0.01, 0.03])
    l_range = 1
    c = (k * l_range) ** 2
    block1 = block1.flatten()
    block2 = block2.flatten()
    mean_block = [np.mean(block1), np.mean(block2)]
    var_matrix = np.cov(block1, block2)
    ssim_numerator = (2 * mean_block[0] * mean_block[1] + c[0]) * (2 * var_matrix[0, 1] + c[1])
    ssim_denominator = (mean_block[0] ** 2 + mean_block[1] ** 2 + c[0]) * (var_matrix[0, 0] + var_matrix[1, 1] + c[1])
    ssim_index = ssim_numerator / ssim_denominator
    return ssim_index


def detect_distortion(color, depth, color_lut, depth_lut, r):
    """
    detect distorted pixels in depth map
    :param color: ndarray
    input color image
    :param depth: ndarray
    input depth map
    :param color_lut: ndarray
    color LUT for fast calculation
    :param depth_lut: ndarray
    depth LUT for fast calculation
    :param r: int
    radius of local window
    :return: ndarray
    a map present the degree of pixel distortion
    """
    (h, w) = depth.shape
    win = 2 * r + 1
    result = np.full((h, w), 0).astype(np.float32)

    color_pad = np.pad(color, ((r, r), (r, r), (0, 0)), 'reflect')
    depth_pad = np.pad(depth, ((r, r), (r, r)), 'reflect')

    color_weight = np.full((win, win), 0).astype(np.float32)
    depth_weight = np.full((win, win), 0).astype(np.float32)

    for i in range(h):
        for j in range(w):
            reg_color = color_pad[i:i + win, j:j + win, :].astype(np.int32)
            reg_depth = depth_pad[i:i + win, j:j + win].astype(np.int32)
            for i_reg in range(win):
                for j_reg in range(win):
                    diffc = reg_color[i_reg, j_reg, :] - reg_color[r, r, :]
                    diffd = reg_depth[i_reg, j_reg] - reg_depth[r, r]
                    color_index = int(np.mean(diffc ** 2))
                    depth_index = int(diffd ** 2)
                    color_weight[i_reg, j_reg] = color_lut[color_index]
                    depth_weight[i_reg, j_reg] = depth_lut[depth_index]

            ssim_index = ssim_block(color_weight, depth_weight)

            if ssim_index < 0:
                ssim_index = 0

            result[i, j] = ssim_index

    return result


def unary(color, depth, cd_lut, r):
    """
    unary probability for CRF
    :param color: ndarray
    input color image
    :param depth: ndarray
    input depth map
    :param cd_lut: ndarray
    color and depth LUT for fast calculation
    :param r: int
    radius of local window
    :return: ndarray
    unary probability for CRF
    """
    threshold_h = 0.4
    threshold_l = 0.2

    color_lut = cd_lut[0]
    depth_lut = cd_lut[1]

    winc = detect_distortion(color, depth, color_lut, depth_lut, r)
    winc[winc > threshold_h] = threshold_h
    winc[winc < threshold_l] = threshold_l

    return winc