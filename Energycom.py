import numpy as np
import math
from SSIM_method import unary


############### This is a module to calculate energy function's components ###############

def compat_matrix(values, n_values, gau_lut):
    """
    Calculate label compatibility matrix: U(xi,xj)= 1- gaussian kernel

    Parameters
    ----------
    values: numpy.array
        unique intensity vector of depth map
    n_values: number
        the number of unique intensities
    gau_lut: numpy.array
        look up table for quickly searching
    sig: number
        the standard variance of gaussian kernel
    """
    result = np.empty((n_values, n_values)).astype(np.float32)
    for i in range(n_values):
        for j in range(n_values):
            delta_x2 = (int(values[i]) - int(values[j])) ** 2
            result[i, j] = 1 - gau_lut[delta_x2]
    return result


def unary_from_ssim(labels, n_values, inrgb, indep, cd_lut, r, zero_unsure=True):
    """
    Calculate unary potential from ssim method

    Parameters
    ----------
    labels: numpy.array
        corresponding labels of every pixel in Depth map.
        Such as depth intensity 255 corresponding to label 35
    n_values: number
        unique intensity vector of depth map
    inrgb: numpy.array
        input RGB image
    indep: numpy.array
        input Depth map
    r: number
        the radius of ssim method window
    zero_unsure: bool
        If `True`, treat the label value `0` as meaning "could be anything",
        If `False`, do not treat the value `0` specially, but just as any
        other class.
    """
    # initializing
    labels = labels.flatten()
    U = np.empty((n_values, len(labels)), dtype='float32')

    ssim_pro = unary(inrgb, indep, cd_lut, r).flatten()
    r_pro = (1 - ssim_pro) / (n_values - 1)  # average the rest probability

    # add unary energy into U
    for i in range(len(labels)):
        U[:, i] = -np.log(r_pro[i])
        if zero_unsure:
            # if label[i]==0: labels[i] - 1 = -1
            U[labels[i] - 1, i] = -np.log(ssim_pro[i])
        else:
            U[labels[i], i] = -np.log(ssim_pro[i])

    # Overwrite 0-labels using uniform probability
    if zero_unsure:
        U[:, labels == 0] = -np.log(1.0 / n_values)

    return U


def materialimg(img, r):
    """
    Project RGB image into a material image for pairwise potential

    Parameters
    ----------
    img: numpy.array
        input RGB image
    r: number
        the radius of material window
    """
    # initializing
    (h, w, d) = img.shape
    win = 2 * r + 1
    result = np.full((h, w, 2 * d), 0)
    img_pad = np.pad(img, ((r, r), (r, r), (0, 0)), 'reflect')

    # calculate R,G,B's mean and var
    for i in range(h):
        for j in range(w):
            reg = img_pad[i:i + win, j:j + win, :]

            for k in range(3):
                result[i, j, k] = np.mean(reg[:, :, k])
                result[i, j, k + 3] = np.var(reg[:, :, k])
    return result.astype(np.uint8)


def create_pairwise_dybilateral(sdims, schan, img, chdim=2):
    """
    Calculate pairwise potential by dynamic standard variance

    Parameters
    ----------
    sdims: numpy.array
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseBilateral`.
    schan: numpy.array
        The scaling factors per channel in the image. This is referred to
        `srgb` in `DenseCRF2D.addPairwiseBilateral`.
    img: numpy.array
        The input image.
    chdim: int, optional
        This specifies where the channel dimension is in the image. For
        example `chdim=2` for a RGB image of size (240, 300, 3). If the
        image has no channel dimension (e.g. it has only one channel) use
        `chdim=-1`.
    """
    # Put the channel dim as axis 0, all others stay relatively the same
    im_feat = np.rollaxis(img, chdim).astype(np.float32)

    # scale image features per channel
    # Allow for a single number in `schan` to broadcast across all channels:
    for i, s in enumerate(schan):
        im_feat[i] /= s

    # create a mesh
    cord_range = [range(s) for s in im_feat.shape[1:]]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s

    feats = np.concatenate([mesh, im_feat])
    return feats.reshape([feats.shape[0], -1])
