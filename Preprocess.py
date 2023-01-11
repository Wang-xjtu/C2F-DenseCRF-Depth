import numpy as np
from PIL import Image


##################### This is a module to preprocess images #####################

# read depth image
def imreadD(Depth):
    """
    read Depth map as numpy.array

    Parameters
    ----------
    Depth: str
        the location of input Depth map
    """
    dim = (np.array(Image.open(Depth))).ndim

    # checking, if rgb covert to gray
    if dim > 2:
        result = (Image.open(Depth)).convert('L')
    else:
        result = Image.open(Depth)
    return np.array(result)


def imreadC(Color):
    """
    read RGB image as numpy.array

    Parameters
    ----------
    Color: str
        the location of input RGB image
    """
    result = Image.open(Color)
    return np.array(result)


def gua(delta_x2, sig):
    """
    Calculate Gaussian kernel for pairwise intensity

    Parameters
    ----------
    delta_x2: number
        input values, such as the intensity of one pixel in gray image (one channel)
    sig: number
        the standard variance of gaussian kernel
    """
    var = sig ** 2
    result = np.exp(-delta_x2 / (2 * var))

    return result


def calculate_lut(lut_range, sig=15):
    """
    :param lut_range: number. the range of LUT
    :param sig: variance of Gaussian kernel
    :return: LUT for fast calculation
    """
    result_lut = np.array(gua(lut_range, sig))
    return result_lut
