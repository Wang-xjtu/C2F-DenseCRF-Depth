import cv2 as cv
import numpy as np
import pydensecrf.densecrf as dcrf
import Energycom as ec
import Preprocess as pre


def c2f_crf_model(img, dep, mer_lut, dis_sigma=None, color_sigma=10, co_balance=10):
    ############### initializing ###############
    h = dep.shape[0]
    w = dep.shape[1]

    # get unique values and corresponding location labels
    values, labels = np.unique(dep, return_inverse=True)

    # 0 is regard as missing region
    Mis = 0 in values
    if Mis:
        values = values[1:]
    n_values = len(values)

    ############### setting unary and pairwise components ###############
    # adopt DenseCRF class
    d = dcrf.DenseCRF(w * h, n_values)
    # setting unary components
    U = ec.unary_from_ssim(labels, n_values, img, dep, mer_lut[1:], r=1, zero_unsure=Mis)
    d.setUnaryEnergy(U)
    # setting pairwise components
    if dis_sigma is not None:  # coarse model
        materialimg = ec.materialimg(img, 1)  # material image instead of input image
        feats = ec.create_pairwise_dybilateral(sdims=(dis_sigma, dis_sigma),
                                               schan=(
                                                   color_sigma, color_sigma, color_sigma, color_sigma, color_sigma,
                                                   color_sigma),
                                               img=materialimg,
                                               chdim=2)  # gi  (6 channels : mean*3 + var*3)

    else:  # fine model
        dis_sigma = 10
        feats = ec.create_pairwise_dybilateral(sdims=(dis_sigma, dis_sigma),
                                               schan=(color_sigma, color_sigma, color_sigma),
                                               img=img,
                                               chdim=2)

    la_compat = ec.compat_matrix(values, n_values, mer_lut[0])  # U(xi,xj)
    d.addPairwiseEnergy(feats, compat=co_balance * la_compat, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    ############### Dense crf inference ###############
    Q = np.array(d.inference(10))  # P( depth|color ) inference
    MAP_labels = np.argmax(Q, axis=0)  # select depth labels for max{P(depth|color)}
    MAP_values = (values[MAP_labels]).reshape((h, w))  # depth labels corresponding to depth values

    return MAP_values


def coarse2fine_depth(rgb_file, depth_file, merge_lut, dis_sigma=35, color_sigma=10, co_balance=10):
    color = pre.imreadC(rgb_file)
    depth = pre.imreadD(depth_file)

    Re_depth = c2f_crf_model(color, depth, merge_lut, dis_sigma, color_sigma, co_balance)  # coarse model
    Re_depth = c2f_crf_model(color, Re_depth, merge_lut, None, color_sigma, co_balance)  # fine model

    # filtering for better smoothing (optional)
    Re_depth_f = cv.ximgproc.guidedFilter(color, Re_depth, 1, 0.4)
    Re_depth_f = cv.ximgproc.weightedMedianFilter(color, Re_depth_f, 5, sigma=8)
    Re_depth_f = cv.medianBlur(Re_depth_f, 3)

    return Re_depth_f
