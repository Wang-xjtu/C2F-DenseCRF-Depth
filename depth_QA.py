import numpy as np
import os
import cv2
import random
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from PIL import Image


##################### This is a module to objectively evaluate depth map #####################

def part_rmse(depth, ground_truth):
    diff = depth * 1.0 - ground_truth * 1.0
    diff[ground_truth == 0] = 0.
    num = np.count_nonzero(ground_truth)
    pmse = np.sum(diff ** 2) / num
    prmse = pmse ** 0.5
    return prmse


def part_mae(depth, ground_truth):
    diff = depth * 1.0 - ground_truth * 1.0
    diff[ground_truth == 0] = 0.
    num = np.count_nonzero(ground_truth)
    pmae = np.sum(abs(diff)) / num
    return pmae


def absRel(depth, ground_truth):
    diff = depth * 1.0 - ground_truth * 1.0
    diff[ground_truth == 0] = 0.
    num = np.count_nonzero(ground_truth)
    rel = np.sum(abs(diff) / (ground_truth + 1e-6)) / num
    return rel


def threshold_accuracy(depth, ground_truth):
    ratio1 = depth * 1.0 / (ground_truth + 1e-6)
    ratio2 = ground_truth * 1.0 / (depth + 1e-6)
    count125 = 0
    count1252 = 0
    count1253 = 0
    num = np.count_nonzero(ground_truth)

    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if ground_truth[i, j] != 0:
                max_ratio = max(ratio1[i, j], ratio2[i, j])
                if max_ratio < 1.25:
                    count125 += 1
                    count1252 += 1
                    count1253 += 1
                elif max_ratio < 1.25 ** 2:
                    count1252 += 1
                    count1253 += 1
                elif max_ratio < 1.25 ** 3:
                    count1253 += 1

    thres_125 = count125 * 1.0 / num
    thres_1252 = count1252 * 1.0 / num
    thres_1253 = count1253 * 1.0 / num

    return thres_125, thres_1252, thres_1253


def evaluation(dataset_dir):
    # clear
    for file in Path(dataset_dir + '/metric').rglob('*.txt'):
        os.remove(str(file))
    # evaluation
    for file in os.listdir(os.path.join(dataset_dir, 'gt')):
        gt_path = os.path.join(dataset_dir, 'gt', file)
        depth_path = os.path.join(dataset_dir, 'depth', file)
        c2f_path = os.path.join(dataset_dir, 'result', file[:-4] + '_c2f.png')

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        c2f = cv2.imread(c2f_path, cv2.IMREAD_GRAYSCALE)

        depth_rmse = part_rmse(depth, gt)
        depth_mae = part_mae(depth, gt)
        depth_absrel = absRel(depth, gt)
        depth_thres_125, depth_thres_1252, depth_thres_1253 = threshold_accuracy(depth, gt)

        c2f_rmse = part_rmse(c2f, gt)
        c2f_mae = part_mae(c2f, gt)
        c2f_absrel = absRel(c2f, gt)
        c2f_thres_125, c2f_thres_1252, c2f_thres_1253 = threshold_accuracy(c2f, gt)

        with open(os.path.join(dataset_dir, 'metric', 'c2f_RMSE.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(c2f_rmse) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'c2f_MAE.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(c2f_mae) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'c2f_absRel.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(c2f_absrel) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'c2f_thres125.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(c2f_thres_125) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'c2f_thres1252.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(c2f_thres_1252) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'c2f_thres1253.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(c2f_thres_1253) + '\n')

        with open(os.path.join(dataset_dir, 'metric', 'depth_RMSE.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(depth_rmse) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'depth_MAE.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(depth_mae) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'depth_absRel.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(depth_absrel) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'depth_thres125.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(depth_thres_125) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'depth_thres1252.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(depth_thres_1252) + '\n')
        with open(os.path.join(dataset_dir, 'metric', 'depth_thres1253.txt'), 'a') as t:
            t.write(file[:-4] + ' = ' + str(depth_thres_1253) + '\n')
