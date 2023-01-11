import Depth_CRF as dc
import Preprocess as pre
import numpy as np
from pathlib import Path
from PIL import Image
from depth_QA import evaluation

if __name__ == '__main__':
    dataset_dir = r'./examples'

    # Look-Up-Table for gaussian kernels
    merge_lut = np.array(
        [pre.calculate_lut(np.array(range((256 + 1) ** 2)), sig=10),  # color in pairwise
         pre.calculate_lut(np.array(range((256 + 1) ** 2)), sig=10),  # color in unary
         pre.calculate_lut(np.array(range((256 + 1) ** 2)), sig=10)])  # distance in unary

    # inference
    rgb_path = dataset_dir + '/rgb'
    for file in Path(rgb_path).rglob('*.png'):
        str_file = str(file)
        rgb_file = str_file
        depth_file = rgb_file.replace('\\rgb\\', '\\depth\\', 1)
        out_file = rgb_file.replace('\\rgb\\', '\\result\\', 1)
        out_file = out_file[:-4] + '_c2f.png'

        c2f = dc.coarse2fine_depth(rgb_file, depth_file, merge_lut)  # results
        result = Image.fromarray(c2f).save(out_file)
        print(out_file)

    # evaluation
    evaluation(dataset_dir)
