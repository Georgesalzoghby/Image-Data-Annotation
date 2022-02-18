import os
import pandas as pd
import numpy as np
import tifffile

INPUT_DIR = "C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID"

image1 = tifffile.imread(os.path.join(INPUT_DIR, '20190720_RI512_CTCF-AID_control_61b_61a_SIR_2C_ALN_THR_1_C1.tif'))
image2 = tifffile.imread(os.path.join(INPUT_DIR, '20190720_RI512_CTCF-AID_control_61b_61a_SIR_2C_ALN_THR_1_C2.tif'))
final_image = np.stack((image1, image2))

with tifffile.TiffWriter('temp.ome.tif') as tif:
    tif.write(final_image, metadata={'axes': 'CZYX'})
