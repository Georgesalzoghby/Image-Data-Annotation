import os
import pandas as pd
import numpy as np
import tifffile

NUMBER_CHANNELS = 1
INPUT_DIR = "C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\Drosophila_TAD"
OUTPUT_DIR = INPUT_DIR + '_merged'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
files_list = sorted(os.listdir(INPUT_DIR))
print(files_list[-1])


for i in range(0, len(files_list)-1, NUMBER_CHANNELS):
    n0 = tifffile.imread(os.path.join(INPUT_DIR, files_list[i]))
    print(files_list[i])
    if NUMBER_CHANNELS == 2:
        n1 = tifffile.imread(os.path.join(INPUT_DIR, files_list[i+1]))
        n0 = np.stack((n0, n1))
        print(files_list[i + 1])

    print()

    with tifffile.TiffWriter(os.path.join(OUTPUT_DIR, files_list[i][:-7] + ".ome-tif")) as tif:
        if NUMBER_CHANNELS == 1:
            tif.write(n0, metadata={'axes': 'ZYX'})
        elif NUMBER_CHANNELS == 2:
            tif.write(n0, metadata={'axes': 'CZYX'})
