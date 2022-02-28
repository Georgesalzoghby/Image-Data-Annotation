import os
import pandas as pd
import numpy as np
import tifffile

INPUT_DIR = "C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID\\"
files_list = os.listdir(INPUT_DIR)
print(files_list[-1])

for i in range(0, len(files_list)-1, 2):
    n0 = tifffile.imread(os.path.join(INPUT_DIR + files_list[i]))
    n1 = tifffile.imread(os.path.join(INPUT_DIR + files_list[i+1]))
    final_image = np.stack((n0, n1))
    print(files_list[i])
    print(files_list[i+1])
    print()

    with tifffile.TiffWriter(os.path.join("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged",files_list[i][:-7] + ".ome-tif")) as tif:
        tif.write(final_image, metadata={'axes': 'CZYX'})
