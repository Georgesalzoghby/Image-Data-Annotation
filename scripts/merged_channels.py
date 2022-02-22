import os
import pandas as pd
import numpy as np
import tifffile

INPUT_DIR = "C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID\\"
files_list = os.listdir(INPUT_DIR)
print(files_list[-1])
i = 0
for file_name in files_list[:-1]:
    print(i)
    n0 = tifffile.imread(os.path.join(INPUT_DIR + files_list[i]))
    n1 = tifffile.imread(os.path.join(INPUT_DIR + files_list[i+1]))
    final_image = np.stack((n0, n1))
    i += 2
    with tifffile.TiffWriter(os.path.join("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged\\",files_list[i])) as tif:
        tif.write(final_image, metadata={'axes': 'CZYX'})
