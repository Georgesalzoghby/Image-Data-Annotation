import os
import json
import numpy as np
import tifffile

INPUT_DIR = "C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\Drosophila_TAD"
OUTPUT_DIR = INPUT_DIR + '_merged'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
    config = json.load(config_file)

nr_channels = config["nr_channels"]


files_list = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')])

# TODO: This line is extracting the metadata into a file
# ./bftools/showinf -nopix -nocore -omexml ./Image/701139/Binary/20190720_RI512_ES-CTL_61b-647_61a-565_DAPI_001_SIR.dv > metadata.xml

for i in range(0, len(files_list), nr_channels):
    n0 = tifffile.imread(os.path.join(INPUT_DIR, files_list[i]))
    print(files_list[i])
    if nr_channels == 2:
        n1 = tifffile.imread(os.path.join(INPUT_DIR, files_list[i+1]))
        n0 = np.stack((n0, n1))
        print(files_list[i + 1])

    print()

    with tifffile.TiffWriter(os.path.join(OUTPUT_DIR, files_list[i][:-7] + ".ome-tif")) as tif:
        if nr_channels == 1:
            tif.write(n0, metadata={'axes': 'ZYX'})
        elif nr_channels == 2:
            tif.write(n0, metadata={'axes': 'CZYX'})
