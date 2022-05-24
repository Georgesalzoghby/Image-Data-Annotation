import os
import json
import numpy as np
import xml.etree.ElementTree as ET
import tifffile
import xtiff


channel_name_mapping = {'683.0': 'Alexa-647', '608.0': 'ATTO-555', '435.0': 'DAPI'}

# image = tifffile.TiffFile('scripts/20200726_RI512_RAD21-AID-AUX_61b-647_61a-565_DAPI_001_SIR.dv.ome.tif')
# root = ET.fromstring(image.ome_metadata)
# for channel in root.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel'):
#     channel.set('Name', channel_name_mapping[channel.get('Name')])
# for channel in root.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels'):
#     channel.set('SizeX', 55)
# for channel in root.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels'):
#     channel.set('SizeY', 55)


INPUT_DIR = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID"
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
    if nr_channels == 1:
        n0 = np.expand_dims(n0, 0)
    if nr_channels == 2:
        n1 = tifffile.imread(os.path.join(INPUT_DIR, files_list[i+1]))
        n0 = np.stack((n0, n1))
        print(files_list[i + 1])

    print()
    options = dict(photometric='minisblack', metadata={'axes': 'CZYX'})

    xtiff.to_tiff(img=n0.transpose((1, 0, 2, 3)),
                  file=os.path.join(OUTPUT_DIR, files_list[i][:-7] + ".ome.tiff"),
                  image_date="2019:05:29 07:11:22",
                  channel_names=["Alexa-647", "ATTO-555"],
                  description="My description",
                  pixel_size=0.03999,
                  pixel_depth=0.125,
                  )
    # with tifffile.TiffWriter(os.path.join(OUTPUT_DIR, files_list[i][:-7] + ".ome.tif")) as tif:
    #     tif.write(n0, **options)
                # photometric='minisblack',
                # metadata={
                #     'axis': 'CZYX',
                #     'channels': nr_channels,
                #     'slices': 1,
                #     'frames': n0.shape[1] * nr_channels,
                #     'hyperstack': True,
                #     'loop': False
                #     })
