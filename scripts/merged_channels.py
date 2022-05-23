import os
import json
import numpy as np
# import xml.etree.ElementTree as ET
import tifffile
import xtiff
from getpass import getpass
from omero.gateway import BlitzGateway

from meta_data.crop_to_raw import crop_to_raw, crop_to_SIR

CHANNEL_NAME_MAPPINGS = {'683.0': 'Alexa-647', '608.0': 'ATTO-555', '435.0': 'DAPI'}

INPUT_DIR = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID"
OUTPUT_DIR = INPUT_DIR + '_merged'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
    config = json.load(config_file)

nr_channels = config["nr_channels"]
files_set = {f[:-7] for f in os.listdir(INPUT_DIR) if f.endswith('.tif')}

try:
    # image = tifffile.TiffFile('scripts/20200726_RI512_RAD21-AID-AUX_61b-647_61a-565_DAPI_001_SIR.dv.ome.tif')
    # root = ET.fromstring(image.ome_metadata)
    # for channel in root.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel'):
    #     channel.set('Name', channel_name_mapping[channel.get('Name')])
    # for channel in root.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels'):
    #     channel.set('SizeX', 55)
    # for channel in root.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels'):
    #     channel.set('SizeY', 55)

    conn = BlitzGateway(username=input("username: "),
                        passwd=getpass("password: "),
                        host="omero.mri.cnrs.fr",
                        port=4064,
                        group="Cavalli Lab")
    conn.connect()

    # TODO: This line is extracting the metadata into a file
    # ./bftools/showinf -nopix -nocore -omexml ./Image/701139/Binary/20190720_RI512_ES-CTL_61b-647_61a-565_DAPI_001_SIR.dv > metadata.xml

    for file_root in files_set:
        image = tifffile.imread(os.path.join(INPUT_DIR, f"{file_root}_C1.tif"))

        for ch in range(1, nr_channels):
            new_channel = tifffile.imread(os.path.join(INPUT_DIR, f"{file_root}_C{ch + 1}.tif"))
            image = np.stack((image, new_channel))
        if nr_channels == 1:
            image = np.expand_dims(image, 0)

        print(f"Processing file {file_root}")

        raw_image_id = crop_to_raw["_".join(file_root.split("_")[:-1])]
        sir_image_id = crop_to_SIR["_".join(file_root.split("_")[:-1])]

        raw_image = conn.getObject('Image', raw_image_id)
        sir_image = conn.getObject('Image', sir_image_id)

        options = dict(photometric='minisblack', metadata={'axes': 'CZYX'})

        channel_names = [CHANNEL_NAME_MAPPINGS[str(raw_image.getChannelLabels()[c])] for c in range(nr_channels)]

        xtiff.to_tiff(img=image.transpose((1, 0, 2, 3)),
                      file=os.path.join(OUTPUT_DIR, file_root + ".ome.tiff"),
                      image_date=str(raw_image.getDate()),
                      channel_names=channel_names,
                      description=sir_image.getDescription(),
                      pixel_size=round(sir_image.getPixelSizeX(), 5),
                      pixel_depth=round(sir_image.getPixelSizeZ(), 5),
                      )

finally:
    conn.close()
