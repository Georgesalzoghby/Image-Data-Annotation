import os
import json
import numpy as np
import tifffile
import xtiff
from getpass import getpass
from omero.gateway import BlitzGateway
from meta_data.crop_to_raw import crop_to_raw, crop_to_SIR


# TODO: check here: https://forum.image.sc/t/python-bioformats-write-image-6d-images-series-handling/28633/4

CHANNEL_NAME_MAPPINGS = {'683.0': 'Alexa-647', '608.0': 'ATTO-555', '435.0': 'DAPI'}

INPUT_DIR = "/home/julio/PycharmProjects/Image-Data-Annotation/assays/CTCF-AID"
OUTPUT_DIR = INPUT_DIR + '_merged'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
    config = json.load(config_file)

nr_channels = config["nr_channels"]
files_set = {f[:-7] for f in os.listdir(INPUT_DIR) if f.endswith('.tif')}

try:
    conn = BlitzGateway(username=input("username: "),
                        passwd=getpass("password: "),
                        host="omero.mri.cnrs.fr",
                        port=4064,
                        group="Cavalli Lab")
    conn.connect()

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

        raw_image_name = raw_image.getName()

        # # Remove deteted DAPI
        # if nr_channels < omexml.image().Pixels.channel_count:
        #     pass
        #     # We check if it is DAPI before removal
        #     # if

        channel_names = [CHANNEL_NAME_MAPPINGS[str(raw_image.getChannelLabels()[c])] for c in range(nr_channels)]

        channels = raw_image.getChannels()
        excitation_wavelengths = [ch.getExcitationWave() for ch in channels[:nr_channels]]
        emission_wavelengths = [ch.getEmissionWave() for ch in channels[:nr_channels]]

        metadata = {
                    # 'axes': 'CZYX',
                    # # OME attributes
                    # 'UUID'
                    # 'Creator'
                    # OME.Image attributes
                    # 'Name'
                    # OME.Image elements
                    'AcquisitionDate': raw_image.getDate().date(),  # Remove time
                    # 'AcquisitionDate': raw_image.getDate(),  # Remove time
                    # 'AcquisitionDate': '2020-11-01T12:12:12',  # Remove time
                    'Description': "3D 3Beam SI",
                    # OME.Image.Pixels attributes:
                    # 'SignificantBits'
                    # 'PhysicalSizeX': 0.04,
                    # 'PhysicalSizeXUnit': 'µm',
                    # 'PhysicalSizeY': 0.04,
                    # 'PhysicalSizeYUnit': 'µm',
                    # 'PhysicalSizeZ': 0.125,
                    # 'PhysicalSizeZUnit': 'µm',
                    # 'TimeIncrement'
                    # 'TimeIncrementUnit'
                    'Plane': {
                              # 'ExposureTime': [.3] * (56 * nr_channels),
                              # 'ExposureTimeUnit': ['msec'] * (56 * nr_channels),
                              # 'PositionX'
                              # 'PositionXUnit'
                              # 'PositionY'
                              # 'PositionYUnit'
                              # 'PositionZ'
                              # 'PositionZUnit'
                              },
                    'Channel': {'Name': channel_names,
                                # 'AcquisitionMode'
                                # 'Color'
                                # 'ContrastMethod'
                                'EmissionWavelength': emission_wavelengths,
                                'EmissionWavelengthUnit': ['nm'] * nr_channels,
                                'ExcitationWavelength': excitation_wavelengths,
                                'ExcitationWavelengthUnit': ['nm'] * nr_channels,
                                # 'Fluor'
                                # 'IlluminationType'
                                # 'NDFilter': "",
                                # 'PinholeSize'
                                # 'PinholeSizeUnit'
                                # 'PockelCellSetting'
                                # 'SamplesPerPixel'
                                }
                    }
        #
        # with tifffile.TiffWriter(os.path.join(OUTPUT_DIR, file_root + ".ome.tif")) as tif:
        #     tif.write(image, **metadata)

        xtiff.to_tiff(img=image.transpose((1, 0, 2, 3)),
                      file=os.path.join(OUTPUT_DIR, file_root + ".ome.tiff"),
                      image_date=raw_image.getDate().date(),
                      channel_names=channel_names,
                      profile=xtiff.tiff.TiffProfile.OME_TIFF,
                      pixel_size=round(sir_image.getPixelSizeX(), 5),
                      pixel_depth=round(sir_image.getPixelSizeZ(), 5),
                      **metadata
                      )

finally:
    conn.close()

    print('Done')
