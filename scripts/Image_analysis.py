import json
import os
from math import sqrt

import numpy as np
import pandas as pd
import xtiff
from scipy import ndimage
from tifffile import imread
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.segmentation import clear_border, watershed, relabel_sequential
from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max
from porespy.metrics import regionprops_3D

# Input and output directories
INPUT_DIR_LIST = [
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID_AUX-CTL',
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID_AUX',
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC',
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA',
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA-CTL',
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ncxNPC',
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/NPC',
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/RAD21-AID_AUX',
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/RAD21-AID_AUX-CTL'
    ]

# Properties to measure
DOMAIN_PROPERTIES = (
    'label',
    'area',
    'filled_area',
    'major_axis_length',
    'centroid',
    'weighted_centroid',
    'equivalent_diameter',
    'max_intensity',
    'mean_intensity',
    'min_intensity',
    # 'coords',
)
SUBDOMAIN_PROPERTIES = (
    'label',
    'area',
    'filled_area',
    'major_axis_length',
    'centroid',
    'weighted_centroid',
    'max_intensity',
    'mean_intensity',
    'min_intensity',
    # 'coords',
)
OVERLAP_PROPERTIES = (
    'label',
    'area',
    'filled_area',
    'centroid',
    # 'coords',
)

# Analysis constants
IMAGE_FILE_EXTENSION = "ome.tiff"
DOMAIN_MIN_VOLUME = 200  # Minimum volume for the regions
SUBDOMAIN_MIN_VOLUME = 36  # Minimum volume for the regions
SUBDOMAIN_MAX_NR = 17  # Maximum number of subdomains. Obtained from Quentins data
SIGMA = 0.5
PIXEL_SIZE = (.125, .04, .04)  # as ZYX
VOXEL_VOLUME = np.prod(PIXEL_SIZE)

markers_list = []

footprint = np.array([[[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0]],
                      [[0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0]],
                      [[0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0]],
                      [[0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0]],
                      [[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0]]])


# Function definitions
def process_channel(channel: np.ndarray, properties: tuple, subdomain_properties: tuple,
                    sigma: float = None, min_volume: int = None,
                    subdomain_min_volume: int = None, binarize: bool = True):
    # Preprocessing
    if sigma is None:
        filtered = channel
    else:
        filtered = gaussian(channel, sigma=sigma, preserve_range=True).astype('uint16')

    # Detecting Domains
    thresholded = filtered > threshold_otsu(filtered)
    domain_labels = label(thresholded)
    domain_labels = clear_border(domain_labels)
    if min_volume is not None:
        domain_labels = domain_labels > 0
        domain_labels = remove_small_objects(domain_labels, connectivity=domain_labels.ndim, min_size=min_volume)
        domain_labels = relabel_sequential(domain_labels.astype('uint8'))[0]
    if binarize:
        domain_labels = domain_labels > 0
        domain_labels = domain_labels.astype('uint8')
    domain_props_dict = regionprops_table(label_image=domain_labels, intensity_image=channel,
                                          properties=properties)
    domain_props_df = pd.DataFrame(domain_props_dict)
    pore_props_3d = regionprops_3D(domain_labels)
    domain_props_df["sphericity"] = 0
    domain_props_df["solidity"] = 0
    for lab in pore_props_3d:
        domain_props_df.loc[domain_props_df.label == lab.label, "sphericity"] = lab.sphericity
        domain_props_df.loc[domain_props_df.label == lab.label, "solidity"] = lab.solidity
    domain_props_df.insert(loc=0, column='roi_type', value='domain')

    # Detecting Subdomains
    coords = peak_local_max(channel,
                            min_distance=3,
                            threshold_abs=threshold_otsu(channel),
                            num_peaks=SUBDOMAIN_MAX_NR,
                            # footprint=footprint,
                            footprint=np.ones((3, 3, 3)),
                            labels=domain_labels
                            )
    mask = np.zeros(channel.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)

    subdomain_labels = watershed(-channel,
                                 markers=markers,
                                 mask=domain_labels
                                 )
    if subdomain_min_volume is not None:
        subdomain_labels = remove_small_objects(subdomain_labels, connectivity=subdomain_labels.ndim,
                                                min_size=subdomain_min_volume)
        subdomain_labels = relabel_sequential(subdomain_labels, offset=2)[0]
    subdomain_props_dict = regionprops_table(label_image=subdomain_labels, intensity_image=channel,
                                             properties=subdomain_properties)
    subdomain_props_df = pd.DataFrame(subdomain_props_dict)
    subdomain_props_df.insert(loc=0, column='roi_type', value='subdomain')

    # TODO: Add here nr of subdomains

    # Merging domain tables
    props_df = pd.concat([domain_props_df, subdomain_props_df], ignore_index=True)

    # Calculating some measurements
    props_df['volume'] = props_df['area'].apply(lambda a: a * VOXEL_VOLUME)
    props_df['volume_units'] = 'micron^3'

    return props_df, domain_labels, subdomain_labels


def process_overlap(labels, domains_df, overlap_properties):
    if labels.shape[0] == 2 and \
            np.max(labels[0]) == 1 and \
            np.max(labels[1]) == 1:

        overlap_labels = np.all(labels, axis=0).astype('uint8')
        overlap_props_dict = regionprops_table(label_image=overlap_labels,
                                               properties=overlap_properties)
        overlap_props_df = pd.DataFrame(overlap_props_dict)

        # if there is no overlap no rows are created. We nevertheless need to measure distance
        if len(overlap_props_df) == 0:
            overlap_props_df.loc[0, 'area'] = 0

        overlap_props_df.insert(loc=0, column='roi_type', value='overlap')

        overlap_props_df['volume'] = overlap_props_df['area'].apply(lambda a: a * VOXEL_VOLUME)
        overlap_props_df['volume_units'] = 'micron^3'

        # jaccard = (|A inter B| / (|A| + |B| - |A inter B|  ))
        overlap_props_df['overlap_fraction'] = abs(overlap_props_df.at[0,'volume']) / \
                            (abs(domains_df.loc[(domains_df['Channel ID'] == 0) & (domains_df['roi_type'] == 'domain'), 'volume'].values[0]) +
                             abs(domains_df.loc[(domains_df['Channel ID'] == 1) & (domains_df['roi_type'] == 'domain'), 'volume'].values[0]) -
                             abs(overlap_props_df.at[0,'volume']))

        overlap_props_df['distance_x'] = \
            abs(domains_df.loc[
                    (domains_df['Channel ID'] == 0) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-2'].values[0] - \
                domains_df.loc[
                    (domains_df['Channel ID'] == 1) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-2'].values[0]) * \
            PIXEL_SIZE[2]
        overlap_props_df['distance_y'] = \
            abs(domains_df.loc[
                    (domains_df['Channel ID'] == 0) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-1'].values[0] - \
                domains_df.loc[
                    (domains_df['Channel ID'] == 1) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-1'].values[0]) * \
            PIXEL_SIZE[1]
        overlap_props_df['distance_z'] = \
            abs(domains_df.loc[
                    (domains_df['Channel ID'] == 0) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-0'].values[0] - \
                domains_df.loc[
                    (domains_df['Channel ID'] == 1) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-0'].values[0]) * \
            PIXEL_SIZE[0]
        overlap_props_df['distance3d'] = sqrt(overlap_props_df.distance_x ** 2 +
                                              overlap_props_df.distance_y ** 2 +
                                              overlap_props_df.distance_z ** 2)
        overlap_props_df['distance_units'] = 'micron'

        # TODO: implement Matrix overlap

        return overlap_props_df, overlap_labels

    else:
        return None, None


def process_image(image, domain_properties, subdomain_properties, overlap_properties,
                  sigma=None, min_volume=None, subdomain_min_volume=None):
    rois_df = pd.DataFrame()

    domain_labels = np.zeros_like(image, dtype='uint8')
    subdomain_labels = np.zeros_like(image, dtype='uint8')

    # this order (starting by channel number) is not defined by default
    for channel_index, channel in enumerate(image):
        channel_props_df, channel_domain_labels, channel_subdomain_labels = \
            process_channel(channel=channel, properties=domain_properties,
                            subdomain_properties=subdomain_properties,
                            sigma=sigma, min_volume=min_volume,
                            subdomain_min_volume=subdomain_min_volume)

        domain_labels[channel_index] = channel_domain_labels
        subdomain_labels[channel_index] = channel_subdomain_labels

        channel_props_df.insert(loc=0, column='Channel ID', value=channel_index)

        rois_df = pd.concat([rois_df, channel_props_df], ignore_index=True)

    overlap_props_df, overlap_labels = process_overlap(labels=domain_labels,
                                                       domains_df=rois_df,
                                                       overlap_properties=overlap_properties)
    if overlap_props_df is not None:
        rois_df = pd.concat([rois_df, overlap_props_df], ignore_index=True)

    return rois_df, domain_labels, subdomain_labels, overlap_labels


def run(input_dir):
    output_dir = f'{input_dir}'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(input_dir, 'assay_config.json'), mode="r") as config_file:
        config = json.load(config_file)

    assay_id = config["assay_id"]

    files_list = [f for f in os.listdir(input_dir) if
                  f.endswith(IMAGE_FILE_EXTENSION) and
                  not f.endswith(f"ROIs.{IMAGE_FILE_EXTENSION}")]

    analysis_df = pd.DataFrame()

    for img_file in files_list:
        print(f'Processing image: {img_file}')
        image = imread(os.path.join(input_dir, img_file))
        if image.ndim == 4:  # More than 1 channel
            image = image.transpose((1, 0, 2, 3))
        elif image.ndim == 3:  # One channel
            # TODO: Verify that this dimension is added in the right axis
            image = np.expand_dims(image, 1)

        rois_df, domain_labels, subdomain_labels, overlap_labels = \
            process_image(image=image,
                          domain_properties=DOMAIN_PROPERTIES,
                          subdomain_properties=SUBDOMAIN_PROPERTIES,
                          overlap_properties=OVERLAP_PROPERTIES,
                          sigma=SIGMA,
                          min_volume=DOMAIN_MIN_VOLUME,
                          subdomain_min_volume=SUBDOMAIN_MIN_VOLUME
                          )

        rois_df.insert(loc=0, column='Image Name', value=img_file)

        xtiff.to_tiff(img=domain_labels.transpose((1, 0, 2, 3)),
                      file=os.path.join(output_dir, f'{img_file[:-9]}_domains-ROIs.ome.tiff')
                      )
        xtiff.to_tiff(img=subdomain_labels.transpose((1, 0, 2, 3)),
                      file=os.path.join(output_dir, f'{img_file[:-9]}_subdomains-ROIs.ome.tiff')
                      )
        if overlap_labels is not None:
            xtiff.to_tiff(img=np.expand_dims(overlap_labels, axis=1),
                          file=os.path.join(output_dir, f'{img_file[:-9]}_overlap-ROIs.ome.tiff')
                          )

        analysis_df = pd.concat([analysis_df, rois_df], ignore_index=True)

    analysis_df.to_csv(os.path.join(output_dir, 'analysis_df.csv'))

    metadata_df = pd.read_csv(os.path.join(input_dir, f"{assay_id}_assays.csv"), header=1)  # TODO:

    merge_df = pd.merge(metadata_df, analysis_df, on="Image Name")
    merge_df.to_csv(os.path.join(output_dir, 'merged_df.csv'))


if __name__ == '__main__':
    for input_dir in INPUT_DIR_LIST:
        run(input_dir)
    print("done")

