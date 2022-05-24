import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tifffile import imread, imsave, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table, regionprops, marching_cubes
from skimage.segmentation import clear_border

input_dir = 'C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged'
output_dir = input_dir + '_matpython'
output_properties = ('label', 'area', 'area_filled', 'axis_major_length', 'axis_minor_length', 'centroid',
                     'centroid_weighted', 'eccentricity', 'equivalent_diameter_area', 'intensity_max', 'intensity_mean',
                     'intensity_min', 'solidity')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

min_volume = "200"  # Minimum volume for the regions

# with open(os.path.join(Input, 'img_original.tif'),mode='r') as img_original:
#     img_original = imread("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged\\20190720_RI512_CTCF-AID_AUX-CTL_61b_61a_SIR_2C_ALN_THR_1.ome-tif")

img_raw = imread("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged\\20190720_RI512_CTCF-AID_AUX-CTL_61b_61a_SIR_2C_ALN_THR_1.ome-tif")


def filter_small_regions(labels, min_volume):
    for region in regionprops(labels):
        # take regions with large enough areas
        if region.area < min_volume:
            labels[region.coords] = 0
    return labels


for channel_raw in img_raw:  #this order (starting by channel number) is not defined by default
    channel_filtered = gaussian(channel_raw, sigma=0.5)    #numpy.amax (3)
    channel_thresholded = channel_filtered > threshold_otsu(channel_filtered)
    # imsave("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged_matpython\\20190720_RI512_CTCF-AID_AUX-CTL_61b_61a_SIR_2C_ALN_THR_thre_1.ome-tif", channel_thresholded)
    channel_labels = label(channel_thresholded)
    channel_labels = clear_border(channel_labels)
    channel_labels = filter_small_regions(channel_labels, min_volume=min_volume)
    channel_properties_dict = regionprops_table(label_image=channel_labels,
                                                intensity_image=channel_raw,
                                                properties=output_properties)

    table = pd.DataFrame(channel_properties_dict)
    print(table)
 # for prop in img_region:
 #     print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
 # img_region = np.array(regionprops(img_labeled))

