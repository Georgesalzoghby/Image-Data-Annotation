import os
import numpy as np
import pandas as pd
from tifffile import imread, imsave, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table, regionprops, marching_cubes
from skimage.segmentation import clear_border

input_dir = 'C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged'
output_dir = input_dir + '_matpython'
output_properties = ('label', 'area', 'filled_area', 'major_axis_length', 'minor_axis_length', 'centroid',
                     'weighted_centroid', 'equivalent_diameter', 'max_intensity', 'mean_intensity',
                     'min_intensity', 'coords')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

min_volume = 200  # Minimum volume for the regions
def filter_small_regions(labels, min_volume):
    for region in regionprops(labels):
        # take regions with large enough areas
        if region.area < min_volume:
            labels[region.coords] = 0
    return labels
# with open(os.path.join(Input, 'img_original.tif'),mode='r') as img_original:
#     img_original = imread("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged\\20190720_RI512_CTCF-AID_AUX-CTL_61b_61a_SIR_2C_ALN_THR_1.ome-tif")

img_raw = imread("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged\\20190720_RI512_CTCF-AID_AUX-CTL_61b_61a_SIR_2C_ALN_THR_1.ome-tif")
img_labels = np.zeros_like(img_raw, dtype='int32')


for channel_index, channel_raw in enumerate(img_raw):  #this order (starting by channel number) is not defined by default
    channel_filtered = gaussian(channel_raw, sigma=0.5)
    channel_thresholded = channel_filtered > threshold_otsu(channel_filtered)
    # imsave("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged_matpython\\20190720_RI512_CTCF-AID_AUX-CTL_61b_61a_SIR_2C_ALN_THR_thre_1.ome-tif", channel_thresholded)
    img_labels[channel_index] = label(channel_thresholded)
    img_labels[channel_index] = clear_border(img_labels[channel_index])
    img_labels[channel_index] = filter_small_regions(img_labels[channel_index], min_volume=min_volume)
    channel_properties_dict = regionprops_table(label_image= img_labels[channel_index],
                                                intensity_image=channel_raw,
                                                properties=output_properties)

    table = pd.DataFrame(channel_properties_dict)
    print(table)
    imsave("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged_matpython\\20190720_RI512_CTCF-AID_AUX-CTL_61b_61a_SIR_2C_ALN_THR_labels_1.ome-tif", img_labels)
    print(img_labels.shape)

 # for prop in img_region:
 #     print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
 # img_region = np.array(regionprops(img_labeled))

