import json
import os
from math import sqrt

import numpy as np
import pandas as pd
import xtiff
from tifffile import imread, imsave, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from porespy.metrics import regionprops_3D

# from sklearn.metrics import jaccard_score

# Input and output directories
INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID_merged'
OUTPUT_DIR = f'{INPUT_DIR}_analysis'

# Properties to measure
ROI_PROPERTIES = ('label', 'area', 'filled_area',
                  'major_axis_length',  # 'minor_axis_length',
                  'centroid',
                  'weighted_centroid', 'equivalent_diameter', 'max_intensity', 'mean_intensity',
                  'min_intensity', 'coords')
OVERLAP_PROPERTIES = ('label', 'area', 'filled_area', 'centroid',
                      'coords')

# Analysis constants
MIN_VOLUME = 200  # Minimum volume for the regions
SIGMA = 0.5
VOXEL_VOLUME = 0.04 * 0.04 * 0.125
PIXEL_SIZE = (.125, .04, .04)  # as ZYX


# Function definitions
def process_channel(channel, properties, sigma=None, min_volume=None, binarize=True):
    if sigma is not None:
        channel = gaussian(channel, sigma=sigma, preserve_range=True).astype('uint16')
    thresholded = channel > threshold_otsu(channel)
    labels = label(thresholded)
    labels = clear_border(labels)
    if min_volume is not None:
        labels = remove_small_objects(labels, connectivity=labels.ndim, min_size=min_volume)
    if binarize:
        labels = labels > 0
        labels = labels.astype('uint8')
    properties_dict = regionprops_table(label_image=labels, intensity_image=channel,
                                        properties=properties)
    properties_table = pd.DataFrame(properties_dict)
    pore_props_3d = regionprops_3D(labels)
    properties_table["sphericity"] = 0
    properties_table["solidity"] = 0
    for l in pore_props_3d:
        properties_table.loc[properties_table.label == l.label, "sphericity"] = l.sphericity
        properties_table.loc[properties_table.label == l.label, "solidity"] = l.solidity

    return labels, properties_table


if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
    config = json.load(config_file)

nr_channels = config["nr_channels"]
files_list = [f for f in os.listdir(INPUT_DIR) if f.endswith('.tiff')]

analysis_df = pd.DataFrame()

for img_file in files_list:

    img_raw = imread(os.path.join(INPUT_DIR, img_file))
    img_raw = img_raw.transpose((1, 0, 2, 3))

    img_labels = np.zeros_like(img_raw, dtype='uint8')

    image_df = pd.DataFrame.from_dict({'Image Name': [img_file]})

    for channel_index, channel_raw in \
            enumerate(img_raw):  # this order (starting by channel number) is not defined by default
        channel_labels, channel_properties_table = process_channel(channel_raw, ROI_PROPERTIES,
                                                                   sigma=SIGMA, min_volume=MIN_VOLUME)

        img_labels[channel_index] = channel_labels

        channel_properties_table['volume'] = channel_properties_table['filled_area'].apply(lambda a: a * VOXEL_VOLUME)
        channel_properties_table['volume_units'] = 'micron^3'
        channel_properties_table = channel_properties_table.add_prefix(f"ch-{channel_index}_")

        channel_properties_table.insert(loc=0, column='Image Name', value=img_file)
        image_df = pd.merge(image_df, channel_properties_table, on='Image Name', how='outer')

    xtiff.to_tiff(img=img_labels.transpose((1, 0, 2, 3)),
                  file=os.path.join(OUTPUT_DIR, f'{img_file[:-9]}_channel-linked-ROIs.ome.tiff')
                  )

    overlap_img = np.all(img_labels, axis=0)
    xtiff.to_tiff(img=np.expand_dims(overlap_img, axis=1),
                  file=os.path.join(OUTPUT_DIR, f'{img_file[:-9]}_unlinked-ROIs.ome.tiff')
                  )

    overlap_labels = label(overlap_img)
    overlap_properties_dict = regionprops_table(label_image=overlap_labels,
                                                properties=OVERLAP_PROPERTIES)
    overlap_properties_table = pd.DataFrame(overlap_properties_dict)
    overlap_properties_table['volume'] = overlap_properties_table['filled_area'].apply(lambda a: a * VOXEL_VOLUME)
    overlap_properties_table['volume_units'] = 'micron^3'

    overlap_properties_table = overlap_properties_table.add_prefix("overlap_")
    overlap_properties_table.insert(loc=0, column='Image Name', value=img_file)

    image_df = pd.merge(image_df, overlap_properties_table, on='Image Name')
    try:
        image_df['distance_x'] = abs(image_df['ch-0_weighted_centroid-2'] - image_df['ch-1_weighted_centroid-2']) * PIXEL_SIZE[2]
        image_df['distance_y'] = abs(image_df['ch-0_weighted_centroid-1'] - image_df['ch-1_weighted_centroid-1']) * PIXEL_SIZE[1]
        image_df['distance_z'] = abs(image_df['ch-0_weighted_centroid-0'] - image_df['ch-1_weighted_centroid-0']) * PIXEL_SIZE[0]
        image_df['distance3d'] = sqrt(image_df.distance_x ** 2 + image_df.distance_y ** 2 + image_df.distance_z ** 2)
    except KeyError:
        pass

    analysis_df = pd.concat([analysis_df, image_df], ignore_index=True)

analysis_df.to_csv(os.path.join(OUTPUT_DIR, 'analysis_df.csv'))

# imsave("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged_matpython\\20190720_RI512_CTCF-AID_AUX-CTL_61b_61a_SIR_2C_ALN_THR_labels_1.ome-tif", img_labels)
# print(img_labels.shape)

# for prop in img_region:
#     print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
# img_region = np.array(regionprops(img_labeled))


#
# n files is the length of the image c1
# if nC == 1
# missubdomainvolume = 36
#
# ######
# alldistmatrixoverlap = [];
# 3D-sim = 1 ; pixel size xy:0.04 | Image size xy:55
# conv WF = 0 ; pixel size xy:0.08 | Image size xy:28
# zsize 0.125
# minimum volume = 200
#
# ##Output summary files
# if nC == 2
# alldistmatrixoverlap = [];
#
# ### Reading, filtering and segmentation of images
#
# filtered image : gaussian filter (imgaussfilt3) with sigma = 0.5
# thresholded image : threshold_otsu (graythresh/imbinarize) #Binarize 2-D grayscale image or 3-D volume by thresholding
#
# ### extraction of segmented objects
#
# labeled image : measure.label (bwconncomp) #Find and count connected components in binary image
# region image : measure.regionprops (regionprops3) # Measure properties of 3-D volumetric image regions
# find image object size > minimumn volume
# mask ismember(labelmatrix)
# mask imclearborder #Suppress light structures connected to image border
#
# if nC == 1
# regionprops(mask, vol, centroid, principalaxislength, surfacearea)
# if nC == 2
# regionprops(mask, centroid)
#
# only keep images with 1 segmented object when centroid is not equal to 1 and then
# jump to the next image file
#
#
#
#
# ### Intermingling between C1 and C2 channels
# if number of channels= 2
# maskmerge by adding the 2 masks, maskoverlap if maskmerge bigger than 1 , projmaskoverlap
#
# Overlap fraction :Jaccard similarity coefficient for image segmentation (mask1,mask2)
# overlapfractionsummary
# --> python: from sklearn.metrics import jaccard_score
#
# ## 3D distance between centroids in um ; which type M?
# xdistance = ( centroid1 of c2 - centroid1 of c1 ) * xypixelsize (0.04 | 0.08)
# y
# z
# distance = sqrt ( xdistance^2 + ydistance^2 + zdistance^2 )
#
# ## Matrix overlap
# statsprojmask2 = regionprops ( 'table', projmask C1 | C2 , 'area', 'centroid' )
# rounding to nearest decimal or integer ( xshift = nx/2 - statsprojmask2.centroid(1) then y for centroid (2)
# shiftmatrix = circshift: shift array circularly
# mergeimc1 = zeros (x,y) --> mergeimc1 = imadd(mergeimc1, double(shiftmatrix))
# double: Convert symbolic values to MATLAB double precision
# imadd: Add two images or add constant to image
#
# dist2D = sqrt((mask2c1.centroid(1)-mask2c2.centroid(1))^2 - (mask2c1.centroid(2)-mask2c2.centroid(2))^2
# alldistmatrixoverlap = [alldistmatrixoverlap; dist2D];
#
#
# ##  Probe structure analysis
#  only for number of channels= 1
# voxelvol = xy * xy * z
#
# #volume in um3
# vol = statsc1.vol(1) * voxelvol
#
# principapaxislength in um
# xdistc1 = statsc1.principalAxisLength(1) * xy
# ydistc1 = statsc1.principalAxisLength(2) * xy
# zdistc1 = statsc1.principalAxisLength(3) * z
# principalAxisLength = sqrt(xdistc1^2 + ydistc1^2 + zdistc1^2)
#
# #sphericity formula; surface area to volume
#
# ##waterrshed segmentation
# image of zeros = im
# mat2gray(im) :Convert matrix to grayscale image
# imA not im
# Inf: Create array of all Inf values
# A = watershed(imA) ; A not mask = 0
#
# connectedcA = measure.labels(A)
# prestatA = regionprops3(connectedcA, 'vol')
# mask idxA if prestatA.vol > minsubdomainvolume
# statsA = regionprops3(maskA, Vol, Centroid, ConvexHull)
# subdomaincentroid = stasA.Centroid
#
# #subdomainvolume = statsA{:,1}.* voxelvol
# table
#
# #nsubdomains = size(statsA,1) #array size
# table
#
# ## Output images
# images with segmentation borders
#
# adjprojC1 = imadjust(mat2gray(projC1),[low out .05;high out 1]) projC1 == channel filtered without max  #imadjust: Adjust image intensity values or colormap
# if nC == 2 --> adjprojC2 then overlayC1C2 = imfuse(adjprojC1,adjprojC2) ##Composite of two images
#
# #bwboundaries: Trace region boundaries in binary image
# bound1 = skimage.segmentation.find_boundaries(projmask2C1, connectivity=1, mode='thick', background=0)
# if nC == 2 --> bound2(projmask2C2) and bound3(projmaskoverlap)
# #num2str: Convert numbers to character array
# of = OF = 'space'num2str(overlapfraction,precision 3) #precision isMaximum number of significant digits
#
#
# figure (position , coord)
# set(gcf, 'color', 'w') #set(H,NameArray,ValueArray)
# ax = gca Current axes or chart
# Axes Position-Related Properties
#
# Set graphics object properties
#
# if nc == 1 --> imshow(adjprojC1)
# colb = y
# else
# imshow(overlayC1C2)
# cold =magenta
#
# for p = 1:length(bound1) ; boundary C1 = bound1{p} ; plot(boundaryC1(:,2), boundaryC1(:,1), 'color', colb, 'LineWidth', 2.25)
# if nC == 2
# for q = 1:length(bound2) ;  boundary C2 = bound1{q}; plot(boundaryC2(:,2), boundaryC1(:,1),'color', 'green', 'LineWidth', 2.25)
# end
# for r = 1:length(bound3) ; boundary C1 = bound1{r} ; plot(boundaryC3(:,2), boundaryC1(:,1), 'color', 'white', 'LineWidth', 3.25)
# end
# text(x,y, of,'Color', 'white','FontSize', 24)
#
# if nC == 2
# strrep: Find and replace substrings
# '_C1_C2_segmented.tif'
# else
# '_' + channel + '_segmented.tif'
#
# Outputimagename = fullfile(outputsubfolder,imname) #Build full file name from parts
# print (outputimagename, '-dtiff' , '-r300')
#
# ##summary files
# if nC==1  # Write table to file
# writetable(volumesummary,fullfile(outputsubfolder,volumesammaryname))
# writetable(principleaxieslegnthsummary,..
# writetable(sphericitysummary,..
# writetable(subdomainvolumesummary,..
# writetable(nsubdomainssummary,..
#
# if nC==2
# writetable(overlapfractionsummary,..
# writetable(distancecentroidsummary,..
#
# #averaged image if nC==2
# shiftoverlap = round(mean(alldistmatrixoverlap)/2)
#
