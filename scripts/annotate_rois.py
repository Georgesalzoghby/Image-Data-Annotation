import os
from getpass import getpass
import subprocess

import pandas as pd
from omero.gateway import BlitzGateway, ColorHolder
from omero.rtypes import rdouble, rint, rstring
from omero.model import RoiI, MaskI
import omero_rois
import numpy as np
from tifffile import tifffile


INPUT_DIRS = {
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID_AUX': 28,
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID_AUX-CTL': 15,
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC': 27,
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA': 30,
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA-CTL': 31,
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ncxNPC': 33,
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/NPC': 32,
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/RAD21-AID_AUX': 29,
    # '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/RAD21-AID_AUX-CTL': 8
}


def annotate_image(image_id, image_table_path):
    cmd = ["omero", "metadata", "populate",
           "--wait", "-1",
           "--allow-nan",
           "--file",
           image_table_path,
           f"Image:{image_id}"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    except subprocess.CalledProcessError as e:
        print(f'Input command: {cmd}')
        print()
        print(f'Error: {e.output}')
        print()
        print(f'Command: {e.cmd}')
        print()


def create_roi(img, shapes, name):
    # create an ROI, link it to Image
    roi = RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    roi.setName(rstring(name))
    for shape in shapes:
        # shape.setTextValue(rstring(name))
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return updateService.saveAndReturnObject(roi)


def masks_from_labels_image_3d(
        labels_3d, rgba=None, c=None, t=None, text=None,
        raise_on_no_mask=True):  # sourcery skip: low-code-quality
    """
    Create a mask shape from a binary image (background=0)

    :param numpy.array labels_3d: labels 3D array
    :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
    :param c: Optional C-index for the mask
    :param t: Optional T-index for the mask
    :param text: Optional text for the mask
    :param raise_on_no_mask: If True (default) throw an exception if no mask
           found, otherwise return an empty Mask
    :return: An OMERO mask
    :raises NoMaskFound: If no labels were found
    :raises InvalidBinaryImage: If the maximum labels is greater than 1
    """
    rois = {}
    for i in range(1, labels_3d.max() + 1):
        if not np.any(labels_3d == i):
            continue

        masks = []
        bin_img = labels_3d == i
        # Find bounding box to minimise size of mask
        xmask = bin_img.sum(0).sum(0).nonzero()[0]
        ymask = bin_img.sum(0).sum(1).nonzero()[0]
        if any(xmask) and any(ymask):
            x0 = min(xmask)
            w = max(xmask) - x0 + 1
            y0 = min(ymask)
            h = max(ymask) - y0 + 1
            submask = bin_img[:, y0:(y0 + h), x0:(x0 + w)]
        else:
            if raise_on_no_mask:
                raise omero_rois.NoMaskFound()
            x0 = 0
            w = 0
            y0 = 0
            h = 0
            submask = []

        for z, plane in enumerate(submask):
            if np.any(plane):
                mask = MaskI()
                # BUG in older versions of Numpy:
                # https://github.com/numpy/numpy/issues/5377
                # Need to convert to an int array
                # mask.setBytes(np.packbits(submask))
                mask.setBytes(np.packbits(np.asarray(plane, dtype=int)))
                mask.setWidth(rdouble(w))
                mask.setHeight(rdouble(h))
                mask.setX(rdouble(x0))
                mask.setY(rdouble(y0))
                mask.setTheZ(rint(z))

                if rgba is not None:
                    ch = ColorHolder.fromRGBA(*rgba)
                    mask.setFillColor(rint(ch.getInt()))
                if c is not None:
                    mask.setTheC(rint(c))
                if t is not None:
                    mask.setTheT(rint(t))
                if text is not None:
                    mask.setTextValue(rstring(text))

                masks.append(mask)

        rois[i] = masks

    return rois


def rois_from_labels_3d(img, labels_3d, rois_table=None, rgba=None, c=None, t=None, text=None):
    rois = masks_from_labels_image_3d(labels_3d, rgba=rgba, c=c, t=t,
                                      raise_on_no_mask=False)

    for label, masks in rois.items():
        if len(masks) > 0:
            if c is None:
                roi_name = f'{text}_label-{label}'
            else:
                roi_name = f'ch-{c}_{text}_label-{label}'
            roi = create_roi(img=img, shapes=masks, name=roi_name)
            rois_table['roi'].loc[rois_table['Roi Name'] == roi_name] = roi.id.val


try:
    conn = BlitzGateway(username=input('username: '),
        passwd=getpass('password: '),
        host="bioimage.france-bioinformatique.fr",
        port=4075,
        secure=True)
    conn.connect()

    updateService = conn.getUpdateService()
    session_key = conn.getSession().getUuid().getValue()

    for input_dir, dataset_id in INPUT_DIRS.items():

        dataset = conn.getObject('Dataset', dataset_id)

        images = dataset.listChildren()

        for image in images:
            image_name = image.getName()
            # if image_name != "20190720_RI512_CTCF-AID_AUX_61b_62_SIR_2C_ALN_THR_10.ome.tiff":
            #     continue
            print(f"annotating image: {image_name}")
            try:
                image_table_path = os.path.join(input_dir, f"{image_name[:-9]}_table.csv")
            except Exception as e:
                raise e

            image_table = pd.read_csv(image_table_path)
            image_table['roi'] = None

            # Domains
            try:
                domains_img = tifffile.imread(os.path.join(input_dir, f"{image_name[:-9]}_domains-ROIs.ome.tiff"))
                if domains_img.ndim == 4:
                    domains_img = domains_img.transpose((1, 0, 2, 3))
                elif domains_img.ndim == 3:
                    domains_img = np.expand_dims(domains_img, 0)

                for c, channel_labels in enumerate(domains_img):
                    if c == 0:
                        rgba = (255, 0, 0, 30)
                    elif c == 1:
                        rgba = (0, 255, 0, 30)
                    else:
                        rgba = (0, 0, 255, 30)

                    rois_from_labels_3d(img=image,
                                        labels_3d=channel_labels,
                                        rois_table=image_table,
                                        rgba=rgba,
                                        c=c,
                                        text='domain')
            except FileNotFoundError:
                pass

            # Subdomains
            try:
                subdomains_img = tifffile.imread(os.path.join(input_dir, f"{image_name[:-9]}_subdomains-ROIs.ome.tiff"))
                if subdomains_img.ndim == 4:
                    subdomains_img = subdomains_img.transpose((1, 0, 2, 3))
                elif subdomains_img.ndim == 3:
                    subdomains_img = np.expand_dims(subdomains_img, 0)
                for c, channel_labels in enumerate(subdomains_img):
                    if c == 0:
                        rgba = (180, 0, 0, 50)
                    elif c == 1:
                        rgba = (0, 180, 0, 50)
                    else:
                        rgba = (0, 0, 180, 50)

                    rois_from_labels_3d(img=image,
                                        labels_3d=channel_labels,
                                        rois_table=image_table,
                                        rgba=rgba,
                                        c=c,
                                        text='subdomain')
            except FileNotFoundError:
                pass

            # Overlaps
            try:
                overlap_img = tifffile.imread(os.path.join(input_dir, f"{image_name[:-9]}_overlap-ROIs.ome.tiff"))
                rgba = (0, 0, 255, 80)
                rois_from_labels_3d(img=image,
                                    labels_3d=overlap_img,
                                    rois_table=image_table,
                                    rgba=rgba,
                                    text='overlap')
            except FileNotFoundError:
                pass

            # Annotate image with table
            csv_header = "# header "
            for dt in list(image_table.dtypes)[:-1]:
                if dt == np.int64:
                    csv_header += 'l,'
                elif dt == np.float64:
                    csv_header += 'd,'
                else:
                    csv_header += 's,'
            csv_header += 'roi'

            image_table_header_path = os.path.join(input_dir, f"{image_name[:-9]}_table_headers.csv")
            with open(image_table_header_path, mode='w') as csv_file:
                csv_file.write(csv_header)
                csv_file.write("\n")
                csv_file.write(image_table.to_csv(index=False, line_terminator='\n'))

            try:
                annotate_image(image.getId(),
                               image_table_path=image_table_header_path)
            except Exception as e:
                raise e

except Exception as e:
    print(e)

finally:
    conn.close()
    print('done')
