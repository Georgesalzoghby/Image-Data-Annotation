## Bettter to use the CLI

from getpass import getpass

from omero.gateway import BlitzGateway, ColorHolder

try:
    conn = BlitzGateway(username=input('username: '),
        passwd=getpass('password: '),
        host="bioimage.france-bioinformatique.fr",
        port=4075,
        secure=True)
    conn.connect()

    updateService = conn.getUpdateService()

    dataset = conn.getObject('Dataset', int(input("Dataset ID: ")))

    images = dataset.listChildren()
    roi_service = conn.getRoiService()

    for image in images:
        image_name = image.getName()
        print(f"deleting rois from image: {image_name}")

        result = roi_service.findByImage(image.id, None)
        roi_ids = [roi.id.val for roi in result.rois]
        conn.deleteObjects("Roi", roi_ids)

except Exception as e:
    print(e)

finally:
    conn.close()
    print('done')
