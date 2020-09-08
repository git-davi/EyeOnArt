from netloader import *


def detect(image):
    # apply segmentation prediction
    r = model.detect([image])[0]
    """
    Runs the detection pipeline.

    images: List of images, potentially of different sizes.

    Returns a list of dicts, one dict per image. The dict contains:
    rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    class_ids: [N] int class IDs
    scores: [N] float probability scores for the class IDs
    masks: [H, W, N] instance binary masks
    """
    
    boxes = []
    masks = []
    
    index = 0
    for score in r['scores']:
        if score < 0.7 :
            continue
        roi = r['rois'][index, :]
        box = [roi[1], roi[0], roi[3]-roi[1], roi[2]-roi[0]]
        boxes.append(box)
        masks.append(r['masks'][:, :, index])
        index += 1

    return boxes, masks