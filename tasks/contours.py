import cv2
import numpy as np

from tools import image_util
from tools import geom
from tools import box_util


def contour(image, r_mask) :
    #image_util.show(image, 'roi')

    mask = np.zeros((image.shape[0],image.shape[1], 1), np.uint8)
    mask[r_mask] = 255

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image

    cont = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cont)

    #test = image.copy()
    #test = cv2.drawContours(test, [hull], -1, (0, 255, 0))
    #image_util.show(test, "contour")

    param = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.05*param, True)

    if len(approx) < 4 :
        return None

    # vertices in order tl bl br tr
    vertices = geom.get_vertices(approx)
    rectif = geom.rectify(vertices)

    transform, _ = cv2.findHomography(vertices, rectif)
    if transform is None :
        return image
    warped_image = cv2.warpPerspective(image, transform, (rectif[2, 0], rectif[2, 1]))

    #image_util.show(warped_image, "warped cut")

    return warped_image


def find_countours(image, box, mask) :
    roi = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    mask_cut = mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    cut = contour(roi, mask_cut)
    if cut is None :
        return None
    return cut
