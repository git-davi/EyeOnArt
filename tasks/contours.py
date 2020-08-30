import cv2
import numpy as np

from tools import image_util
from tools import geom
from tools import box_util

from netloader import *


def contour(image) :

    # apply segmentation prediction
    r = model.detect([image], verbose=1)[0]

    mask = np.zeros((image.shape[0],image.shape[1], 1), np.uint8)
    mask[r['masks']] = 255

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    cont = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cont)

    #hull_mask = np.zeros((image.shape[0],image.shape[1], 1), np.uint8)
    #hull_mask = cv2.drawContours(hull_mask, [hull], -1, (255, 255, 255))
    
    param = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.05*param, True)

    if len(approx) < 4 :
        return None

    # vertices in order tl bl br tr
    vertices = geom.get_vertices(approx)
    rectif = geom.rectify(vertices)

    transform, _ = cv2.findHomography(vertices, rectif)
    if transform is None :
        return None
    warped_image = cv2.warpPerspective(image, transform, (rectif[2, 0], rectif[2, 1]))

    return warped_image


def find_countours(image, box) :
    roi = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    cut = contour(roi)
    if cut is None :
        return None
    return cut

def is_it_a_fucking_rombo(vertices):
    tl,tr,br,bl = vertices
    #altri controlli equivalenti br[1]-bl[1]
    if(abs(tl[1]-tr[1]) > 95):
        return True
    else:
        return False

def sort_rhombus(vertices):
    p1,p2,p3,p4=vertices
    tl,tr,br,bl = [None, None, None, None]
    min_x=10000
    max_x=0
    min_y=10000
    max_y=0

    for i in vertices:
        if (i[0] <= min_x):
            tl=i
            min_x=i[0]
    for i in vertices:
        if (i[0] >= max_x):
            br=i
            max_x=i[0]
    for i in vertices:
        if (i[1] <= min_y):
            tr=i
            min_y=i[1]
    for i in vertices:
        if (i[1] >= max_y):
            bl=i
            max_y=i[1]
    return np.array([tl, tr, br, bl])