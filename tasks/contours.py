import cv2
import numpy as np
import time

from tools import image_util
from tools import geom
from tools import rectification_utils



def alt_Countours(image):
    img_countours = image.copy()
    gray_scale = cv2.GaussianBlur(image, (7, 7), 1)
    gray_scale = cv2.cvtColor(gray_scale, cv2.COLOR_BGR2GRAY)
    blank_image = np.zeros((gray_scale.shape[0],gray_scale.shape[1]), np.uint8)

    # experimental color reduction
    
    for i in range(blank_image.shape[0]):
        for j in range(blank_image.shape[1]):
            # from 256 to 8 shades of gray
            blank_image[i][j] = int((int(gray_scale[i][j] / 32))*32)

    img_canny = cv2.Canny(blank_image, 50, 20)
    kernel = np.ones((5, 5))
    img_dilated = cv2.dilate(img_canny, kernel, iterations=1)
    vertices = getContours(img_dilated, img_countours)

    if vertices is None:
        return None

    # UPDATE THIS FUNCTION the concept is the same but now we have to calc the angle of segments
    rect_points = geom.rectify_points(vertices)
    
    transform, _ = cv2.findHomography(vertices, rect_points)
    warped_image = cv2.warpPerspective(image, transform, (image.shape[1], image.shape[0]))

    rounded = np.round(rect_points).astype(int)
    rounded[rounded < 0] = 0

    image_util.show(warped_image)
    cut = warped_image[rounded[0, 1]:rounded[2, 1], rounded[0, 0]:rounded[2, 0]]
    cut = image_util.remove_border(cut, 0.15)
    image_util.show(cut)

    return cut
    

def getContours(src, out):
    #contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cont = max(contours, key=cv2.contourArea)
    cv2.drawContours(out, cont, -1, (255, 0, 255), 7)
    
    hull = cv2.convexHull(cont)

    hull_mask = np.zeros((src.shape[0],src.shape[1], 1), np.uint8)
    hull_mask = cv2.drawContours(hull_mask, [hull], -1, (255, 255, 255))

    param = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02*param, True)


    if len(approx) < 4 :
        return None

    # vertices in order tl bl br tr
    vertices = geom.get_vertices(approx)

    '''
    for pt in vertices:
        ziocan = (pt[0],pt[1])
        print(ziocan)
        hull_mask = cv2.circle(hull_mask, ziocan, 30, (255, 0, 0), 10)
    
    image_util.show(hull_mask)
    #approx_tup = [tuple(p[0]) for p in approx]
    '''

    return vertices


def find_countours(image, boxes) :
    rectified = []
    for box in boxes :
        roi = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        #rectified.append(contour(roi))
        if roi is not None:
            rectified.append(alt_Countours(roi))
