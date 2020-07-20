import cv2
import numpy as np

from tools import image_util
from tools import geom



def contour(image) :
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    imgray_blurred = cv2.GaussianBlur(imgray, (5, 5), 1)
    edges = cv2.Canny(imgray_blurred, 50, 20)
    edges_dilated = cv2.dilate(edges, None, iterations=5)
    contours, hierarchy = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cont = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cont)

    hull_mask = np.zeros((image.shape[0],image.shape[1], 1), np.uint8)
    hull_mask = cv2.drawContours(hull_mask, [hull], -1, (255, 255, 255))
    #image_util.show(hull_mask)
    
    lines = cv2.HoughLines(hull_mask, 1, np.pi / 180, 100)

    if lines is None :
        # try matching with all roi
        return image

    # scale lines for kmeans
    lines[:, :, 0], min_rho, max_rho = geom.feature_scaling(lines[:, :, 0])
    lines[:, :, 1], min_theta, max_theta = geom.feature_scaling(lines[:, :, 1])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    try :
        _, _, cluster_lines = cv2.kmeans(lines, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    except Exception :
        return image
    
    # lines descaling
    cluster_lines[:, 0] = geom.feature_descaling(cluster_lines[:, 0], min_rho, max_rho )
    cluster_lines[:, 1] = geom.feature_descaling(cluster_lines[:, 1], min_theta, max_theta )
    
    # return lines in order t r b l
    ordered_lines = geom.order_lines(cluster_lines)
    horizontal_lines = np.array([ordered_lines[0], ordered_lines[2]])
    vertical_lines = np.array([ordered_lines[1], ordered_lines[3]])

    # debugging
    #image_util.draw_lines(horizontal_lines, vertical_lines, image)

    inters = geom.segmented_intersections(horizontal_lines,vertical_lines)

    #for coord in inters:
    #    cv2.drawMarker(image,(round(coord[0]),round(coord[1])),(255,255,255))

    # order points tl tr br bl
    ordered_points = geom.order_points(inters)
    rect_points = geom.rectify_points(ordered_points, ordered_lines)

    #for point in rect_points :
    #    cv2.drawMarker(image,(round(point[0]),round(point[1])),(0,0,255))
    
    transform, _ = cv2.findHomography(ordered_points, rect_points)
    warped_image = cv2.warpPerspective(image, transform, (image.shape[1], image.shape[0]))

    image_util.show(image)

    rounded = np.round(rect_points).astype(int)
    rounded[rounded < 0] = 0
    cut = warped_image[rounded[0, 1]:rounded[2, 1], rounded[0, 0]:rounded[2, 0]]
    image_util.show(cut)

    return cut


def find_countours(image, boxes) :
    rectified = []
    for box in boxes :
        roi = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        rectified.append(contour(roi))