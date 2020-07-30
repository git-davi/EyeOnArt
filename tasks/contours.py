import cv2
import numpy as np

from tools import image_util
from tools import geom
from tools import box_util


def contour(image) :
    edges = cv2.Canny(image, 50, 20)
    image_util.show(edges)
    edges_dilated = cv2.dilate(edges, None, iterations=1)
    contours, hierarchy = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cont = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cont)

    hull_mask = np.zeros((image.shape[0],image.shape[1], 1), np.uint8)
    hull_mask = cv2.drawContours(hull_mask, [hull], -1, (255, 255, 255))
    
    lines = cv2.HoughLines(hull_mask, 1, np.pi / 180, 100)

    # try matching with all roi
    if len(lines) < 4 :
        return None

    # scale lines for kmeans
    lines[:, :, 0], min_rho, max_rho = geom.feature_scaling(lines[:, :, 0])
    lines[:, :, 1], min_theta, max_theta = geom.feature_scaling(lines[:, :, 1])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    try :
        _, _, cluster_lines = cv2.kmeans(lines, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    except Exception :
        return None
    
    # lines descaling
    cluster_lines[:, 0] = geom.feature_descaling(cluster_lines[:, 0], min_rho, max_rho )
    cluster_lines[:, 1] = geom.feature_descaling(cluster_lines[:, 1], min_theta, max_theta )
    
    # return lines in order t r b l
    ordered_lines = geom.order_lines(cluster_lines)
    horizontal_lines = np.array([ordered_lines[0], ordered_lines[2]])
    vertical_lines = np.array([ordered_lines[1], ordered_lines[3]])

    # debugging
    image_util.draw_lines(horizontal_lines, vertical_lines, image)
    image_util.show(image)

    try :
        inters = geom.segmented_intersections(horizontal_lines,vertical_lines)
    except Exception :
        return None

    if len(inters) != 4 :
        return None

    #for coord in inters:
    #    cv2.drawMarker(image,(round(coord[0]),round(coord[1])),(255,255,255))

    # order points tl tr br bl
    ordered_points = geom.order_points(inters)
    rect_points = geom.rectify_points(ordered_points, ordered_lines)

    #for point in rect_points :
    #    cv2.drawMarker(image,(round(point[0]),round(point[1])),(0,0,255))
    
    transform, _ = cv2.findHomography(ordered_points, rect_points)
    if transform is None :
        return None
    warped_image = cv2.warpPerspective(image, transform, (image.shape[1], image.shape[0]))

    #image_util.show(warped_image)

    rounded = np.round(rect_points).astype(int)
    rounded[rounded < 0] = 0

    cut = warped_image[rounded[0, 1]:rounded[2, 1], rounded[0, 0]:rounded[2, 0]]
    
    #image_util.show(cut)
    cut = image_util.remove_border(cut, 0.2)
    #save_img_cut(cut)

    if box_util.is_bad_cut(cut) :
        return None
    return cut


def save_img_cut(img):
    dir_path = 'rectified_imgs/'
    file_name = '{}rectified_{}.jpg'.format(dir_path, np.random.randint(100, size=1)[0])
    print('Img saved at: {}'.format(file_name))
    image_util.show(img)
    try:
        cv2.imwrite(file_name, img)
    except Exception as e:
        print(e)


def find_countours(image, boxes) :
    rectified = []
    for box in boxes :
        roi = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        cut = contour(roi)
        if cut is not None :
            rectified.append(cut)
    return rectified
