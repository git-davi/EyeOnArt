import cv2
import numpy as np

from tools import image_util
from tools import geom
from tools import box_util


def contour(image) :
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    imgray_blurred = cv2.GaussianBlur(imgray, (5, 5), 1)
    edges = cv2.Canny(imgray_blurred, 50, 20)
    edges_dilated = cv2.dilate(edges, None, iterations=5)
    contours, hierarchy = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cont = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cont)

    hull_mask = np.zeros((image.shape[0]*4,image.shape[1]*4, 1), np.uint8)
    # traslate the mask
    hull[:, :, 0] = hull[:, :, 0] + round(image.shape[1]/2)
    hull[:, :, 1] = hull[:, :, 1] + round(image.shape[0]/2)
    hull_mask = cv2.drawContours(hull_mask, [hull], -1, (255, 255, 255))

    lines = cv2.HoughLines(hull_mask, 1, np.pi / 180, 100)

    # try matching with all roi if less than 4 lines found
    if len(lines) < 4 :
        return image

    # return lines in order t r b l
    border_lines = geom.extract_borders(lines)
    if border_lines is None :
        return image

    horizontal_lines = np.array([border_lines[0], border_lines[2]])
    vertical_lines = np.array([border_lines[1], border_lines[3]])

    inters = geom.segmented_intersections(horizontal_lines,vertical_lines)

    # traslate back points
    inters[:, 0] = inters[:, 0] - round(image.shape[1]/2)
    inters[:, 1] = inters[:, 1] - round(image.shape[0]/2)

    #for coord in inters:
    #    cv2.drawMarker(image,(round(coord[0]),round(coord[1])),(255,255,255))
    #image_util.show(image)
    
    #check for duplicates
    if np.unique(inters).shape != inters.shape :
        return image

    # order points tl tr br bl
    ordered_points = geom.order_points(inters)
    rect_points = geom.rectify_points(ordered_points, border_lines)

    #for point in rect_points :
    #    cv2.drawMarker(image,(round(point[0]),round(point[1])),(0,0,255))

    print(rect_points)

    transform, _ = cv2.findHomography(ordered_points, rect_points)
    warped_image = cv2.warpPerspective(image, transform, (image.shape[1], image.shape[0]))

    rounded = np.round(rect_points).astype(int)
    rounded[rounded < 0] = 0

    cut = warped_image[rounded[0, 1]:rounded[2, 1], rounded[0, 0]:rounded[2, 0]]

    #cut = image_util.remove_border(cut, 0.2)

    #save_img_cut(cut)
    #image_util.show(cut)
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