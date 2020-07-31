import cv2
import numpy as np

from tools import image_util
from tools import geom
from tools import box_util


def contour(image) :
    edges = cv2.Canny(image, 50, 20)
    edges_dilated = cv2.dilate(edges, None, iterations=1)
    contours, hierarchy = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cont = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cont)

    hull_mask = np.zeros((image.shape[0],image.shape[1], 1), np.uint8)
    hull_mask = cv2.drawContours(hull_mask, [hull], -1, (255, 255, 255))
    
    param = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.1*param, True)

    if len(approx) < 4 :
        return None

    # vertices in order tl bl br tr
    vertices = geom.get_vertices(approx)
    print("original image vertices" + '\n')
    print(vertices)
    flag=is_it_a_fucking_rombo(vertices)
    print("è un rombo?" + '\n')
    print(flag)

    for pt in vertices:
        ziocan = (pt[0],pt[1])
        #print(ziocan)
        hull_mask = cv2.circle(hull_mask, ziocan, 30, (255, 0, 0), 10)
    
    #hull_mask = cv2.drawContours(hull_mask, [approx], -1, (255, 255, 255))

    image_util.show(hull_mask)

    
    # order points tl tr br bl
    if(flag == False):
        rect_points = geom.rectify_points(vertices)
        print(rect_points)
    elif (flag):
        sorted_vertices= sort_rhombus(vertices)
        print("sorted vertices" + '\n')
        print(sorted_vertices)
        rect_points=geom.rectify_rhombus(sorted_vertices)
        print("These are the rectified points " + '\n')
        print(rect_points)

    for point in rect_points :
        cv2.drawMarker(image,(round(point[0]),round(point[1])),(0,0,255))
    
    image_util.show(image)

    transform, _ = cv2.findHomography(vertices, rect_points)
    if transform is None :
        return None
    warped_image = cv2.warpPerspective(image, transform, (image.shape[1], image.shape[0]))

    image_util.show(warped_image)

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

def is_it_a_fucking_rombo(vertices):
    tl,tr,br,bl = vertices
    #altri controlli equivalenti br[1]-bl[1]
    if(abs(tl[1]-tr[1]) > 70 ):
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