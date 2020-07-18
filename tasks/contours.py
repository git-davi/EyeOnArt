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

    
    lines = cv2.HoughLines(hull_mask, 1, np.pi / 180, 100)

    # scale lines for kmeans
    lines[:, :, 0], min_rho, max_rho = geom.feature_scaling(lines[:, :, 0])
    lines[:, :, 1], min_theta, max_theta = geom.feature_scaling(lines[:, :, 1])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, _, cluster_lines = cv2.kmeans(lines, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # lines descaling
    cluster_lines[:, 0] = geom.feature_descaling(cluster_lines[:, 0], min_rho, max_rho )
    cluster_lines[:, 1] = geom.feature_descaling(cluster_lines[:, 1], min_theta, max_theta )
    

    for rho,theta in cluster_lines :
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv2.line(image,(x1,y1),(x2,y2),(255, 255, 255), 1)

    image_util.show(image)
        

def find_countours(image, boxes) :
    for box in boxes :
        roi = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        contour(roi)