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
    image_util.show(hull_mask)
    hull_mask_dilated = cv2.dilate(hull_mask, None, iterations=1)
    
    lines = cv2.HoughLines(hull_mask, 1, np.pi / 180, 100)

    # scale lines for kmeans
    lines[:, :, 0], min_rho, max_rho = geom.feature_scaling(lines[:, :, 0])
    lines[:, :, 1], min_theta, max_theta = geom.feature_scaling(lines[:, :, 1])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, _, cluster_lines = cv2.kmeans(lines, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # lines descaling
    cluster_lines[:, 0] = geom.feature_descaling(cluster_lines[:, 0], min_rho, max_rho )
    cluster_lines[:, 1] = geom.feature_descaling(cluster_lines[:, 1], min_theta, max_theta )
    
    horizontal_lines,vertical_lines = segment_by_angle_kmeans(cluster_lines)

    for rho,theta in horizontal_lines :
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv2.line(image,(x1,y1),(x2,y2),(255, 0, 0), 1)

    for rho,theta in vertical_lines :
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv2.line(image,(x1,y1),(x2,y2),(0, 255, 0),1)

    print(cluster_lines)
    inters=segmented_intersections(horizontal_lines,vertical_lines)

    for coord in inters:
        cv2.drawMarker(image,(coord[0][0],coord[0][1]),(255,255,255))
    #print(inters)
    top_right,top_left,bottom_right,bottom_left=inters[0],inters[1],inters[2],inters[3]
    print("top_right = {}".format(top_right))
    print("top_left = {}".format(top_left))
    print("bottom_right = {}".format(bottom_right))
    print("bottom_left = {}".format(bottom_left))
    image_util.show(image)
        
def rectified_points(tr,tl,br,bl,dist_lines):
    #da moltiplicare anche l'angolo, contenuto in dist_lines
    top=cv.norm(tr-tl)
    bottom=cv.norm(br-bl)
    left=cv.norm(bl-tl)
    right=cv.norm(br-tr)

#x points
    xl,xr= tl[0],tr[0] if (top > bottom) else bl[0],br[0]
#y points
    yl,yr= tl[0],bl[0] if (left > right) else tr[0],br[0]

    return xl,xr,yl,yr
 

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[1] for line in lines])
    #print(" angles are " , angles)
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    hor=[]
    ver=[]

    for i in range(len(lines)):
        if labels[i]==0:
            hor.append(lines[i])
        if labels[i]==1:
            ver.append(lines[i])
    return hor,ver


def segmented_intersections(horz,verz):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i in range(len(horz)):
        for j in range(len(verz)):
            intersections.append(intersection(horz[i],verz[j]))
    return intersections

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form."""
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def find_countours(image, boxes) :
    for box in boxes :
        roi = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        contour(roi)