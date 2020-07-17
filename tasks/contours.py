import cv2
from tools import image_util

def find_countours(image) :
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    imgray_blurred = cv2.GaussianBlur(imgray, (5, 5), 1)
    ret, thresh = cv2.threshold(imgray_blurred, 127, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # debug purpose
    image = cv2.drawContours(image, contours, -1, (0,255,0))
    image_util.show(image)