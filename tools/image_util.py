import cv2
import numpy as np


def show(image) :
    # display output image    
    cv2.imshow("object detection", image)
    # wait until any key is pressed
    cv2.waitKey()
    # release resources
    cv2.destroyAllWindows()



def draw_lines(horizontal_lines, vertical_lines, image) :

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