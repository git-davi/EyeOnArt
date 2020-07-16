import cv2


def show(image) :
    # display output image    
    cv2.imshow("object detection", image)
    # wait until any key is pressed
    cv2.waitKey()
    # release resources
    cv2.destroyAllWindows()