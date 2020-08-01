from tasks.pattern_matching import pattern_matching
from tools.image_util import resize
import cv2
from tasks.homography_test import find_homography

if __name__ == '__main__':
    #pattern_matching()
    find_homography()

    """
    img = cv2.imread('rectified_imgs/rectified_47.jpg')
    resized = resize(img, 900)
    cv2.imwrite('rectified_imgs/rectified_47_big.jpg', resized)
    """

