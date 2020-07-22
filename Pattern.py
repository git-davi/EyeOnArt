from tasks.pattern_matching import im_show
from tools.image_util import resize
import cv2

if __name__ == '__main__':
    im_show()
    """
    img = cv2.imread('rectified_imgs/rectified_45.jpg')
    resized = resize(img, 1200)
    cv2.imwrite('rectified_imgs/rectified_45_res.jpg', resized)
    """
