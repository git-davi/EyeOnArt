from tasks.pattern_matching import im_show
from tools.image_util import resize
import cv2
from tasks.people_localization import localize_people

if __name__ == '__main__':
    results = im_show()
    
    # variables for testing purposes
    boxes = []
    matches = []
    matches.append(results)
    boxes.append(1)
    
    # in the final version a list of painting matches and a list of person bbox
    # have to be fed to the localize_people method... The first list should
    # contain the matches with only the highest confidence of the ENTIRE FRAME!

    localize_people(results, boxes)
    """
    img = cv2.imread('rectified_imgs/rectified_45.jpg')
    resized = resize(img, 1200)
    cv2.imwrite('rectified_imgs/rectified_45_res.jpg', resized)
    """
