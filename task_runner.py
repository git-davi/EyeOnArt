from tasks import detection
from netloader import *
from tools import box_util
from tools import image_util
from tasks import contours
from tasks import pattern_matching


def start(image) :
    # painting detection
    print("Detecting ROI")
    painting_boxes = detection.detect(image, painting_net, painting_classes)
    #box_util.box_drawer(image, painting_boxes, (0, 255, 0), "painting")
    #image_util.show(image)

    # painting rectification
    print("Rectifing frame and cutting the paintings")
    cuts = contours.find_countours(image, painting_boxes)

    # painting retrieval
    print("Retrival for painting from DB")
    pattern_matching.match(cuts)

    # people detection
    print("Detecting ROI for people")
    people_boxes = detection.detect(image, people_net, people_classes)
    people_boxes = box_util.remove_fake_people(painting_boxes, people_boxes)
    print("Done")
    #box_util.box_drawer(image, people_boxes, (255, 0, 0), "person")
    #image_util.show(image)

    # people localization