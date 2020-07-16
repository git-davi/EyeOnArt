from tasks import detection
from netloader import *
from tools import box_util
from tools import image_util


def start(image) :
    # painting detection
    print("Detecting ROI for paintings")
    painting_boxes = detection.detect(image, painting_net, painting_classes)
    print("Done")
    box_util.box_drawer(image, painting_boxes, (0, 255, 0), "painting")
    image_util.show(image)

    # painting rectification


    # painting retrieval


    # people detection
    print("Detecting ROI for people")
    people_boxes = detection.detect(image, people_net, people_classes)
    people_boxes = box_util.remove_fake_people(painting_boxes, people_boxes)
    print("Done")
    box_util.box_drawer(image, people_boxes, (255, 0, 0), "person")
    image_util.show(image)

    # people localization