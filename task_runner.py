from tasks import detection
from netloader import *
from tools import box_util
from tools import image_util
from tasks import contours
from tasks import pattern_matching
from tasks import people_detection


def start(image) :
    # painting detection
    print("Detecting ROI for paintings")
    painting_boxes = detection.detect(image, painting_net, painting_classes)
    view = image.copy()
    box_util.box_drawer(view, painting_boxes, (0, 255, 0), "painting")
    image_util.show(view)

    # painting rectification
    print("Rectifing frame and cutting the paintings")
    cuts = contours.find_countours(image, painting_boxes)


    # painting retrieval
    print("Retrival for painting from DB")
    best_match = []
    best_match.append(pattern_matching.match(cuts))


    # people detection
    print("Detecting ROI for people")
    people_boxes = detection.detect(image, people_net, people_classes)
    people_boxes = box_util.remove_fake_people(painting_boxes, people_boxes)
    print("Done")
    #box_util.box_drawer(image, people_boxes, (255, 0, 0), "person")
    #image_util.show(image)

    # people localization
    # test people

    #people_boxes.append(1)
    people_detection.localize_people(best_match,people_boxes)