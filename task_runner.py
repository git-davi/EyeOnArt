from tasks import detection
from netloader import *

def start(image) :
    # painting detection
    print("Detecting ROI for paintings")
    detection.detect(image, painting_net, painting_classes)
    print("Done")

    # painting rectification


    # painting retrieval


    # people detection
    print("Detecting ROI for people")
    detection.detect(image, people_net, people_classes)
    # check if person bounding box is inside of painting.
    # if it is then it's not a person
    print("Done")


    # people localization