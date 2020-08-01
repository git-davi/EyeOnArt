from tasks import detection
from netloader import *
from tools import box_util
from tools import image_util
from tasks import contours
from tasks import pattern_matching
from tasks import people_detection

import pandas as pd 


def start(image, index) :
    # painting detection
    print("Detecting ROI for paintings")
    painting_boxes = detection.detect(image, painting_net, painting_classes)
    view = image.copy()
    #box_util.box_drawer(view, painting_boxes, (0, 255, 0), "painting")
    #image_util.show(view)

    data = pd.read_csv('material/data.csv', index_col="Image")
    best_match_history = []

    for box in painting_boxes:
        # painting rectification
        cut = contours.find_countours(image, box)

        if cut is None :
            continue

        # painting retrieval
        best_matches = pattern_matching.match_cut(cut)
        
        if best_matches[0]['file'] is None :
            box_util.box_drawer(view, box, (0, 255, 0), "retrieval failed")
            continue

        best_match = best_matches[0]
        
        best_match_history.append(best_match)

        # get name and author from data csv file
        painting_info = data.loc[best_match['file']]
        title = painting_info['Title']

        box_util.box_drawer(view, box, (0, 255, 0), title)

    # show frame to and save it
    save_img_cut(view, index)

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


def save_img_cut(img, index):
    dir_path = 'output/'
    file_name = '{}view_{}.jpg'.format(dir_path, index)
    image_util.show(img)
    try:
        cv2.imwrite(file_name, img)
        print('Img saved at: {}'.format(file_name))
    except Exception as e:
        print(e)