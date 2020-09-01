from tasks import detection
from netloader import *
from dbloader import *
from tools import box_util
from tools import image_util
from tasks import contours
from tasks import pattern_matching
from tasks import people_detection

import pandas as pd 
from tqdm import tqdm


def start(image, index) :
    # painting detection
    print("Detecting ROI for paintings")
    painting_boxes = detection.detect(image, painting_net, painting_classes)
    print("Done")
    
    view = image.copy()
    

    data = pd.read_csv('material/data.csv', index_col="Image")
    best_match_history = []

    print("Starting rectification and retrieval for each ROI...")
    for box in tqdm(painting_boxes):
        # painting rectification
        cut = contours.find_countours(image, box)

        if cut is None :
            box_util.box_drawer(view, box, (0, 255, 0), "rectification failed")
            continue

        # painting retrieval
        best_matches = pattern_matching.match_cut(cut, paint_db)
        
        if best_matches[0]['file'] is None :
            box_util.box_drawer(view, box, (0, 255, 0), "retrieval failed")
            continue

        best_match = best_matches[0]
        
        best_match_history.append(best_match)

        # get name and author from data csv file
        painting_info = data.loc[best_match['file']]
        title = painting_info['Title']

        box_util.box_drawer(view, box, (0, 255, 0), title)

    print("Done")
    # show frame to and save it
    image_util.save_img_cut(view, 'view', index)

    # people detection
    print("Detecting ROI for people")
    people_boxes = detection.detect(image, people_net, people_classes)
    people_boxes = box_util.remove_fake_people(painting_boxes, people_boxes)
    print("Done")

    # people localization
    people_detection.localize_people(best_match_history, people_boxes, data)

