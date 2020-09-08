from tasks import painting_detection
from netloader import *
from dbloader import *
from tools import box_util
from tools import image_util
from tasks import contours
from tasks import pattern_matching
from tasks import people_detection
from tasks import people_localization
from tqdm import tqdm


def start(image, index) :
    # painting detection and segmentation
    print("Detecting ROI and doing segmentation for paintings...")
    painting_boxes, painting_masks = painting_detection.detect(image)

    view = image.copy()
    best_match_history = []


    print("Starting rectification and retrieval for each ROI...")
    for i in tqdm(range(len(painting_boxes))):
        box = painting_boxes[i]
        mask = painting_masks[i]

        # painting rectification
        cut = contours.find_countours(image, box, mask)
        
        if cut is None :
            box_util.box_drawer(view, box, (0, 255, 0), "rectification failed")
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

    # people detection
    print("Detecting people...")
    people_boxes = people_detection.detect(image, yolo_net, yolo_classes)
    people_boxes = box_util.remove_fake_people(painting_boxes, people_boxes)

    for box in people_boxes:
        box_util.box_drawer(view, box, (255, 0, 0), "Person")

    # show frame to and save it
    image_util.save_img_cut(view, 'view', index)

    # people localization
    people_localization.localize_people(best_match_history, people_boxes, data)
    
