import cv2
import numpy as np


def box_drawer(img, box, color, label) :
    
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    draw_bounding_box(img, label, color, x, y, x+w, y+h)


def adjust_box(img, x, y, w, h) :
    
    x = round(x)
    y = round(y)
    w = round(w)
    h = round(h)

    # adjust borders exceding
    x = 5 if x <= 0 else x
    y = 5 if y <= 0 else y
    w = img.shape[1] - x - 5 if x + w >= img.shape[1] else w
    h = img.shape[0] - y - 5 if y + h >= img.shape[0] else h

    return [x, y, w, h]
    

def draw_bounding_box(img, label, color, x, y, x_plus_w, y_plus_h):

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x+10,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    

def remove_fake_people(painting_boxes, people_boxes) :
    real_people_boxes = people_boxes.copy()
    for person in people_boxes :
        for painting in painting_boxes :
            # expand painting box
            w_scale = painting[2] * 0.1
            h_scale = painting[3] * 0.1

            painting[0] = painting[0] - w_scale
            painting[1] = painting[1] - h_scale
            painting[2] = painting[2] + (w_scale*2)
            painting[3] = painting[3] + (h_scale*2)

            conditions = []
            conditions.append(person[0] > painting[0])
            conditions.append(person[1] > painting[1])
            conditions.append(person[3] + person[1] < painting[3] + painting[1])
            conditions.append(person[2] + person[0] < painting[2] + painting[0])

            if all(conditions) :
                real_people_boxes.remove(person)
                break

    return real_people_boxes

def is_bad_cut(cut) :
    s_min, s_max = np.sort(cut.shape[:2])
    if s_min == 0 or s_max/s_min > 5 :
        return True
    return False
