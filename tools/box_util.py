import cv2


def box_drawer(img, boxes, color, label) :
    for box in boxes :
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
            conditions = []
            conditions.append(person[0] > painting[0])
            conditions.append(person[1] > painting[1])
            conditions.append(person[2] < painting[2])
            conditions.append(person[3] < painting[3])

            if all(conditions) :
                real_people_boxes.remove(person)
                break

    return real_people_boxes