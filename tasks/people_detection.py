import os
import cv2
import numpy as np
from tools import image_util
import csv

# people localization is based on painting matches:
# according to the reference namefile we can get 
# the location that is defined in data.csv file

def localize_people(matches, p_boxes):
    if len(p_boxes) > 0 and len(matches) > 0:
        try:
            data_f = open('material/data.csv')
        except Exception as e:
            print("DATA FILE EXCEPTION")
            print(e)
        
        csv_reader = csv.DictReader(data_f)
        
        # voting system to decide the room
        votes = {}
        
        for m in matches:
            for row in csv_reader:
                if row["Image"] == m["file"]:
                    if row["Room"] in votes:
                        votes[row["Room"]] += 1
                    else:
                        votes[row["Room"]] = 1
        
        result = max(votes, key=votes.get)
        print("Detected ", len(p_boxes), "person/people in room ", result)
        display_localization(result)
        
def display_localization(result):
    try:
        positions_f = open('material/rooms.csv')
        map_image = cv2.imread('material/map.png')
    except Exception as e:
        print(e)
    reader = csv.DictReader(positions_f)
    for row in reader:
        if row['Room'] == result:
            map_image = cv2.circle(map_image, (int(row['X']),int(row['Y'])), 3, (255,0,0), 5)
            break
    cv2.imshow("Map", map_image)
    cv2.waitKey(0)
