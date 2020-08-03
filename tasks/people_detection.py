import os
import cv2
import numpy as np
from tools import image_util
import pandas as pd


# people localization is based on painting matches:
# according to the reference namefile we can get 
# the location that is defined in data.csv file

def localize_people(matches, p_boxes, data):

    if len(p_boxes) <= 0 :
        print("No people detected in this frame")
        return
    else :
        print("Detected some people...")

    if matches:
        best_match = max(matches, key=lambda x: x['score'])

        best_info = data.loc[best_match['file']]
        room = best_info['Room']

        print("Detected ", len(p_boxes), "person/people in room ", room)
        display_localization(room)
        

def display_localization(result):
    rooms = pd.read_csv('material/rooms.csv', index_col='Room')
    map_image = cv2.imread('material/map.png')
    
    room = rooms.loc[result]

    map_image = cv2.circle(map_image, (int(room['X']),int(room['Y'])), 20, (0,0,255), 5)
    
    image_util.show(map_image, 'map')
