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
            print("FILE EXCEPTION")
            print(e)
        
        csv_reader = csv.DictReader(data_f)
        votes = {}
        # more than 1 painting can be recognized in a scene?
        for m in matches:
            for row in csv_reader:
                if row["Image"] == m['sprite']:
                    if row["Room"] in votes:
                        votes[row["Room"]] += 1
                    else:
                        votes[row["Room"]] = 1
        
        result = max(votes, key=votes.get)
        print("Detected ", len(p_boxes), "person/people in room ", result)