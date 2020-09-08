import os
import glob
import cv2
import numpy as np
from tools import image_util

from dbloader import *


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def match_cut(cut):
    # increase brightness
    cut = increase_brightness(cut, 10)

    gray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)

    h, w = gray.shape[:2]

    MIN_MATCH_COUNT = 3
    MAX_MATCH = 0

    matches_list = []

    for f, des1 in des_db.items():
            try:
                matches = flann.knnMatch(des1,des2,k=2)
            except Exception:
                continue

            # store all the good matches as per Lowe's ratio test.
            good = []
            for pair in matches:
                if len(pair) == 2:
                    if pair[0].distance < 0.6*pair[1].distance:
                        good.append(pair[0])

            matches_list.append({
                'file': f,
                'score': len(good)
            })

    if not matches_list:
        return [{
            'file': None,
            'score': -1
        }]

    matches_list = list(sorted(matches_list, key=lambda k: k['score'], reverse=True))
    
    if matches_list[0]["score"] < 5 :
        return [{
            'file': None,
            'score': -1
        }]

    #print(f'Best match is FILE : {matches_list[0]["file"]}')
    #print(f'SCORE : {matches_list[0]["score"]}')
    #print(f'2nd SCORE : {matches_list[1]["score"]}')

    # debug
    #image_util.show(cut, 'cut')
    #image_util.show(cv2.imread(f'material/paintings_db/{matches_list[0]["file"]}'), 'db')

    return matches_list

'''
def match(cut) :
    best = match_cut(cut)

    #best = list(sorted(best, key=lambda k: k['score'], reverse=True))
    if best['file'] is None :
        return None
    return best
'''