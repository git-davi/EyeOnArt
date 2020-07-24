import cv2
import glob
import os
import numpy as np
import time


def find_homography():
    image = cv2.imread("rectified_imgs/rectified_51.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #gray = image
    h, w = gray.shape[:2]

    MIN_MATCH_COUNT = 3
    MAX_MATCH = 0

    matches_list = []

    for f in os.listdir("material/paintings_db/"):
        for file in glob.glob(f'material/paintings_db/{f}'):
            print("Matching for file {}".format(file))

            template = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

            patchSize = 16

            orb = cv2.ORB_create(edgeThreshold = patchSize,
                                    patchSize = patchSize)

            kp1, des1 = orb.detectAndCompute(template, None)
            kp2, des2 = orb.detectAndCompute(gray, None)

            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6,
                       key_size = 12,
                       multi_probe_level = 1)
            search_params = dict(checks = 50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []

            for pair in matches:
                if len(pair) == 2:
                    if pair[0].distance < 0.75*pair[1].distance:
                        good.append(pair[0])

            matches_list.append({
                'file': f,
                'score': len(good)
            })

    matches_list = list(sorted(matches_list, key=lambda k: k['score'], reverse=True))
    [print(f'IMAGE: {el["file"]} - SCORE: {el["score"]}') for el in matches_list]
    '''
    print("Best match for FILE : {}".format(matches_list[0]["file"]))
    print("SCORE : {}".format(matches_list[0]["score"]))
    '''