import os
import cv2
import pandas as pd


print('Loading paintings infos...')
data = pd.read_csv('material/data.csv', index_col="Image")

print("Extracting db descriptors...")

des_db = {}

patchSize = 24
orb = cv2.ORB_create(edgeThreshold = patchSize,
                    patchSize = patchSize)

#orb = cv2.ORB_create()

for f in os.listdir("material/paintings_db/"):
    template = cv2.imread(f'material/paintings_db/{f}', cv2.IMREAD_GRAYSCALE)

    kp1, des1 = orb.detectAndCompute(template, None)
    des_db[f] = des1


print("Initiating FLANN matcher...")

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
           table_number = 6,
           key_size = 12,
           multi_probe_level = 1)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

print("Finished")