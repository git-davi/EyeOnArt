import os
import cv2

paint_db = {}

print("Loading paintings db...")

for f in os.listdir("material/paintings_db/"):
            paint_db[f] = cv2.imread(f'material/paintings_db/{f}', cv2.IMREAD_GRAYSCALE)

print("Loaded")