import os
import cv2
import numpy as np
from tools import image_util

def im_show():
    paintings_found = []
    painting_db = 'material/paintings_db/'
    norm_prob = []
    rectified_db = 'rectified_imgs/'

    # Leggo l'immagine rettificata e la converto in gray
    sample = 'rectified_imgs/rectified_45.jpg'
    image = cv2.imread(sample)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f'[DEBUG] proccessing image {sample}')

    # Ciclo il db dei paintings
    for paint in os.listdir(painting_db):
        found = None
        print(f'[DEBUG] confronto con {paint}')

        template_path = painting_db + paint

        try:
            template = cv2.imread(template_path)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(e)
            continue

        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]

        # Loop over the scales
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = image_util.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        if found is not None:
            (maxVal, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        norm_prob.append(maxVal)

        print(f'IMG: {paint} - SCORE: {maxVal}')
        paintings_found.append({
            'maxVal': maxVal,
            'startX': startX,
            'startY': startY,
            'endX': endX,
            'endY': endY,
            'sprite': paint
        })

    team = list(sorted(paintings_found, key=lambda k: k['maxVal'], reverse=True))[:3]
    norm = sorted([(float(i) - np.min(norm_prob)) / (np.max(norm_prob) - np.min(norm_prob)) for i in norm_prob], reverse=True)[:3]

    for i in range(3):
        team[i]["maxVal"] = norm[i]

    print('\n\n')
    for p in team:
        cv2.rectangle(image, (p['startX'], p['startY']), (p['endX'], p['endY']), (0, 0, 255), 2)
        cv2.putText(image, p['sprite'], (p['endX'] + 10, p['startY'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print(f'{p["sprite"]}\tSCORE: {p["maxVal"]}')
    print('\n\n')

    cv2.imshow("Image", image)
    cv2.waitKey(0)

    # TODO: not really sure what is best to be returned
    return team