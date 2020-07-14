import cv2
import argparse
import detection

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=
'''
-------------
|  EyeOnArt |
-------------
Painting Detection project.

               __________________________
             /|        Art Gallery       |
            / |  ____     ____     ____  |
           /  | |o   |   |  , |   | _  | |
          /   | |  O |   |.   |   |(@) | |
         /    | |_,k,|   |_,-,|   |\|p | |
        / /|  | |  h |   | ,; |   | |  | |
       / / |  | |_z__|   |____|   |____| |
      / /@;|  |  z z                     |
     /  |Y | z|_{)_______________________|
    /   | /  /z /H
   / /| |/  /z   Y
  / / |    / {)  d
 / / %|   /  /|
|  |&"|  /    Y
|  | /  /     d
|  |/  /
|     /
|    /
|   /
|  /
| /
|/

Credits :
Davide Casalini, Robert Covic & Stefano Rossi.
'''
)

parser.add_argument("filename", type=str, help="The filename to the source video you want to elaborate")

args = parser.parse_args()


video = cv2.VideoCapture(args.filename)

success, image = video.read()
while(success) :
    detection.start(image)
    success, image = video.read()


print "Finished!"
print "See you soon :)"