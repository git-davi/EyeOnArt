import cv2
import argparse
import task_runner

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
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

success, image = video.read()
count = 1
while(success) :
    print("------ Tasks started on frame {} out of {} ------".format(count, length))
    task_runner.start(image)
    success, image = video.read()
    count += 1


print("Finished!")
print("See you soon :)")