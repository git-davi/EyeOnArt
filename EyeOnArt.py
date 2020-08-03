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
Davide Casalini, Robert Covic & Stefano Rossi
'''
)
parser.add_argument("filename", type=str, help="The filename to the source video you want to elaborate")
parser.add_argument('--skip', type=int, dest='skip', default=0, metavar="N_FRAME",help="The number of frames to skip")
args = parser.parse_args()


video = cv2.VideoCapture(args.filename)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

success, image = video.read()
count = 0
while(success) :
    
    if count % (args.skip + 1) == 0 :
        print("------ Tasks started on frame {} out of {} ------".format(count+1, length))
        task_runner.start(image, count)
    
    success, image = video.read()
    count += 1


print("Finished!")
print("See you soon :)")