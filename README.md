# EyeOnArt
Computer Vision project for painting detection from videos.  
The videos are captured in the art museum Galleria Estense, Modena.

## Setup
Download the weights and cfg file from [here](https://drive.google.com/file/d/1cQfSZnJememvvT7jVZx_KU5w86RQxgAV/view?usp=sharing).  
Unzip the directory in the root dir of the project.  
  
Setup the python environment :
```shell
$ pip install -r requirements.txt
```

## Output
The outputs will be placed in the `output` dir.  
There will be a view for each frame containing the bounding boxes for every detected painting.
On top of every ROI you will find the title for the best match.
Every function (that represents a step) of the pipeline is returning the requested output.  
  

## Usage Example
```shell
$ python EyeOnArt.py material/test.mp4
```
  
## Help
```shell
usage: EyeOnArt.py [-h] [--skip N_FRAME] filename

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

positional arguments:
  filename        The filename to the source video you want to elaborate

optional arguments:
  -h, --help      show this help message and exit
  --skip N_FRAME  The number of frames to skip
```