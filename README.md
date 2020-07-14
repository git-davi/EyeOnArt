# EyeOnArt
Computer Vision project for painting detection from videos.  
The videos are captured in the art museum Galleria Estense, Modena.

## Requirements
- Python 2.7
- `requirements.txt` file for python packages

## Usage
```shell
$ python EyeOnArt.py material/test.mp4
```
  
You can create a virtualenv called **venv** in the root dir of the project.  
It won't be pushed.


## Help
```shell
usage: EyeOnArt.py [-h] filename

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

positional arguments:
  filename    The filename to the source video you want to elaborate

optional arguments:
  -h, --help  show this help message and exit
```