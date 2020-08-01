# EyeOnArt
Computer Vision project for painting detection from videos.  
The videos are captured in the art museum Galleria Estense, Modena.

## Setup
Download the weights and cfg file from [here](https://drive.google.com/file/d/11VwWheCa8JXgEKCqbhdf5c9FO9z3-ihY/view?usp=sharing).  
Unzip the directory in the root dir of the project.  
  
Setup the python environment :
```shell
$ pip install -r requirements.txt
```


## Usage Example
```shell
$ python EyeOnArt.py material/test.mp4
```
  
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