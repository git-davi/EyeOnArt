import cv2


print("Loading neural nets and classes...")

# YOLO
painting_net = cv2.dnn.readNet('cfg/weights/painting_w.weights', 'cfg/net/painting_c.cfg')
painting_classes = ['painting']

people_net = cv2.dnn.readNet('cfg/weights/people_w.weights', 'cfg/net/people_c.cfg')
people_classes = []
with open('cfg/coco/people.names') as f:
    people_classes = [line.strip() for line in f.readlines()]


# MASK_RCNN
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from mrcnn.config import Config
from mrcnn import model as modellib, utils


class paintingConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "painting"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + painting
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
class InferenceConfig(paintingConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                        config=config,
                        model_dir="cfg/mask_rcnn")

model.load_weights("cfg/mask_rcnn/mask_rcnn_painting.h5", by_name=True)

print("Loaded")