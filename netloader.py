import cv2

print("Loading neural nets and classes...")

painting_net = cv2.dnn.readNet('cfg/weights/painting.weights', 'cfg/net/painting.cfg')
painting_classes = ['painting']

people_net = cv2.dnn.readNet('cfg/weights/yolov3.weights', 'cfg/net/yolov3.cfg')
people_classes = []
with open('cfg/coco/coco.names') as f :
    people_classes = [line.strip() for line in f.readlines()]

print("Loaded")