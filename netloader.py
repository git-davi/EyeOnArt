import cv2

print("Loading neural nets and classes...")

painting_net = cv2.dnn.readNet('cfg/weights/painting_w.weights', 'cfg/net/painting_c.cfg')
painting_classes = ['painting']

people_net = cv2.dnn.readNet('cfg/weights/people_w.weights', 'cfg/net/people_c.cfg')
people_classes = []
with open('cfg/coco/people.names') as f:
    people_classes = [line.strip() for line in f.readlines()]

print("Loaded")