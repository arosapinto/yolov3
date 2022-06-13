import cv2
import matplotlib.pyplot as plt
import numpy as np


#https://www.youtube.com/watch?v=zm9h4mYymk0

## loading yolo models
net = cv2.dnn.readNetFromDarknet('C:\\Test folder\\images\yolo\\Yolo_object_detection\\yolov2.cfg','C:\\Test folder\\images\\yolo\\Yolo_object_detection\\yolov2.weights')

classes = []
with open('C:\\Test folder\\images\yolo\\Yolo_object_detection\\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

my_img = cv2.imread('C:\\Test folder\\images\yolo\\Yolo_object_detection\\63787a69acd8ae8bd5069f2011d04289.jpg')
# my_img = cv2.imread('C:\\Test folder\\images\yolo\\Yolo_object_detection\\caoegato2.jfif')

plt.imshow(my_img)

wt, ht, _ = my_img.shape

# image need to be converted to darket/yolo format
blob = cv2.dnn.blobFromImage(my_img, 1/255, (416,416), (0,0,0), swapRB= True, crop=False)

net.setInput(blob)
last_layer = net.getUnconnectedOutLayersNames() # gives the last layer of the network
layer_out = net.forward(last_layer) # give sthe output of the last layer

boxes = []
confidences = []
class_ids = []

for output in layer_out:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > .6:
            center_x = int(detection[0] * wt)
            center_y = int(detection[1] * ht)
            w = int(detection[2] * wt)
            h = int(detection[3] * ht)
            
            x = int(center_x - w/2)
            y = int(center_x - h/2)
            
            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size=(len(boxes), 3))


for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i], 2))
    color = colors[i]
    cv2.rectangle(my_img, (x,y), (x+w,y+h), color, 2)
    cv2.putText(my_img, label + " "+confidence, (x,y+20), font, 2, (0,0,0), 2)

cv2.imshow('img', my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()