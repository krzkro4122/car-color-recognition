import cv2
import numpy as np
# import semantic_segmentation

from copy import copy

# --- Camera or demo video or demo picture --- #
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"images/cars.mp4")
img = cv2.imread(r"assets/0197.jpg")

whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

# --- LOAD MODEL --- #
# Coco Names
classesFile = r"config/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Model Files
# modelConfiguration = r"../config/yolov3-tiny.cfg"
# modelWeights = r"../config/yolov3-tiny.weights"
modelConfiguration = r"config/yolov3-320.cfg"
modelWeights = r"config/yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT) , int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2) , int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        # return img[y:y + h, x:x + w] # Detect just 1 object

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0 , 255), 2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


# --- IMAGE --- #
blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
net.setInput(blob)

layersNames = net.getLayerNames()
outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
outputs = net.forward(outputNames)

img_before = copy(img)
img_after = findObjects(outputs, img)
_, mask = semantic_segmentation.segment(img_after)

mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
thresh, mask_black_and_white = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
masked_image = cv2.bitwise_or(img_after, img_after, mask=mask_black_and_white)

cv2.imshow('Image before', img_before)
cv2.imshow('Image after', img_after)
cv2.imshow('Mask', mask)
cv2.imshow('Masked image', masked_image)
cv2.waitKey(0)

# # --- VIDEO --- #
# while cap.isOpened():
#     success, img = cap.read()

#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
#     net.setInput(blob)

#     layersNames = net.getLayerNames()
#     outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
#     outputs = net.forward(outputNames)
#     findObjects(outputs, img)

#     cv2.imshow('Image', img)
#     cv2.waitKey(1)

# cap.release()
# cv2.destroyAllWindows()