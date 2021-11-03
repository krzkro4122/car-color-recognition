import cv2
import joblib
import numpy as np

from preprocess_train_data import ColorDescriptor
from copy import copy
from fastseg import MobileV3Large
from fastseg.image import colorize


# --- DETECTION CONFIGURATION --- #
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
                if classNames[classId] == 'car':
                    w, h = int(det[2] * wT) , int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2) , int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0] # Only for video
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        return x, y, w, h

        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0 , 255), 2)
        # cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
        #           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

def detect_from_image(img):

    print("Unpickling model...", end="")
    config_file = r"config/color_recognition_mlp.pkl"
    mlp = joblib.load(config_file)
    print("Done")


    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    try:
        x, y, w, h = findObjects(outputs, img)
    except:
        print(f"Couldn't find a car in given frame")
        exit()

    img_cropped = img[y:y + h, x:x + w]

    width = height = 228
    img_cropped = cv2.resize(img_cropped, (width, height))

    cd = ColorDescriptor((16, 16, 16))
    image = cd.describe(img_cropped)

    prediction = mlp.predict([image])

    img_cropped = cv2.putText(img_cropped, prediction[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Starting image', img)
    cv2.imshow('Image cropped', img_cropped)
    # cv2.imshow('Mask', colorized)
    cv2.waitKey(0)

def detect_from_video(cap):

    print("Unpickling model...", end="")
    config_file = r"config/color_recognition_mlp.pkl"
    mlp = joblib.load(config_file)
    print("Done")

    while cap.isOpened():
        success, img = cap.read()

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        img = findObjects(outputs, img)

        width = height = 228
        img = cv2.resize(img, (width, height)) #[:,:,::-1]

        cd = ColorDescriptor((16, 16, 16))
        image = cd.describe(img)

        prediction = mlp.predict([image])

        img = cv2.putText(img, prediction[0], (50, 50), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- Camera or demo video or demo picture --- #
    img = cv2.imread(r"assets/real_tests/sanfran_white.jpg")
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"assets/cars.mp4")

    detect_from_image(img)
    # detect_from_video(cap)
