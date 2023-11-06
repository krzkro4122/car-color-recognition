from turtle import width
import cv2 as cv
import numpy as np

# Parametry detekcji
width_height = 320
confidence_threshold = 0.5
non_maximum_suppresion_threshold = 0.2

# Wydobycie nazw klas zbioru COCO z pliku
classes_file = r"config/coco.names"
class_names = []
with open(classes_file, "rt") as f:
    class_names = f.read().rstrip("\n").split("\n")

# Pliki konfiguracyjne modelu YOLOv3
modelConfiguration = r"config/yolov3-320.cfg"
modelWeights = r"config/yolov3.weights"

# Definicja glebokiej sieci neuronowej
deep_neural_network = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
deep_neural_network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
deep_neural_network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def dnn_passthrough(img):
    blob = cv.dnn.blobFromImage(img, 1 / 255, (width_height, width_height), [0, 0, 0], 1, crop=False)
    deep_neural_network.setInput(blob)
    layersNames = deep_neural_network.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in deep_neural_network.getUnconnectedOutLayers()]
    return deep_neural_network.forward(outputNames)

def findObjects(outputs, img):
    height, width, _ = img.shape
    bounding_boxes = []
    class_indexes = []
    confidences = []

    for output in outputs:

        for determinant in output:
            scores = determinant[5:]
            class_index = np.argmax(scores)
            confidence = scores[class_index]

            if confidence > confidence_threshold and class_names[class_index] == 'car':
                w, h = int(determinant[2] * width) , int(determinant[3] * height)
                x, y = int((determinant[0] * width) - w / 2) , int((determinant[1] * height) - h / 2)
                bounding_boxes.append([x, y, w, h])
                class_indexes.append(class_index)
                confidences.append(float(confidence))

    indices = cv.dnn.NMSBoxes(
        bounding_boxes,
        confidences,
        confidence_threshold,
        non_maximum_suppresion_threshold)

    output = []
    for i in indices:
        box = bounding_boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        output.append(box)

    return output

def put_rectangles(img, boxes):
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

if __name__ == "__main__":
    image_path = r'assets/0197.jpg'
    image = cv.imread(image_path)
    outputs = dnn_passthrough(image)

    try:
        boxes = findObjects(outputs, image)
    except:
        print("Couldn't find a car in given frame")


    # --- TEST --- #

    print(boxes)
    put_rectangles(image, boxes)
    cv.imshow('Showcase', image)
    cv.waitKey(0)