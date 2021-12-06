import cv2
import joblib
import numpy as np

from preprocess_train_data import ColorDescriptor, unpickle_data, unpickle_model


# --- DETECTION CONFIGURATION --- #
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

# --- LOAD MODEL --- #
# Coco Names
classesFile = r"config/coco.names"
classNames = []
with open(classesFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

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
            if confidence > confThreshold and classNames[classId] == 'car':
                w, h = int(det[2] * wT) , int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2) , int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        return x, y, w, h

def dnn_passthrough(img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    return net.forward(outputNames)

def color2tuple(color_label):
    colors = {
        "black":  (0, 0, 0),
        "blue":   (0xFF, 0, 0),
        "cyan":   (0xF5, 0xCE, 0x42),
        "gray":   (0x59, 0x59, 0x59),
        "green":  (0x00, 0xFF, 0),
        "red":    (0, 0, 0xFF),
        "white":  (0xFF, 0xFF, 0xFF),
        "yellow": (0, 0xE6, 0xff),
    }
    return colors[color_label]

def show_image(img, prediction, x, y, w, h):
    text_color = color2tuple(prediction[0])

    cv2.rectangle(img, (x, y), (x + w, y + h), text_color, 2)
    cv2.putText(img,f'{prediction[0]}',
                (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)

def detect_from_image(img):

    mlp = unpickle_model("config/color_recognition_mlp.pkl")

    outputs = dnn_passthrough(img)

    try:
        x, y, w, h = findObjects(outputs, img)
    except:
        print("Couldn't find a car in given frame")
        exit()

    img_cropped = img[y : y + h, x : x + w]

    width = height = 228
    img_cropped = cv2.resize(img_cropped, (width, height))

    cd = ColorDescriptor((16, 16, 16))
    image_colors = cd.describe(img_cropped)

    prediction = mlp.predict([image_colors])

    show_image(img, prediction, x, y, w, h)

def detect_from_video(cap):

    mlp = unpickle_model("config/color_recognition_mlp.pkl")

    while cap.isOpened():
        success, img = cap.read()
        # Stop if no video left to show
        if not success:
            break

        outputs = dnn_passthrough(img)

        img = findObjects(outputs, img)

        width = height = 228
        img = cv2.resize(img, (width, height))

        cd = ColorDescriptor((16, 16, 16))
        image = cd.describe(img)

        prediction = mlp.predict([img])

        img = cv2.putText(img, prediction[0], (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, color2tuple(prediction[0]), 2, cv2.LINE_AA)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # --- IMAGES --- #
    def run_on_images():
        yellow_car = cv2.imread(r"assets\real_tests\sanfran_yellow.jpg")
        blue_car = cv2.imread(r"assets\real_tests\seattle_blue.jpg")
        red_car = cv2.imread(r"assets\0197.jpg")
        gta_screen = cv2.imread(r"assets\gta5.jpg")

        detect_from_image(yellow_car)
        detect_from_image(blue_car)
        detect_from_image(red_car)
        detect_from_image(gta_screen)

    # --- VIDEO --- #
    def run_on_video():
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(r"assets/cars.mp4")
        detect_from_video(cap)

    run_on_images()
    # run_on_video()
