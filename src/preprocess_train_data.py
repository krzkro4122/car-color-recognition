import os
import cv2
import copy
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins
        self.counter = 0
        self.start = time.time()

    def describe(self, image):
        features = []

        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

        self.counter += 1

        return features


def timer(func):
    def wrapper(*args, **kwargs):
        start_stamp = time.time()
        start = time.time()
        rv = func(*args, **kwargs)
        end_stamp = time.time()
        time_passed = end_stamp - start_stamp
        print(f"Run took: {time_passed:.0f}s ({int(time_passed//60)}m {int(time_passed%60)}s)")

        return rv

    return wrapper


def resize_and_pickle_all(src, pklname, include, width=228, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary,
    together with labels and metadata. The dictionary is written to a pickle file
    named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """

    print("Resizing and pickling arrays of images...", end="")

    height = height if height is not None else width

    data = {
        "label": [],
        "filename": [],
        "data": [],
        "description": f"resized car images to ({width}x{height}) in rgb",
    }

    pklname = f"{pklname}_{width}x{height}px.pkl"

    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(f"{subdir}...", end="")
            current_path = os.path.join(src, subdir)

            for index, file in enumerate(os.listdir(current_path)):
                # if index >= 500:
                #     break
                if file[-3:] in {"jpg", "png"}:
                    im = cv2.imread(os.path.join(current_path, file))
                    im = cv2.resize(im, (width, height))
                    data["label"].append(subdir)
                    data["filename"].append(file)
                    data["data"].append(im)

        joblib.dump(data, pklname)

    print("Done")


def feature_extraction(X_train, X_test, data):
    print("Feature extraction...", end="")
    # Histograms instead of raw images
    cd = ColorDescriptor((16, 16, 16))

    X_train = list(map(lambda x: cd.describe(x), X_train))
    X_test = list(map(lambda x: cd.describe(x), X_test))
    print("Done ({}/{} images. [{:.2f}s])".format(cd.counter, len(data["data"]), time.time() - cd.start))

    return X_train, X_test


def unpickle_data(pkl_name, width):
    print("Unpickling images data...", end="")
    data = joblib.load(f"{pkl_name}_{width}x{width}px.pkl")
    print("Done")

    # Post-unpickling Info
    # print("Number of samples: ", len(data["data"]))
    # print("Keys: ", list(data.keys()))
    # print("Description: ", data["description"])
    # print("Image shape: ", data["data"][0].shape)
    # print("Labels:", np.unique(data["label"]))
    print(f"The data spread: {Counter(data['label'])}")

    return data


def train_and_pickle_model(X_train, y_train, config_file):
    print("Machine is learning...", end="")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 100, 100),
        max_iter=5000,
        alpha=0.005,
        solver="adam",
        random_state=2,
        activation="relu",
        learning_rate="constant",
    )
    mlp.fit(X_train, y_train)
    print("Done")

    print("Pickling model...", end="")
    joblib.dump(mlp, config_file)
    print("Done")


def unpickle_model(config_file):
    print("Unpickling model...", end="")
    mlp = joblib.load(config_file)
    print("Done")
    return mlp


def generate_metrics(y_pred, y_test, data, X_test, mlp):
    print("Percentage correct: ", 100 * np.sum(y_pred == y_test) / len(y_test))

    print("Results on the test set:")
    print(classification_report(y_test, y_pred))

    print("Let's confuse some Matrices...")
    # confusion_matrices = []

    # for label in data["label"]:
    #     print(label)
    cm = confusion_matrix(y_test, y_pred, labels=data["label"])

    disp = ConfusionMatrixDisplay.from_estimator(mlp, X_test, y_test)
    plt.show()

    print(cm)


@timer
def main():
    data_path = fr'{os.path.expanduser("~")}/rep/car-color-recognition/assets/train'
    pkl_name = "config/car_colors"
    config_file = "config/color_recognition_mlp.pkl"
    include = {"white", "gray", "yellow", "black", "blue", "green", "red", "cyan"}
    width = 228

    # resize_and_pickle_all(src=data_path, pklname=pkl_name, width=width, include=include)

    data = unpickle_data(pkl_name, width)

    X = np.array(data["data"])
    y = np.array(data["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        shuffle=True,
        random_state=42,
    )

    X_train, X_test = feature_extraction(X_train, X_test, data)

    # train_and_pickle_model(X_train, y_train, config_file)

    mlp = unpickle_model(config_file)

    y_pred = mlp.predict(X_test)

    generate_metrics(y_pred, y_test, data, X_test, mlp)


if __name__ == "__main__":
    main()
