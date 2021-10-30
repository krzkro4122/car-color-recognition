import joblib
import os
import cv2
import copy
import numpy as np
import semantic_segmentation

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def resize_all(src, pklname, include, width=228, height=None):
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

    height = height if height is not None else width

    data = dict()
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    data['description'] = f"resized car images to ({width}x{height}) in rgb"

    pklname = f"{pklname}_{width}x{height}px.pkl"

    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)

            for index, file in enumerate(os.listdir(current_path)):
                if index >= 1000:
                    break
                if file[-3:] in {'jpg', 'png'}:
                    im = cv2.imread(os.path.join(current_path, file))
                    im = cv2.resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)

        joblib.dump(data, pklname)

import tensorflow as tf
class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        features = []
        # # Add a mask
        # _, mask = semantic_segmentation.segment(image)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # thresh, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # Extract color histogram
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
        return features


if __name__ == "__main__":

    data_path = fr'{os.getenv("HOME")}/rep/car-color-recognition/assets/train'
    pkl_name = 'config/car_colors'
    include = {'white', 'gray', 'yellow', 'black', 'blue', 'green', 'red', 'cyan'}
    width = 228

    # print("Resizing and pickling images...")
    # resize_all(src=data_path, pklname=pkl_name, width=width, include=include)
    # print("Pickling done")

    print("Unpickling data...", end="")
    data = joblib.load(f'{pkl_name}_{width}x{width}px.pkl')
    print("Done")

    # # Post-unpickling Info
    # print('number of samples: ', len(data['data']))
    # print('keys: ', list(data.keys()))
    # print('description: ', data['description'])
    # print('image shape: ', data['data'][0].shape)
    # print('labels:', np.unique(data['label']))
    print(f"The data spread: {Counter(data['label'])}")

    X = np.array(data['data'])
    y = np.array(data['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    # Histograms instead of raw images
    cd = ColorDescriptor((16, 16, 16))
    print("Feature extraction...", end="")
    X_train_bkp = copy.copy(X_train)
    X_test_bkp = copy.copy(X_test)
    X_train = list(map(lambda x:cd.describe(x), X_train))
    X_test = list(map(lambda x:cd.describe(x), X_test))
    print("Done")

    print("Machine is learning...", end="")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 100, 100),
        max_iter=5000, alpha=0.005,
        solver='adam', random_state=2,
        activation='relu', learning_rate='constant'
    )

    # # Grid search
    # from sklearn.model_selection import GridSearchCV

    # parameter_space = {
    #     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (100, 100, 100)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.05, 1e-3, 5e-3, 0.5e-3],
    #     'learning_rate': ['constant','adaptive'],
    # }

    # clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    # clf.fit(X_train, y_train)

    # print('Best parameters found:\n', clf.best_params_)

    # # All results
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # y_true, y_pred = y_test , clf.predict(X_test)

    # from sklearn.metrics import classification_report
    # print('Results on the test set:')
    # print(classification_report(y_true, y_pred))

    mlp.fit(X_train, y_train)
    print("Done")

    y_pred = mlp.predict(X_test)

    # print(y_pred[200])
    # cv2.imshow('lol', X_test_bkp[200])
    # cv2.waitKey(0)

    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))