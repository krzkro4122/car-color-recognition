import os
import cv2
import copy
import time
import joblib
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from mask import MaskTransformer


def log(variable):
    variables = globals().items()
    variable_info = list(variables)[-1]
    print(f"{variable_info[0]}: {variable_info[1]}")

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

    pklname = f"{pklname}_{width}x{height}pxSMALL.pkl"

    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)

            for index, file in enumerate(os.listdir(current_path)):
                if index >= 500:
                    break
                if file[-3:] in {'jpg', 'png'}:
                    im = cv2.imread(os.path.join(current_path, file))
                    im = cv2.resize(im, (width, height))
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)

        joblib.dump(data, pklname)


class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins
        self.mt = MaskTransformer()
        self.counter = 0
        self.start = time.time()

    def describe(self, image):
        features = []

        mask = self.mt.mask_frame(image)

        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

        self.counter += 1
        if self.counter % 50 == 0:
            print("Batch complete [{:.2f}s] - {}/{}".format(time.time()-self.start, self.counter, len(data['data'])))

        return features


if __name__ == "__main__":

    start_stamp = time.time()

    data_path = fr'{os.getenv("HOME")}/rep/car-color-recognition/assets/train'
    pkl_name = 'config/car_colors'
    include = {'white', 'gray', 'yellow', 'black', 'blue', 'green', 'red', 'cyan'}
    width = 228

    # print("Resizing and pickling arrays of images...")
    # resize_all(src=data_path, pklname=pkl_name, width=width, include=include)
    # print("Pickling done")

    print("Unpickling images data...", end="")
    data = joblib.load(f'{pkl_name}_{width}x{width}pxSMALL.pkl')
    print("Done")

    # Post-unpickling Info
    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['description'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))
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

    print("Feature extraction...")
    X_train_bkp = copy.copy(X_train)
    X_test_bkp = copy.copy(X_test)
    X_train = []
    X_test = []
    for index, x in enumerate(X_train_bkp):
        try:
            X_train.append(cd.describe(X_train_bkp[index]))
        except:
            pass

    for index, x in enumerate(X_test_bkp):
        try:
            X_test.append(cd.describe(X_test_bkp[index]))
        except:
            pass
    print("Done ({}/{} images. [{:.2f}])".format(cd.counter, len(data['data']), time.time()-cd.start))

    # features_pkl = r"config/features.pkl"
    # joblib.dump([X_train, X_test], features_pkl)
    # joblib.load(features_pkl)

    print("Machine is learning...", end="")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 100, 100),
        max_iter=5000, alpha=0.005,
        solver='adam', random_state=2,
        activation='relu', learning_rate='constant'
    )
    mlp.fit(X_train, y_train)
    print("Done")

    config_file = 'config/color_recognition_mlp_masked.pkl'

    print("Pickling model...", end="")
    joblib.dump(mlp, config_file)
    print("Done")

    print("Unpickling model...", end="")
    mlp = joblib.load(config_file)
    print("Done")

    y_pred = mlp.predict(X_test)
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

    from sklearn.metrics import classification_report
    print('Results on the test set:')
    print(classification_report(y_test, y_pred))

    end_stamp = time.time()
    time_passed = end_stamp - start_stamp
    print(f"Run took: {time_passed:.0f}s ({time_passed/60:.2f}m)")

    # # Show predictions for a given color
    # color = 'black'
    # for index, prediction in enumerate(y_pred):
    #     # if prediction == color:
    #     if prediction != y_test[index]:
    #         img = X_test_bkp[index]
    #         img = cv2.putText(img, prediction.upper(), (50, 50), cv2.FONT_HERSHEY_PLAIN,
    #             2, (255, 255, 255), 2, cv2.LINE_AA)
    #         cv2.imshow(f"{index}", img)
    #         cv2.waitKey(0)
