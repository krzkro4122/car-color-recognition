from ast import increment_lineno
import joblib
import os

from skimage.io import imread
from skimage.transform import resize

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

            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)

        joblib.dump(data, pklname)


if __name__ == "__main__":
    data_path = fr'{os.getenv("HOME")}/rep/car-color-recognition/assets/train'
    pkl_name = 'config/car_colors'
    include = {'white', 'gray', 'yellow', 'black', 'blue', 'green', 'red', 'cyan'}
    width = 128

    # resize_all(src=data_path, pklname=pkl_name, width=width, include=include)

    from collections import Counter
    import numpy as np

    data = joblib.load(f'{pkl_name}_{width}x{width}px.pkl')

    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['description'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))

    Counter(data['label'])

    X = np.array(data['data'])
    y = np.array(data['label'])

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    
