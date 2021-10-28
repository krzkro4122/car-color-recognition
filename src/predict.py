import cv2

import numpy as np
import tensorflow as tf

IMG_SIZE = 224

loaded_model = tf.keras.models.load_model('config/color_model.h5')
loaded_model.layers[0].input_shape #(None, 160, 160, 3)

# batch_holder = np.zeros((20, IMG_SIZE, IMG_SIZE, 3))
# img_dir='test_set/'
# for i,img in enumerate(os.listdir(img_dir)):
#   img = image.load_img(os.path.join(img_dir,img), target_size=(IMG_SIZE,IMG_SIZE))
#   batch_holder[i, :] = img

class_indices = {'black': 0, 'blue': 1, 'cyan': 2, 'gray': 3, 'green': 4, 'red': 5, 'white': 6, 'yellow': 7}

image_path="test/black_box.png"
image = cv2.imread(image_path)

img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
img = np.expand_dims(img, axis=0)
result = loaded_model.predict(img)
print(result)

for index, i in enumerate(result[0]):
    if i:
        label_name = list(class_indices.keys())[list(class_indices.values()).index(index)]

print(label_name)

image = cv2.putText(image, label_name, (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
cv2.imshow('output', image)
cv2.waitKey(0)