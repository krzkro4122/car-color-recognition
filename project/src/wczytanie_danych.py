import os
import cv2 as cv

data_path = r'assets/train' # Główny folder danych
data = { "images": [], "label": [] }

for subdir in os.listdir(data_path): # Przejście przez każdy podfolder
    current_path = os.path.join(data_path, subdir)

    for file_name in os.listdir(current_path): # Przejście przez każdy plik w podfolderze
        image_path = os.path.join(current_path, file_name)
        image = cv.imread(image_path)
        data["imagees"].append(image)
        data["label"].append(subdir)