import cv2
import time
import numpy as np

from fastseg import MobileV3Small
from fastseg.image import colorize


image = cv2.imread("assets/tesla.jpg")
# image = image[:, :, ::-1].copy() # Convert to RGB

print("Evaluating model...", end="")
start = time.time()

model = MobileV3Small.from_pretrained()
model.eval()

stop = time.time()
print(f"Done in {stop-start:.2f}s")

labels = model.predict_one(image)

colorized = colorize(labels)
colorized = np.array(colorized)[:, :, ::-1]

cv2.imshow('lol', colorized)
cv2.waitKey(0)
