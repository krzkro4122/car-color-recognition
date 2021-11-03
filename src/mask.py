import numpy as np
import pixellib
import glob
import cv2

from pixellib.torchbackend.instance import instanceSegmentation

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def mask_image(image):
    ins = instanceSegmentation()
    ins.load_model(r"config/pointrend_resnet50.pkl", detection_speed="rapid")
    target_classes = ins.select_target_classes(car = True)
    # for image_name in glob.glob("assets/testing/0002.jpg"):
    results, output = ins.segmentFrame(image, segment_target_classes = target_classes, extract_segmented_objects=True,
    save_extracted_objects=True, mask_points_values=True)

    height, width, _ = image.shape
    dummy = create_blank(width=width, height=height)

    masked_image = dummy.copy()

    for extracted_object in results['extracted_objects']:
        x_offset = results['boxes'][0][0]
        y_offset = results['boxes'][0][1]
        height = extracted_object.shape[0]
        width  = extracted_object.shape[1]

        # print(results['boxes'])
        # print(f"width: {width}")
        # print(f"height: {height}")
        # print(f"x_offset: {x_offset}")
        # print(f"y_offset: {y_offset}")

        # print(masked_image.shape)
        # print(extracted_object.shape)
        # print(masked_image[y_offset:y_offset + height, x_offset:x_offset + width].shape)

        temporary_masked_image = masked_image.copy()
        temporary_masked_image[y_offset:y_offset + height, x_offset:x_offset + width] = extracted_object

        masked_image = cv2.bitwise_or(masked_image, temporary_masked_image)

    return masked_image

if __name__ == "__main__":
    image = cv2.imread(r"assets/testing/0002.jpg")
    masked_image = mask_image(image)

    cv2.imshow('Image', image)
    cv2.imshow('Masked image', masked_image)
    cv2.waitKey(0)