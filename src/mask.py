import numpy as np
import glob
import cv2
import os

import warnings
warnings.filterwarnings("ignore")

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


class MaskTransformer:

    def __init__(self) -> None:
        self.ins = instanceSegmentation()
        self.ins.load_model(r"config/pointrend_resnet50.pkl", detection_speed="rapid")
        self.target_classes = self.ins.select_target_classes(car = True)

    def mask_frame(self, image):

        results, output = self.ins.segmentFrame(image, segment_target_classes = self.target_classes,
            extract_segmented_objects=True, mask_points_values=True)

        height, width, _ = image.shape
        dummy = create_blank(width=width, height=height)

        masked_image = dummy.copy()

        mask = None

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

            masked_image_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            thresh, mask = cv2.threshold(masked_image_gray, 1, 255, cv2.THRESH_BINARY)

        return mask

    def mask_batch(self):

        all_batches = glob.glob(r"assets/train/*")
        all_batches.sort(reverse=True)

        masked_counter = 0
        all_counter = 0

        for batch in all_batches:

            print(os.path.split(batch)[1])

            image_names = glob.glob(os.path.join(batch, "*"))
            image_names.sort()

            for index, image_name in enumerate(image_names):

                # if index >= 20:
                #     break

                image = cv2.imread(image_name)

                results, output = self.ins.segmentFrame(image, segment_target_classes = self.target_classes,
                    extract_segmented_objects=True, mask_points_values=True)

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
                    try:
                        temporary_masked_image[y_offset:y_offset + height, x_offset:x_offset + width] = extracted_object
                    except:
                        pass

                    masked_image = cv2.bitwise_or(masked_image, temporary_masked_image)

                head, tail = os.path.split(batch)
                output_path = os.path.join(r"assets", r"train_mask", tail, str(index).zfill(4) + r".jpg")

                all_counter += 1

                if masked_image.any() != 0:
                    cv2.imwrite(output_path, masked_image)
                    masked_counter += 1

                # print(f"Written image taken from {image_name}\n to\n {output_path}")
        print(f"Written {masked_counter}/{all_counter} images")


if __name__ == "__main__":
    import time
    start = time.time()

    mt = MaskTransformer()

    image = cv2.imread(r"assets/train/cyan/0059.jpg")
    masked_image = mt.mask_frame(image)
    # mt.mask_batch()

    print(f"Run took {((time.time()-start)):.2f}s")

    cv2.imshow('Image', image)
    cv2.imshow('Masked image', masked_image)
    cv2.waitKey(0)