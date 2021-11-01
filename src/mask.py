import pixellib
import glob

from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model(r"config/pointrend_resnet50.pkl", detection_speed="normal")
# ins.load_model(r"config/pointrend_resnet50.pkl", detection_speed="rapid")
target_classes = ins.select_target_classes(car = True)
for image_name in glob.glob("assets/testing/0014.jpg"):
    results, output = ins.segmentImage(image_name, segment_target_classes = target_classes, extract_segmented_objects=True,
    save_extracted_objects=True, output_image_name="test.jpg")

print(results)