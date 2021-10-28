import pixellib
from pixellib.semantic import semantic_segmentation

def segment(image):
    segment_image = semantic_segmentation()
    segment_image.load_pascalvoc_model(r"config/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    segmap, output_array = segment_image.segmentAsPascalvoc(image, process_frame=True)

    return segmap, output_array
