from pixellib.semantic import semantic_segmentation
from pixellib.torchbackend.instance import instanceSegmentation
import cv2


def segment(image):
    segment_image = semantic_segmentation()
    segment_image.load_pascalvoc_model(r"config/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    segmap, output_array = segment_image.segmentFrameAsPascalvoc(image)

    return segmap, output_array


def instance(image):
    ins = instanceSegmentation()
    ins.load_model(r"config/pointrend_resnet50.pkl")
    segmap, output_array = ins.segmentFrame(image)

    return segmap, output_array


def main():
    print("Loading image...", end="")
    img_pre = cv2.imread(r"assets/gta5.jpg")
    print("Done")

    print("Semantic...", end="")
    img_sem = segment(img_pre)
    print("Done")

    print("Instance...", end="")
    img_ins = instance(img_pre)
    print("Done")

    cv2.imshow("semantic", img_sem)
    cv2.imshow("instance", img_ins)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
