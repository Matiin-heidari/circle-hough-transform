import argparse

from utilities.image_circle_handling import *
from utilities.imarray import imarray


def command_line_runtime(file_path, threshold_value):
    img = imarray(file_path)
    res = smoothen(img,display=False)
    res = edge(res,128,display=False)
    res = detectCircles(res, threshold=threshold_value, region=20, radius=[100,10])
    displayCircles(res, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circle detection script")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for detection")
    args = parser.parse_args()
    command_line_runtime(args.image, args.threshold)
    print("Done")