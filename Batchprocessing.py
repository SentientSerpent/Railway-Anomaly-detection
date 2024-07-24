# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time
import math
import scipy.ndimage
import os

# construct the argument parse and parse the arguments
font = cv2.FONT_HERSHEY_SIMPLEX
def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)
    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(
        data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory containing images")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory for modified images")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# load the images from the input directory
input_dir = args["input"]
output_dir = args["output"]

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])

# Open text file for writing results
with open(os.path.join(output_dir, 'results.txt'), 'w') as file:
    for image_path in image_paths:
        # load the image
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        frame = imutils.resize(image, width=800)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame.copy()

        fudgefactor = 1.5 
        sigma = 50 # for Gaussian Kernel
        kernel = 2*math.ceil(2*sigma)+1 # Kernel size

        gray_image = gray_image/255.0
        blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
        gray_image = cv2.subtract(gray_image, blur)

        # compute sobel response
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        mag = np.hypot(sobelx, sobely)
        ang = np.arctan2(sobely, sobelx)

        # threshold
        threshold = 4 * fudgefactor * np.mean(mag)
        mag[mag < threshold] = 0

        # non-maximal suppression
        mag = orientated_non_max_suppression(mag, ang)

        # create mask
        mag[mag > 0] = 255
        mag = mag.astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
        num_white = np.sum(result == 255)
        num_black = np.sum(result == 0)
        ratio = (num_white/num_black)*100

        if ratio > 0.7:
            result_text = "Cracked"
            cv2.putText(img, 'Cracked', (0, 30),
                        font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            result_text = "Not Cracked"
            cv2.putText(img, 'Not Cracked', (0, 30),
                        font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Write results to text file
        file.write(f"Image: {image_name}, Outcome: {result_text}, Ratio: {ratio}\n")

        # Save modified image to output directory
        cv2.imwrite(os.path.join(output_dir, image_name), img)

        # Show results
        cv2.imshow('im', img)
        cv2.imshow('im2', result)
        cv2.waitKey(1)  # Change from 0 to 1 to proceed to the next image automatically

cv2.destroyAllWindows()
