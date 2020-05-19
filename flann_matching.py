import os
import statistics
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

#parser = argparse.ArgumentParser()
all_pngs = glob.glob('./*.png')
#parser.add_argument('--input1', type=str, required=True)
#parser.add_argument('--input2', type=str, required=True)

#args = parser.parse_args()
all_matches = []

for png in all_pngs:
    img1 = cv2.imread(png,0) # queryImage
    img2 = cv2.imread(png,0) # trainImage

    # Initiate detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    #FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]
    good_matches = []

    for pair in matches:
        try:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        except ValueError:
            pass
    all_matches.append(len(good_matches))

print(statistics.mean(all_matches))
