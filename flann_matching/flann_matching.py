import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

img1 = cv2.imread('1-000.png',0)          # queryImage
img2 = cv2.imread('1-000.png',0) # trainImage

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
#for i, pair in enumerate(matches):
#    try:
#        m, n = pair
#        if m.distance < 0.7 * n.distance:
#            good_matches.append(m)
            #matchesMask[i]=[1,0]
#    except ValueError:
#        pass

for pair in matches:
    try:
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    except ValueError:
        pass

print(len(good_matches))

img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Good matches', img_matches)

#draw_params = dict(matchColor = (0,255,0),
#                   singlePointColor = (255,0,0),
#                   matchesMask = matchesMask,
#                   flags = 0)

# create BFMatcher object
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
#matches = bf.match(des1,des2)

# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:40],None,flags=2)
#img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)


#plt.imshow(img3),plt.show()
cv2.waitKey()
