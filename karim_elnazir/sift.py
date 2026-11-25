import cv2
import numpy as np


def detect_sift(img, max_points=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_points)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def keypoint_xy(keypoints):
    if not keypoints:
        return np.zeros((0, 2), np.float32)
    return np.float32([k.pt for k in keypoints])


def draw_keypoints(img, keypoints):
    return cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
