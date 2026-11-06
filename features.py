"""Hand-crafted image features for the classical baseline (shared by baseline.py and app.py).

On a 256x256 version of the image:
  * HOG: 9 orientations, 32x32-pixel cells, 2x2-cell blocks                     -> 1764 values
  * LBP: uniform patterns (P=8, R=1), 10-bin histogram per cell of a 4x4 grid    -> 160
  * Intensity: global 32-bin histogram + mean and std per cell of a 4x4 grid     -> 64
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

FEAT_SIZE = 256


def grid_cells(img, n=4):
    h = img.shape[0] // n
    return [img[i * h:(i + 1) * h, j * h:(j + 1) * h] for i in range(n) for j in range(n)]


def extract_features(img512):
    """One uint8 512x512 image -> 1988-dim float32 feature vector."""
    img = cv2.resize(img512, (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA)
    f_hog = hog(img, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")  # values 0..9
    f_lbp = np.concatenate([np.bincount(c.astype(int).ravel(), minlength=10) / c.size for c in grid_cells(lbp)])
    f_hist = np.histogram(img, bins=32, range=(0, 256))[0] / img.size
    f_grid = np.array([[c.mean() / 255, c.std() / 255] for c in grid_cells(img)]).ravel()
    return np.concatenate([f_hog, f_lbp, f_hist, f_grid]).astype(np.float32)
