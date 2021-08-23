import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter



def open_idx(df: pd.DataFrame, root_path, i):
    '''

    :param df: Input dataframe containing the images
    :param root_path: the root path of the project
    :param i: the index of the image
    :return: the image object
    '''
    file_name = df['NombreArchivo'][i]
    folder = df['Conjunto'][i]
    return Image.open(os.path.join(root_path, folder, file_name))


def show_bgr_image_in_plt(img):
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def find_contour_mask(img):
    global thresh, morph, mask, result1, result2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold
    thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)[1]
    # apply morphology
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((9, 9), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel)
    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c
    # get bounding box
    x, y, w, h = cv2.boundingRect(big_contour)
    # draw filled contour on black background
    mask = np.zeros_like(gray)
    mask = cv2.merge([mask, mask, mask])
    cv2.drawContours(mask, [big_contour], -1, (255, 255, 255), cv2.FILLED)
    # apply mask to input
    result1 = img.copy()
    result1 = cv2.bitwise_and(result1, mask)
    # crop result
    result2 = result1[y:y + h, x:x + w]
    return result2

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

def preprocessing(image):
    image = np.asarray(image)
