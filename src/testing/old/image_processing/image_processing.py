import cv2
import numpy as np
from PIL import Image

__all__ = ['getContours', 'calculateErrors', 'getFOWInfo']


def getContours(img):
    # RGB to BGR
    ocv_img = img[:, :, ::-1].copy()

    # To grayscale
    ocv_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # To binary
    thres, ocv_bin = cv2.threshold(ocv_gray, 160, 255, cv2.THRESH_BINARY)

    # OpenCV Canny
    canny = cv2.Canny(ocv_bin, 100, 200)

    # OpenCV Contours
    return *cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE), ocv_img


def calculateErrors(contour, img):
    contours = np.concatenate(contour)

    x, y, w, h = cv2.boundingRect(contours)

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    line = cv2.line(img, (x + int(w / 2), 0), (x + int(w / 2), 192), (255, 0, 0), 3)

    minAR = cv2.minAreaRect(contours)
    (x_m, y_m), (w_m, h_m), ang = minAR

    mAR = cv2.boxPoints(minAR)
    mAR = np.int0(mAR)
    cv2.drawContours(img, [mAR], 0, (0, 0, 255), 3)

    if ang < -45:
        ang = 90 + ang
    if w_m < h_m and ang > 0:
        ang = (90 - ang) * -1
    if w_m > h_m and ang < 0:
        ang = 90 + ang

    sp_x = (256 / 2)
    pv_x = (x + (w / 2))

    lateral_error = sp_x - pv_x

    return lateral_error, ang


def getFOWInfo(camera, stream):
    camera.capture(stream, format='jpeg', use_video_port=True)

    im = Image.open(stream)
    stream.seek(0)
    stream.truncate(0)

    ocv_img = np.array(im)

    *_, w, h = reversed(ocv_img.shape)
    center = (int(w / 2), int(h / 2))

    return w, h, center
