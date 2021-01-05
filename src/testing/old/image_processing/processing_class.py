import cv2
import time
import numpy as np
from PIL import Image
import bisect
from src.image_processing.filter_and_sorting import *

__all__ = ['ImageProcessing', 'filterContours', 'chooseContour']


class ImageProcessing:
    LANE_TOGGLE = {'RTL': 'LTR', 'LTR': 'RTL'}

    def __init__(self, track=3, camera=None, stream=None, intDetectDelay=6.9, intReactDelay=1.9):
        self.rawImg = None
        self.ocvImg = None
        self.canny = None

        self.fowWidth = 0
        self.fowHeight = 0
        self.center = None

        self.track = track
        self.intApproach = False
        self.laneSide = 'RTL'
        self.intDetectDelay = intDetectDelay
        self.intReactDelay = intReactDelay
        self.intDelay = 0

        if track == 3:
            self.intFunc = self.intFuncTrack3
            self.intCheck = lambda t, l, r: (len(t) + len(l) + len(r) >= 8)

        elif track == 2:
            self.intFunc = self.intFuncTrack2
            self.intCheck = lambda t, l, r: (len(t) + len(l) + len(r) >= 5)

        if camera and stream:
            self.fowWidth, self.fowHeight, self.center = self.getFOWInfo(camera, stream)

    def getContours(self, img):
        # RGB to BGR
        self.ocvImg = img[:, :, ::-1].copy()

        # To grayscale
        ocv_gray = cv2.cvtColor(self.ocvImg, cv2.COLOR_BGR2GRAY)

        # To binary
        thres, ocv_bin = cv2.threshold(ocv_gray, 160, 255, cv2.THRESH_BINARY)

        # OpenCV Canny
        self.canny = cv2.Canny(ocv_bin, 100, 200)

        # OpenCV Contours
        return cv2.findContours(self.canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def calculateErrors(self, contour, smallContour):
        """need to be called after getContours"""

        if smallContour is None:
            angContour = np.concatenate(contour)
            latContour = angContour
        elif self.intApproach:
            angContour = np.concatenate(smallContour)
            latContour = angContour
        else:
            angContour = np.concatenate(contour)
            latContour = np.concatenate(smallContour)

        x, y, w, h = cv2.boundingRect(latContour)

        cv2.rectangle(self.ocvImg, (x, y), (x + w, y + h), (21, 244, 238), 3)
        line = cv2.line(self.ocvImg, (x + int(w / 2), 0), (x + int(w / 2), 192), (255, 0, 0), 3)

        minAR = cv2.minAreaRect(angContour)
        (x_m, y_m), (w_m, h_m), ang = minAR

        mAR = cv2.boxPoints(minAR)
        mAR = np.int0(mAR)
        cv2.drawContours(self.ocvImg, [mAR], 0, (0, 0, 255), 3)

        if ang < -45:
            ang = 90 + ang
        if w_m < h_m and ang > 0:
            ang = (90 - ang) * -1
        if w_m > h_m and ang < 0:
            ang = 90 + ang

        sp_x = 256 / 2
        pv_x = (x + (w / 2))

        lateral_error = sp_x - pv_x

        return lateral_error, ang

    def checkForIntersection(self, conts, extrema, errMargin=3):
        if self.intApproach:
            if (time.time() - self.intDelay) < self.intReactDelay:
                return self.intFunc(conts, extrema)
            else:
                self.intApproach = False
                return None

        t, l, _, r = zip(*extrema)

        t = tuple(filter(lambda pos: pos[1] <= errMargin, t))
        l = tuple(filter(lambda pos: pos[0] <= errMargin, l))
        r = tuple(filter(lambda pos: pos[0] >= (self.fowWidth - errMargin), r))

        if self.intCheck(t, l, r) and (time.time() - self.intDelay > self.intDetectDelay):

            print('\n Intersection discovered ->  ({t} {r} {l}) \n'.format(t=len(t), r=len(r), l=len(l)))

            if self.track == 2:
                self.laneSide = self.LANE_TOGGLE.get(self.laneSide)
            self.intApproach = True
            self.intDelay = time.time()

            return self.intFunc(conts, extrema)

        return None

    def intFuncTrack2(self, conts, extrema):
        conts, extrema = sortContours(conts, extrema, order='TTB', then=self.laneSide)
        return conts[0], None

    def intFuncTrack3(self, conts, extrema):
        conts, extrema = sortContours(conts, extrema, order='RTL', then='BTT')

        return conts[0], self.shrinkContour(conts[0], extrema[0])

    def chooseContour(self, conts):
        extrema = getExtremaOfContours(conts)

        if self.track in (2, 3):
            intersection = self.checkForIntersection(conts, extrema)
            if intersection is not None:
                return intersection

        conts, extrema = sortContours(conts, extrema, order=self.laneSide, then='BTT')

        return conts[0], self.shrinkContour(conts[0], extrema[0])

    def shrinkContour(self, cont, ext):
        indexArr = cont[:, 0, 1].argsort()
        ySorted = cont[indexArr]
        y = ySorted[:, 0, 1]
        start = bisect.bisect_right(y, self.center[1])

        if len(ySorted) == start:
            return None

        return ySorted[start:]

    def getCannyImage(self, img):
        self.getContours(img)
        return self.canny

    def setFowInfo(self, w, h, center):
        self.fowWidth = w
        self.fowHeight = h
        self.center = center

    @classmethod
    def getFOWInfo(cls, camera, stream) -> tuple:
        camera.capture(stream, format='jpeg', use_video_port=True)

        im = Image.open(stream)
        stream.seek(0)
        stream.truncate(0)

        ocv_img = np.array(im)

        *_, w, h = reversed(ocv_img.shape)
        center = (int(w / 2), int(h / 2))

        return w, h, center
