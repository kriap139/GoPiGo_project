import cv2
import time
import numpy as np
from PIL import Image
import bisect
from src.image_processing.filter_and_sorting import *

__all__ = ['ImageProcessing', 'filterContours']


class ImageProcessing:
    LANE_TOGGLE = {'RTL': 'LTR', 'LTR': 'RTL'}

    def __init__(self, track=0, camera=None, stream=None, intDetectDelay=6.9, intReactDelay=1.9):
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
            self.intCheck = lambda t, l, r: (
                    ((len(t) >= 3) and ((len(l) >= 3) or (len(r) >= 3))) or
                    ((len(t) == 0) and ((len(l) >= 3) and (len(r) >= 3)))
            )

        elif track == 2:
            self.intFunc = self.intFuncTrack2
            self.intCheck = lambda t, l, r: (
                    (len(t) >= 2) and (len(l) >= 2) and (len(r) >= 2)
            )

        if camera and stream:
            self.fowWidth, self.fowHeight, self.center = self.getFOWInfo(camera, stream)

    def getContours(self, img):
        """Find contours present in an image
        :returns [Contours], [hierarchy]
        """

        # RGB to BGR
        #self.ocvImg = img[:, :, ::-1].copy()
        self.ocvImg = img

        # To grayscale
        ocv_gray = cv2.cvtColor(self.ocvImg, cv2.COLOR_BGR2GRAY)

        # To binary
        thres, ocv_bin = cv2.threshold(ocv_gray, 160, 255, cv2.THRESH_BINARY)

        # OpenCV Canny
        self.canny = cv2.Canny(ocv_bin, 100, 200)

        # OpenCV Contours
        return cv2.findContours(self.canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def calculateErrors(self, contour, smallContour):
        """
        Calculates the lateral and angular error, between a passed contour and the center of the image.
        PS. Can only be called after getContours

        :returns lateral_error, angular_error
        """

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

        sp_x = self.center[0] / 2
        pv_x = (x + (w / 2))

        lateral_error = sp_x - pv_x

        return lateral_error, ang

    def checkForIntersection(self, conts, extrema, errMargin=3):
        """Checks if an intersection is present in the Image. If so specialized logic will be used to
        (intFuckTrack2 and intFuncTrack3) pick the desired contour

        :returns None or tuple(contour, smallContour) or tuple(contour, None)
        """

        delay = time.time() - self.intDelay

        if self.intApproach:
            if delay < self.intReactDelay:
                return self.intFunc(conts, extrema, delay)
            else:
                self.intApproach = False
                return None

        t, l, _, r = zip(*extrema)  # [(t0, t1, ...), (l0, l1, ...), (r0, r1, ...)], t=top, l=left, r=right, b=skipped
        t = tuple(filter(lambda pos: pos[1] <= errMargin, t))
        l = tuple(filter(lambda pos: pos[0] <= errMargin, l))
        r = tuple(filter(lambda pos: pos[0] >= (self.fowWidth - errMargin), r))

        if self.intCheck(t, l, r) and (delay > self.intDetectDelay):

            #print('\n Intersection discovered ->  ({t} {r} {l}) \n'.format(t=len(t), r=len(r), l=len(l)))

            if self.track == 2:
                self.laneSide = self.LANE_TOGGLE.get(self.laneSide)

            self.intApproach = True
            self.intDelay = time.time()

            return self.intFunc(conts, extrema, delay)

        return None

    def intFuncTrack2(self, conts, extrema, delay):
        """Code executed when an intersection is discovered on track 2. The picked Contour will be replaced with a
           straight line from the left/right bottom contour, to the top position of the Picked contour.
        """

        c, e = sortContours(conts, extrema, order='BTT', then=self.laneSide)
        ext = e[0]

        conts, extrema = sortContours(conts, extrema, order='TTB', then=self.laneSide)

        contourArr = [[ext[2][0], ext[2][1]], [extrema[0][0][0], extrema[0][0][1]]]
        createdContour = np.array(contourArr).reshape((-1, 1, 2)).astype(np.int32)

        return createdContour, None

    def intFuncTrack3(self, conts, extrema, delay):
        """Code executed When an intersection is discovered on track 3."""

        conts, extrema = sortContours(conts, extrema, order='RTL', then='BTT')

        return conts[0], self.shrinkContour(conts[0], toVal=self.center[1])

    def chooseContour(self, conts):
        """Finds the desired Contour, using the extrema values of each contour to estimate the
        contours position in the image."""

        extrema = getExtremaOfContours(conts)

        if self.track in (2, 3):
            intersection = self.checkForIntersection(conts, extrema)

            if intersection is not None:
                return intersection

        conts, extrema = sortContours(conts, extrema, order=self.laneSide, then='BTT')
        return conts[0], self.shrinkContour(conts[0], toVal=self.center[1])

    def shrinkContour(self, cont, toVal):
        """Creates a Cropped copy of the passed contour. This contour wil go from the
        coordinate with the biggest y value (of the original passed contour), to the coordinate with a y value, that is
        closest to the tovVal param."""

        indexArr = cont[:, 0, 1].argsort()
        ySorted = cont[indexArr]
        y = ySorted[:, 0, 1]

        start = bisect.bisect_right(y, toVal)

        if len(ySorted) == start:
            return None

        return ySorted[start:]

    def getCannyImage(self, img):
        self.getContours(img)
        return self.canny

    def setFowInfo(self, w, h, center: tuple):
        self.fowWidth = w
        self.fowHeight = h
        self.center = center

    @classmethod
    def getFOWInfo(cls, camera, stream) -> tuple:
        """Gets the height, width and center-point of the Field Of View (the size of images)"""

        camera.capture(stream, format='jpeg', use_video_port=True)
        im = Image.open(stream)
        ocv_img = np.array(im)

        stream.seek(0)
        stream.truncate(0)

        *_, w, h = reversed(ocv_img.shape)
        center = (int(w / 2), int(h / 2))

        return w, h, center
