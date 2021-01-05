import cv2 as cv2
import numpy as np
import random as rand

__all__ = ['drawContoursOneByOne', 'getRandomColor', 'filterContours', 'getExtremaOfContours',
           'chooseContour', 'getRandomColors', 'sortContours', 'drawExtremaOneByOne', 'getDotProduct',
           'sortTesting']


def drawContoursOneByOne(conts, img, sameColors=True):
    if not sameColors:
        colors = list(getRandomColors(len(conts)))

        for i, cont in enumerate(conts):
            cv2.drawContours(img, [cont], -1, colors[i], 3)
            cv2.imshow("contours", img)
            cv2.waitKey(0)
    else:
        for cont in conts:
            cv2.drawContours(img, [cont], -1, (0, 255, 0), 3)
            cv2.imshow("contours", img)
            cv2.waitKey(0)


def drawExtremaOneByOne(conts, extrema, img):
    for i in range(conts.__len__()):
        colors = getRandomColors(5)
        cv2.drawContours(img, [conts[i]], -1, (0, 255, 0), 3)

        cv2.circle(img, extrema[i][0], 8, colors[0], -1)

        cv2.imshow("contours", img)
        cv2.waitKey(0)

        cv2.circle(img, extrema[i][1], 8, colors[1], -1)

        cv2.imshow("contours", img)
        cv2.waitKey(0)

        cv2.circle(img, extrema[i][2], 8, colors[2], -1)

        cv2.imshow("contours", img)
        cv2.waitKey(0)

        cv2.circle(img, extrema[i][3], 8, colors[3], -1)

        cv2.imshow("contours", img)
        cv2.waitKey(0)


def getRandomColors(N):
    return tuple(getRandomColor() for i in range(N))


def getRandomColor():
    return rand.randint(0, 256), rand.randint(0, 256), rand.randint(0, 256)


def filterContours(conts, hierarchy, requiredLength=19, filterThreshold=0.6):
    if not conts.__len__():
        return tuple()

    lengths = tuple(cv2.arcLength(cont, True) for cont in conts)
    longest = np.max(lengths)
    filtered = []

    for i in range(conts.__len__()):
        if lengths[i] > requiredLength and (lengths[i] / longest > filterThreshold):
            filtered.append(conts[i])

    return filtered


def getDotProduct(pos1, pos2):
    return pos1[0]*pos2[0] + pos1[1]*pos1[1]


def getExtremaOfContours(conts):
    extrema = []
    for c in conts:
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBottom = tuple(c[c[:, :, 1].argmax()][0])
        extrema.append((extTop, extLeft, extBottom, extRight))

    return extrema


def sortContoursOG(conts, extrema=None, order='RTL', then=None, unzip=True):
    if (order == 'RTL') or (order == 'LTR'):
        i = 3
        j = 0
    elif (order == 'TTB') or (order == 'BTT'):
        i = 0
        j = 1
    else:
        raise TypeError('Order Argument has to be one of these values (RTL, LTR, TTB, BTT)')

    rev = True if (order == 'RTL') or (order == 'BTT') else False

    if extrema is None:
        extrema = getExtremaOfContours(conts)

    sortedConts = list(zip(conts, extrema))
    sortedConts.sort(key=lambda tup: tup[1][i][j], reverse=rev)

    if not unzip:
        return sortedConts

    return tuple(zip(*sortedConts))


def sortContours(conts, extrema=None, order='RTL', then=None,keepAll=False, unzip=True):
    if order == 'RTL':
        i = 3
        j = 0
        rev = True
    elif order == 'LTR':
        i = 1
        j = 0
        rev = False
    elif order == 'TTB':
        i = 0
        j = 1
        rev = False
    elif order == 'BTT':
        i = 2
        j = 1
        rev = True
    else:
        raise TypeError('order Parameter has to be one of these values (RTL, LTR, TTB, BTT)')

    if extrema is None:
        extrema = getExtremaOfContours(conts)

    sortedConts = list(zip(conts, extrema))
    sortedConts.sort(key=lambda tup: tup[1][i][j], reverse=rev)

    if then:
        compVal = sortedConts[0][1][i][j]
        newList = [o for o in sortedConts if o[1][i][j] == compVal]

        if then == 'RTL':
            newList.sort(key=lambda tup: tup[1][i][0], reverse=True)
        elif then == 'LTR':
            newList.sort(key=lambda tup: tup[1][i][0], reverse=False)
        elif then == 'TTB':
            newList.sort(key=lambda tup: tup[1][i][1], reverse=False)
        elif then == 'BTT':
            newList.sort(key=lambda tup: tup[1][i][1], reverse=True)

        if keepAll:
            for i in range(newList.__len__(), sortedConts.__len__()):
                newList.append(sortedConts[i])

        sortedConts = newList

    return tuple(zip(*sortedConts))


def sortTesting(conts, img, imgShape):
    extrema = getExtremaOfContours(conts)
    conts, extrema = sortContours(conts, extrema, order='BTT')
    drawContoursOneByOne(conts, img)

    return conts


def chooseContour(conts, imgShape):
    extrema = getExtremaOfContours(conts)
    #conts, extrema = sortContours(conts, extrema)

    return conts[0]


class Vector:
    PI_OVER_TWO = np.pi / 2

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, other):
        return self.x * other.x + self.y + other.y

    def len(self):
        return np.sqrt(self.x**2 + self.y**2)

    def theta(self):
        if self.x == 0:
            return self.PI_OVER_TWO
        else:
            return np.arctan((self.y / self.x))

    def __str__(self):
        return "({x},{y})".format(x=self.x, y=self.y)

    @classmethod
    def create(cls, p1: tuple, p2: tuple):
        """creates a Vector from p1 to p2"""

        return cls(p2[0] - p1[0], p2[1] - p1[1])
