from src.image_processing.processing_class import *
from src.image_processing.filter_and_sorting import *
from src.testing.testing_utils import *
from PIL import Image
import numpy as np
import time
import cv2

imp = ImageProcessing(track=2, intReactDelay=1.9, intDetectDelay=6)
CREATE_VIDEO = False

PATH_TO_VIDEO_FOLDER = "C:\\Users\\kripi\\OneDrive - Østfold University College\\TeknologiProsjekt\\tmp\\edits"
BANE = 3
VIDEO_NUMBER = 5
CHILD_FOLDER = ''
VIDEO_NAME = 'raw-cam-feed.mp4'
VIDEO_PATH = f"{PATH_TO_VIDEO_FOLDER}\\bane{BANE}\\{VIDEO_NUMBER}\\{CHILD_FOLDER}\\{VIDEO_NAME}"

# Kan få memory error hvis du kjører koden med en 32-bit python interpreter, Grunnet at en
# 32-bit interpeter maks kan adressere 4gb, så denne kodedelen bør kjøres på en 64-bit interpeter
if CREATE_VIDEO:
    SAVE_FOLDER_PATH = f"{PATH_TO_VIDEO_FOLDER}\\bane{BANE}\\{VIDEO_NUMBER}\\canny-sequence"
    images = ImageBuffer(video=VIDEO_PATH)
    images.map(func=imp.getCannyImage)
    images.saveSequence(folder=SAVE_FOLDER_PATH)

    for img in images:
        cv2.imshow("Test", img)
        cv2.waitKey(1)
    exit(0)

# Bør også kjøres på en 64-bit interpreter, grunnett samme årsak som over
images = ImageBuffer(video=VIDEO_PATH)

test = 0

selected = tuple(range(1333, 1360))
s = [513, 906, 1293, 915, 1271]
im = Image.open(".\\test_images\\t39c.png")

while test < images.__len__():

    if test >= 103:
        ocv_img = np.array(im)
        #ocv_img = images[test]

        *_, w, h = reversed(ocv_img.shape)
        center = (int(w / 2), int(h / 2))
        imp.setFowInfo(w, h, center)

        contours, hierarchy = imp.getContours(ocv_img)
        contours = filterContours(contours, filterThreshold=0.16)

        if contours.__len__():
            #drawContoursOneByOne(contours, imp.ocvImg)
            #cv2.drawContours(imp.ocvImg, contours, -1, (0, 255, 0), 3)
            contour, smallContour = imp.chooseContour(contours)
            #drawContoursOneByOne([contour], imp.ocvImg)

            #if imp.intApproach and (time.time() - imp.intDelay < imp.intReactDelay):
                #selected.append(test)

            print('\r', test, end='')

            #if imp.intApproach:
             #    imp.intDelay = time.time()


            #latErr, angErr = imp.calculateErrors(contour, smallContour)
            #drawContoursOneByOne(contours, imp.ocvImg)
            drawExtremaOneByOne(contours, getExtremaOfContours(contours), imp.ocvImg)
            #print(f"lateralError: {latErr} angularError: {angErr} index: {test}\n")

            #cv2.imshow("Test", imp.ocvImg)
            #cv2.waitKey(1)

    test += 1

#print(selected)
cv2.destroyAllWindows()