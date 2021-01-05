from src.controller.pid_controller import PIDController
from src.image_processing import *
from easygopigo3 import EasyGoPiGo3
from io import BytesIO
from PIL import Image
import numpy as np
import picamera
import time
import cv2

gpg = EasyGoPiGo3()
camera = picamera.PiCamera()
camera.resolution = (256, 192)
camera.start_preview()
time.sleep(2)

stream = BytesIO()
imp = ImageProcessing(track=3, camera=camera, stream=stream)
controller = PIDController(kp=1.2, kd=1.5, maxAngularSpeed=200)

wheelDist = 0.125
wheelRadius = 2 * 0.033
linearSpeed = 2 * 100
test = 0

while test <= 600:

    camera.capture(stream, format='jpeg', use_video_port=True)
    im = Image.open(stream)
    ocv_img = np.array(im)

    cv2.imshow("Raw Image", ocv_img)
    cv2.waitKey(1)
    
    contours, hierarchy = imp.getContours(ocv_img)
    contours = filterContours(contours, filterThreshold=0.16)

    if contours.__len__():
        cv2.drawContours(imp.ocvImg, contours, -1, (0, 255, 0), 3)

        contours, smallContour = imp.chooseContour(contours)

        latErr, angErr = imp.calculateErrors(contours, smallContour)

        angularSpeed, terms = controller.getAngularSpeed(latErr, angErr)

        msg = "lateralError:{latErr} angularError:{angErr}\n P: {P} D:{D} \n" \
              "angularSpeed: {angSpeed}\n"

        print(msg.format(P=terms[0], D=terms[1], latErr=latErr, angErr=angErr,
                         angSpeed=angularSpeed))
    else:
        angularSpeed = controller.getSearchSpeed()
        print("Search angularSpeed: {angSpeed}".format(angSpeed=angularSpeed))

    leftMotorSpeed = (linearSpeed + ((angularSpeed * wheelDist) / wheelRadius))
    rightMotorSpeed = (linearSpeed - ((angularSpeed * wheelDist) / wheelRadius))

    gpg.set_motor_dps(gpg.MOTOR_LEFT, dps=leftMotorSpeed)
    gpg.set_motor_dps(gpg.MOTOR_RIGHT, dps=rightMotorSpeed)

    cv2.imshow("Camera Feed", imp.ocvImg)
    cv2.waitKey(1)

    stream.seek(0)
    stream.truncate(0)

    test += 1

gpg.stop()
cv2.destroyAllWindows()
