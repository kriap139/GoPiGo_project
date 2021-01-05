from easygopigo3 import EasyGoPiGo3
import picamera
import cv2
from PIL import Image, ImageEnhance # native res er 1024 x 768
import numpy as np, time, math
from time import sleep
from io import BytesIO
#from pid_controler import PIDController
from contour_filter_funcs import *

gpg = EasyGoPiGo3()
camera = picamera.PiCamera()
camera.resolution = (256,192)
camera.start_preview()
time.sleep(2)

kp = 1.2
ki = 0.01
kd = 20
timeAcc=0
maxAngVel = 200
linVel = 100
searchRadius = 9
radiusIncrement = 1
angVel_no = 100

#controller = PIDController(kp,ki,kd, maxAngularSpeed=100)
stream = BytesIO()
test = 0
while True:
    
    if (test>100): #Max iterations
        break
    
    camera.capture(stream, format='jpeg', use_video_port=True)
    im = Image.open(stream)
    ocv_img = np.array(im)

    #RGB to BGR 
    ocv_img = ocv_img[:, :, ::-1].copy()
    stream.seek(0)
    #cv2.imshow("raw", ocv_img)

    #To grayscale
    ocv_gray = cv2.cvtColor(ocv_img, cv2.COLOR_BGR2GRAY)

    #To binary
    thres, ocv_bin = cv2.threshold(ocv_gray, 160, 255, cv2.THRESH_BINARY)

    #OpenCV Canny
    canny = cv2.Canny(ocv_bin,100,200)
    #cv2.imshow("canny", canny)

    #OpenCV Contours
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ocv_img, contours, -1, (0, 255, 0), 3)

    contours = filterContours(contours)

    *_, w, h = reversed(ocv_img.shape)
    center = (int(w/2), int(h/2))

    if len(contours) > 0:
        #Contours may have to be sorted. Sorting from right to left as standard.

        contours = sortTesting(contours, ocv_img, imgShape=(w, h, center))
        
        contours = np.concatenate((contours[0]))
        
        x, y, w, h = cv2.boundingRect(contours)
        cv2.rectangle(ocv_img,(x,y),(x+w,y+h),(0,0,255),3)
        line = cv2.line(ocv_img,(x+int(w/2),0),(x+int(w/2),192),(255,0,0),3)

        minAR = cv2.minAreaRect(contours)
        (x_m, y_m),(w_m,h_m),ang = minAR

        mAR = cv2.boxPoints(minAR)
        mAR = np.int0(mAR)
        cv2.drawContours(ocv_img,[mAR],0,(0,0,255),3)

        if ang < -45:
            ang = 90+ang
        if w_m < h_m and ang > 0:
            ang = (90-ang)*-1
        if w_m > h_m and ang < 0:
            ang = 90+ang
        
        ang = int(ang)
        
        sp_x = int(256/2)
        pv_x = (x+int(w/2))
        lateral_error = sp_x - pv_x
        #print("Lat err: "+str(lateral_error) + " Angle: "+str(ang))
        
#####
        if (lateral_error >= 2 or lateral_error <= -2):
            timeAcc = timeAcc +1
        else:
            timeAcc=0
        p_term = -kp * lateral_error
        d_term = -kp * kd * np.sin(ang)
        i_term = ki * timeAcc * p_term
    
        angVel = p_term + d_term + i_term
        if (angVel > maxAngVel):
            angVel = maxAngVel
        elif (angVel < -1*maxAngVel):
            angVel = -1*maxAngVel
       
        #print(lateral_error)
        print("P: "+str(p_term)+" I:"+str(i_term)+" D: "+str(d_term) +
              " lateralError: " + str(lateral_error) + " angularError: "
              + str(angVel))
        angVel_no = 100
        
####
        
        

        
    else:
        #print("no c")
        angVel_no -= 0.5
        if (angVel_no <= 30):
            angVel_no=30
        angVel = angVel_no
        #Search for track algorithm
        #gpg.orbit(90, searchRadius)
        #searchRadius += radiusIncrement
    #angVel = controller.calculateAngularSpeed(lateral_error, ang)
        print(angVel)
    wheelDist = 0.125
    wheelRadius = 0.033
    
    
        
    leftMotorSpeed = ((2*linVel) + ((angVel*wheelDist)/(2*wheelRadius)))
    rightMotorSpeed = ((2*linVel) - ((angVel*wheelDist)/(2*wheelRadius)))

    #print("Left speed "+str(leftMotorSpeed)+" Right speed "+str(rightMotorSpeed))
    gpg.set_motor_dps(gpg.MOTOR_LEFT, dps=leftMotorSpeed)
    gpg.set_motor_dps(gpg.MOTOR_RIGHT, dps=rightMotorSpeed)
    cv2.imshow("contours", ocv_img)

    setpoint = 0
    
    stream.truncate(0)
    cv2.waitKey(1)
    #print(test)
    test = test+1
    
    #Controller
    

    #GoPiGo Motors and speed
    #maxAngVel = 30
    
    
    #linVel = 100
    #angVel = controller.calculateAngularSpeed(lateral_error, ang)
##    wheelDist = 0.125
##    wheelRadius = 0.033
##    
##    #if (angVel > maxAngVel):
##     #   angVel = maxAngVel
##    #elif (angVel < -1*maxAngVel):
##     #   angVel = -1*maxAngVel
##    #print(angVel)
##        
##    leftMotorSpeed = ((2*linVel) - ((angVel*wheelDist)/(2*wheelRadius)))
##    rightMotorSpeed = ((2*linVel) + ((angVel*wheelDist)/(2*wheelRadius)))
##
##    #print("Left speed "+str(leftMotorSpeed)+" Right speed "+str(rightMotorSpeed))
##    gpg.set_motor_dps(gpg.MOTOR_LEFT, dps=leftMotorSpeed)
##    gpg.set_motor_dps(gpg.MOTOR_RIGHT, dps=rightMotorSpeed)

    
    
gpg.stop()
cv2.destroyAllWindows()



    
    
    

