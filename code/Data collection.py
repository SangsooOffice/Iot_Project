import cv2
import RPi.GPIO as GPIO
import time
import os
import numpy as np
PWMA = 18
AIN1 = 22
AIN2 = 27
PWMB = 23
BIN1 = 25
BIN2 = 24
def motor_back(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,False) #AIN2
    GPIO.output(AIN1,True)  #AIN1
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,False) #BIN2
    GPIO.output(BIN1,True)  #BIN1
def motor_go(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)  #AIN2
    GPIO.output(AIN1,False) #AIN1
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)  #BIN2
    GPIO.output(BIN1,False) #BIN1def motor_stop():

def motor_right(speed,subspeed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)  #AIN2
    GPIO.output(AIN1,False) #AIN1
    R_Motor.ChangeDutyCycle(subspeed)
    GPIO.output(BIN2,False) #BIN2
    GPIO.output(BIN1,True)  #BIN1
def motor_left(speed,subspeed):
    L_Motor.ChangeDutyCycle(subspeed)
    GPIO.output(AIN2,False) #AIN2
    GPIO.output(AIN1,True)  #AIN1
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)  #BIN2
    GPIO.output(BIN1,False) #BIN1
def motor_stop():
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN1,0) #AIN1
    GPIO.output(AIN2,0)  #AIN2
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,0)  #BIN2
GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(PWMA,GPIO.OUT)
GPIO.setup(BIN1,GPIO.OUT)
GPIO.setup(BIN2,GPIO.OUT)
GPIO.setup(PWMB,GPIO.OUT)
L_Motor= GPIO.PWM(PWMA,500)
L_Motor.start(0)
R_Motor = GPIO.PWM(PWMB,500)
R_Motor.start(0)
speedSet = 40

def main():
    camera = cv2.VideoCapture(-1)
    camera.set(3, 640)
    camera.set(4, 480)
    num = 104

    path = f"/media/aaaa/B8CD-43101/img/{num}"
    filepath = path+f"/train_{num}"



    i = 0

    carState = "stop"

    while( camera.isOpened() ):
        keyValue = cv2.waitKey(10)
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("go")
            carState = "go"
            motor_go(30)
        elif keyValue == 84:
            print("stop")
            carState = "stop"
            motor_stop()
        elif keyValue == 97:
            print("left_2")
            carState = "left_2"
            motor_left(speedSet,0)
        elif keyValue == 100:
            print("right_2")
            carState = "right_2"
            motor_right(speedSet,0)
        elif keyValue == 81:
            print("left")
            carState = "left"
            motor_left(speedSet,speedSet)
        elif keyValue == 83:
            print("right")
            carState = "right"
            motor_right(speedSet,speedSet)
        _, image = camera.read()
        image = cv2.flip(image,-1)
        height, _, _ = image.shape 


        save_image = image[int(height/(14/10)):,:,:]
        lab = cv2.cvtColor(save_image, cv2.COLOR_BGR2Lab)
        

        lower_white = np.array([0, 0, 0])
        upper_white = np.array([255, 255, 255])
        

        mask = cv2.inRange(lab, lower_white, upper_white)
        

        white_line_image = cv2.bitwise_and(save_image, save_image, mask=mask)
  
        

        gray = cv2.cvtColor(white_line_image, cv2.COLOR_BGR2GRAY)
   
        

        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        

        kernel = np.ones((9,9),np.uint8)
        opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        

        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        
        binary_image = cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11)

        
        edges = cv2.Canny(binary_image, 50, 150)


        save_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

       
        

        save_image = cv2.resize(save_image, (220, 80))

        cv2.imshow('Detected White Lines', save_image)

        if carState == "left":
            cv2.imwrite("%s_%05d_%03d_%03d.png" % (filepath, i,1, 0), save_image)
            i += 1
        elif carState == "left_2":
            cv2.imwrite("%s_%05d_%03d_%03d.png" % (filepath, i,1, 45), save_image)
            i += 1
        elif carState == "right_2":
            cv2.imwrite("%s_%05d_%03d_%03d.png" % (filepath, i,1, 135), save_image)
            i += 1
        elif carState == "right":
            cv2.imwrite("%s_%05d_%03d_%03d.png" % (filepath, i,1, 180), save_image)
            i += 1

        elif carState == "go":
            cv2.imwrite("%s_%05d_%03d_%03d.png" % (filepath, i,1, 90), save_image)
            i += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
     main()
     GPIO.cleanup()


