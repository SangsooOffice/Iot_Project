import cv2
import RPi.GPIO as GPIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
from tensorflow.keras.optimizers import Adam
import time
from multiprocessing import Process, Manager

class Run:
    
    def __init__(self):
        self.classNames = {0: 'background',
                1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
                13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
                18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
                32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
                37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
                75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
                80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
        PWMA = 18
        self.AIN1 = 22
        self.AIN2 = 27
        PWMB = 23
        self.BIN1 = 25
        self.BIN2 = 24
        GPIO.setwarnings(False) 
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.AIN2,GPIO.OUT)
        GPIO.setup(self.AIN1,GPIO.OUT)
        GPIO.setup(PWMA,GPIO.OUT)
        GPIO.setup(self.BIN1,GPIO.OUT)
        GPIO.setup(self.BIN2,GPIO.OUT)
        GPIO.setup(PWMB,GPIO.OUT)
        self.L_Motor= GPIO.PWM(PWMA,500)
        self.L_Motor.start(0)
        self.R_Motor = GPIO.PWM(PWMB,500)
        self.R_Motor.start(0)
        self.speedSet = 40

        self.main()


    def motor_back(self,speed):
        self.L_Motor.ChangeDutyCycle(speed)
        GPIO.output(self.AIN2,False) #self.AIN2
        GPIO.output(self.AIN1,True)  #self.AIN1
        self.R_Motor.ChangeDutyCycle(speed)
        GPIO.output(self.BIN2,False) #self.BIN2
        GPIO.output(self.BIN1,True)  #self.BIN1
    def motor_go(self,speed,speed2):
        self.L_Motor.ChangeDutyCycle(speed)
        GPIO.output(self.AIN1,0) #self.AIN1
        GPIO.output(self.AIN2,1)  #self.AIN2
        self.R_Motor.ChangeDutyCycle(speed2)
        GPIO.output(self.BIN1,0)
        GPIO.output(self.BIN2,1)  #self.BIN2
    
    def motor_stop(self):
        self.L_Motor.ChangeDutyCycle(0)
        GPIO.output(self.AIN1,0) #self.AIN1
        GPIO.output(self.AIN2,0)  #self.AIN2
        self.R_Motor.ChangeDutyCycle(0)
        GPIO.output(self.BIN1,0)
        GPIO.output(self.BIN2,0)  #self.BIN2
    def motor_right(self,speed,speed_right):
        self.L_Motor.ChangeDutyCycle(speed)
        GPIO.output(self.AIN1,0) #self.AIN1
        GPIO.output(self.AIN2,1)  #self.AIN2
        self.R_Motor.ChangeDutyCycle(speed_right)
        GPIO.output(self.BIN1,1)
        GPIO.output(self.BIN2,0)  #self.BIN2
    def motor_left(self,speed,speed_turn):
        self.L_Motor.ChangeDutyCycle(speed_turn)
        GPIO.output(self.AIN1,1) #self.AIN1
        GPIO.output(self.AIN2,0)  #self.AIN2
        self.R_Motor.ChangeDutyCycle(speed)
        GPIO.output(self.BIN1,0)
        GPIO.output(self.BIN2,1)  #self.BIN2
    def img_preprocess(self,image):

        height, _, _ = image.shape
        image = image[int(height/(14/10)):,:,:]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        lower_white = np.array([0, 0, 0])
        upper_white = np.array([255, 255, 255])

        mask = cv2.inRange(lab, lower_white, upper_white)

        white_line_image = cv2.bitwise_and(image, image, mask=mask)


        gray = cv2.cvtColor(white_line_image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (9, 9), 0)


        kernel = np.ones((9,9),np.uint8)
        opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        binary_image = cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11)


        edges = cv2.Canny(binary_image, 50, 150)

        save_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        save_image = cv2.resize(save_image, (220, 80))
        save_image = save_image / 255



        return save_image
    
    def my_callback(self):
        while True:
            input_str = int(input("input :"))
            if input_str == 1:
                print("go")
                self.carstate = "go"
            elif input_str == 2:
                print("stop")
                self.carstate = "stop"
    def id_class_name(self,class_id, classes):
     for key, value in classes.items():
          if class_id == key:
               return value

    def TThread(self):
        global image
        global image_ok
        image = None

        model_o = cv2.dnn.readNetFromTensorflow('/home/aaaa/AI_CAR/OpencvDnn/models/frozen_inference_graph.pb', '/home/aaaa/AI_CAR/OpencvDnn/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        while True:
            if image is not None and image_ok==1:
                print("obstacle detact")
                height, width, _ = image.shape
                center_x = int(width / 2) 

            
                crop_width = 200 

                left_x = max(0, center_x - crop_width) 
                right_x = min(width, center_x + crop_width) 

                cropped_image = image[:, left_x:right_x, :]
                model_o.setInput(cv2.dnn.blobFromImage(cropped_image, size=(150,150), swapRB=True))
                output = model_o.forward()
                for detection in output[0, 0, :, :]:
                    confidence = detection[2]
                    if confidence > 0.5:
                        class_id = detection[1]
                        
                        
                        class_name = class_name=self.id_class_name(class_id,self.classNames)
                        print(str(str(class_id) + " " + str(detection[2]) + " " + class_name))
                        if class_name == "bottle":
                            self.carstate = "stop"

    def main(self):
        global image
        global image_ok
        image_ok = 0



        camera = cv2.VideoCapture(-1)
        camera.set(3, 640)
        camera.set(4, 480)
        self.carstate = "stop"

        kthread = threading.Thread(target=self.my_callback)
        kthread.start()
        eth = threading.Thread(target=self.TThread)
        eth.start()
        model_path = '/home/aaaa/model/model22/1_lane_2.tflite'
        
        # tflite 불러오기
        tflite = tf.lite.Interpreter(model_path)
        tflite.allocate_tensors()


        counter = 0
        start_time = time.time()
        try:
            while( camera.isOpened()):
                
                image_ok = 0
                _, image = camera.read()
                if image is not None:
                    image = cv2.flip(image,-1)
                    image_ok = 1


                    preprocessed = self.img_preprocess(image)

                    X = np.asarray([preprocessed]).astype('float32')

                   # tflite 예측
                    inp = tflite.get_input_details()
                    out = tflite.get_output_details()

                    tflite.set_tensor(inp[0]['index'], X)

                    tflite.invoke()

           
                    self.steering_angle = int(tflite.get_tensor(out[0]['index'])[0])
                    print("pre :",self.steering_angle)
                    
                    if self.carstate == "go":
                        print("gogo")
                        if self.steering_angle >= 85 and self.steering_angle <= 95:
                            print("go")
                            self.motor_go(100,100)
                        elif self.steering_angle > 85:
                                self.motor_right(self.speedSet,self.speedSet)
                        elif self.steering_angle < 95:
                                self.motor_left(self.speedSet,self.speedSet)
                        elif self.steering_angle < 0:
                            self.motor_stop()
                    elif self.carstate == "stop":
                        self.motor_stop()
                    


                    counter += 1
                    if counter == 5:
                        end_time = time.time()
                        time_taken = end_time - start_time
                        print(f"5번 예측당 걸린 시간: {time_taken} seconds")
                        counter = 0
                        start_time = time.time()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            pass
                

        
        
if __name__ == '__main__':
    Run()
    GPIO.cleanup()
