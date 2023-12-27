import os
import numpy as np
import tensorflow as tf
import cv2
from keras import models
import winsound
import time
from imutils.video import WebcamVideoStream  # Webcam영상 스트림을 위해 import
# py -m pip install imutils
import winsound

testPath = r'C:\Users\User\Desktop\RPS Game\datasets\test'
inputShape = (50, 50, 3)
numClass = 4 

# 이미지 읽기 함수
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype=tf.float32)
    return image

def sound(frq, dur):
    winsound.Beep(frq, dur)

# 모델 불러오기
model = models.load_model('my_model.h5')


# Webcam을 host로 하여 ip어드레스와 port:4747지정
# DroidCam App상에 표시된 IP주소와 포트번호를 입력 
host = "{}:4747/video".format("http://192.168.0.43") 
capture = WebcamVideoStream(src=host).start()    # 비디오 스트림 시작. capture = cv2.VideoCapture(0) 부분에 해당

# 검출 간격 설정
detection_interval = 5  # 2초에 한 번 검출

last_detection_time = time.time()

while True:
    frame = capture.read()
    if cv2.waitKey(1) == ord('q'):
        break
    else:
        cv2.imshow("VideoFrame", frame)

        save = cv2.imwrite(testPath + "/temp.jpg", frame, params= None)
        img = read_image(testPath + "/temp.jpg")
        img = tf.image.resize(img, inputShape[:2])

        image = np.array(img)
        # print(image.shape)
        # plt.imshow(image)
        # plt.title('Check the Image and Predict Together!')
        # plt.show()

        testImage = image[tf.newaxis, ...]
        pred = model.predict(testImage)
        #print(pred)
        num = np.argmax(pred)

        if num == 0:    
            print("현관입니다.") 
            sound(1000, 50)

        elif num == 1:  
            print("계단입니다.") 
            sound(500, 70)

        elif num == 2: 
            print("교실입니다.") 
            sound(1500, 50)

        elif num == 3:  
            print("쉼터입니다.")
            sound(2000, 50) 
        else:
            pass
            
cv2.destroyAllWindows()
os._exit(True)
