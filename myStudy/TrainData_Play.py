# =================================================================================================
# Title       : 트레이닝 & 테스트 코드
# Date        : 2020-05-26
# Description : 1. 실로봇 연결 코드 추가
# =================================================================================================
# 173번째 줄의 COM Port확인(번호) 및 조절할 것, 176번째 줄의 비디오캡쳐도 확인(0인가 1인가..)
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import serial # py -m pip install pyserial로 설치 할 것
from cv2 import cv2
from glob import glob
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageGrab

trainPath = r'D:\AI Study\Xylobot AI Project\RockPaperScissors AI Game\dataset\train\\'
testPath = r'D:\AI Study\Xylobot AI Project\RockPaperScissors AI Game\dataset\test\\'

# =================================================================================================
# Hyperparameter Tuning
# =================================================================================================
numEpoch = 20
batchSize = 10
#learningRate = 0.00001
learningRate = 0.001
dropoutRate = 0.3
inputShare = (50, 50, 3)
numClass = 3

# =================================================================================================
# Constant for Xylobot
# =================================================================================================
OPERATION_READY = 1
OPERATION_BASIC = 3

LED_NAME_OFF = 0
LED_NAME_RED = 1
LED_NAME_ORANGE = 2
LED_NAME_YELLOW = 3
LED_NAME_GREEN = 4
LED_NAME_BLUE = 5
LED_NAME_DARKBULE = 6
LED_NAME_PURPLE = 7
LED_NAME_HIGH_RED = 8
LED_NAME_WHITE = 9


# =================================================================================================
# Function code
# =================================================================================================

def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype=tf.float32)
    return image

def MakeCheckSum(buffers):
    i = 0
    checksum = 0
    while i < 9:
        checksum = checksum + buffers[i]
        i += 1

    return checksum

def MakePacket_Mode(mode):
    buffers = []

    buffers.append(255)
    buffers.append(255)
    buffers.append(65)
    buffers.append(0)
    buffers.append(mode)
    buffers.append(0)
    buffers.append(0)
    buffers.append(0)
    buffers.append(0)
    buffers.append(MakeCheckSum(buffers) % 256)

    return buffers

def MakePacket_LedName(name):
    buffers = []
    
    buffers.append(255)
    buffers.append(255)
    buffers.append(69)
    buffers.append(0)
    buffers.append(name)
    buffers.append(0)
    buffers.append(0)
    buffers.append(0)
    buffers.append(0)
    buffers.append(MakeCheckSum(buffers) % 256)

    return buffers

def MakePacket_TargetPosition(position1, position2, position3):
    buffers = []

    buffers.append(255)
    buffers.append(255)
    buffers.append(173)
    buffers.append(position1 // 256)
    buffers.append(position1 % 256)
    buffers.append(position2 // 256)
    buffers.append(position2 % 256)
    buffers.append(position3 // 256)
    buffers.append(position3 % 256)
    buffers.append(MakeCheckSum(buffers) % 256)

    return buffers
# =================================================================================================
# Main code
# =================================================================================================

# train_dataGenerator = ImageDataGenerator(
#     rescale=1.0/255.0,
#     width_shift_range=0.3,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

train_dataGenerator = ImageDataGenerator(
    rescale=1.0/255.
)

test_dataGenerator = ImageDataGenerator(
    rescale=1./255.
)
train_generator = train_dataGenerator.flow_from_directory(
    trainPath,
    target_size=inputShare[:2],
    batch_size=batchSize,
    color_mode='rgb',
    class_mode='categorical'
)
# validation_generator = test_dataGenerator.flow_from_directory(
#     testPath,
#     target_size=inputShare[:2],
#     batch_size=batchSize,
#     color_mode='rgb',
#     class_mode='categorical'
# )


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShare))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(dropoutRate))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(dropoutRate))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(dropoutRate))
model.add(layers.Dense(numClass, activation='softmax'))

model.compile(optimizer=tf.optimizers.Adam(learningRate),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=numEpoch,
    # validation_data=validation_generator,      # Label된 testset이 없는 경우 생략하는 것이 맞음
    # validation_steps=len(validation_generator) # Label된 testset이 없는 경우 생략하는 것이 맞음
)

ser = serial.Serial('COM9', 115200)         # 컴포트 열기
ser.write(MakePacket_Mode(OPERATION_BASIC)) # 실로봇 기본 상태로 변경

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
tempNum = 0

while True:
    ret, frame = capture.read()
    if ret == True:
        cv2.imshow("VideoFrame", frame)
        keyInput = cv2.waitKey(20)

        save = cv2.imwrite(testPath + "/1.jpg", frame, params= None)
        img = read_image(testPath + "/1.jpg")
        img = tf.image.resize(img, inputShare[:2])

        image = np.array(img)
        #print(image.shape)

        #plt.imshow(image)
        #plt.title('Check the Image and Predict Together!')
        #plt.show()

        testImage = image[tf.newaxis, ...]
        pred = model.predict(testImage)
        #print(pred)
        num = np.argmax(pred)

        tempNum += 1

        if num == 0:    #사람 : [가위]로 판단 --> 로봇 : 바위
            ser.write(MakePacket_TargetPosition(512, 220, 220)) #바위
            #print("사람은 [가위]를 냈네요. ===> 로봇은 [바위]를 냅니다. " + str(tempNum))
            print("사람은 [가위]를 냈네요. ===> 로봇은 [바위]를 냅니다.") 

        elif num == 1:  #사람 : [바위]로 판단 --> 로봇 : 보  
            ser.write(MakePacket_TargetPosition(512, 512, 512)) #보
            #print("사람 : 바위 ===> 로봇 :  보 " + str(tempNum))
            print("사람은 [바위]를 냈네요. ===> 로봇은 [ 보 ]를 냅니다.") 

        elif num == 2:  #사람 : [보]로 판단 --> 로봇 : 가위
            ser.write(MakePacket_TargetPosition(314, 353, 341)) #가위
            #print("사람 :  보  ===> 로봇 : 가위 " + str(tempNum))
            print("사람은 [ 보 ]를 냈네요. ===> 로봇은 [가위]를 냅니다.") 

        if keyInput == ord('q'):
            ser.write(MakePacket_Mode(OPERATION_READY))
            break

    else:
         break
            

time.sleep(1)
capture.release()
ser.close()
cv2.destroyAllWindows()
os._exit(True)