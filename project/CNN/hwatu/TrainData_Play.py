import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2
from glob import glob
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageGrab
import winsound

trainPath = r'C:\Users\User\Desktop\rlarmsdn\python\Project 2023.8.28\RockPaperScissors AI Game\dataset\train\\'
testPath = r'C:\Users\User\Desktop\rlarmsdn\python\Project 2023.8.28\RockPaperScissors AI Game\dataset\test\\'

numEpoch = 2
batchSize = 10
learningRate = 0.001
dropoutRate = 0.3
inputShare = (50, 50, 3)
numClass = 4
frequency = 1000  # 비프 소리의 주파수 (Hz)
duration = 1000   # 비프 소리의 지속 시간 (밀리초)


def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype=tf.float32)
    return image

train_dataGenerator = ImageDataGenerator(
    rescale=1.0/255.0,
    width_shift_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True
)

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
)


capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
tempNum = 0

folderPath = r"C:\Users\User\Desktop\rlarmsdn\python\Project 2023.8.28\RockPaperScissors AI Game\img"       # 이미지가 들어있는 폴더 경로 지정
myList = os.listdir(folderPath)   # folderPath안의 이미지를 myList에 저장

overlayList = []                  # 웹캠 영상위에 겹쳐서 표시하기 위한 이미지
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

totalFingers = 0  # 펼쳐진 손가락의 갯수, 0부터 5까지 테스트 해볼 것

while True:
    ret, frame = capture.read()
    success, img2 = capture.read()       ###### 영상 획득 기본 코드 ######
    if ret == True:
        cv2.imshow("VideoFrame", frame)
        keyInput = cv2.waitKey(20)

        save = cv2.imwrite(testPath + "/1.jpg", frame, params= None)
        img = read_image(testPath + "/1.jpg")
        img = tf.image.resize(img, inputShare[:2])

        image = np.array(img)

        testImage = image[tf.newaxis, ...]
        pred = model.predict(testImage)
        num = np.argmax(pred)

        tempNum += 1
        if num == 0:
            h, w, c = overlayList[totalFingers-4].shape  # 오버레이 이미지의 높이, 폭, 채널
            img2[0:h, 0:w] = overlayList[totalFingers-4]  # 오버레이 이미지를 왼쪽 구석에 표시
            cv2.imshow("Image", img2)        ###### 영상 획득 기본 코드 ######
            print("사구 재경기") 

        elif num == 1:
            h, w, c = overlayList[totalFingers-3].shape  # 오버레이 이미지의 높이, 폭, 채널
            img2[0:h, 0:w] = overlayList[totalFingers-3]  # 오버레이 이미지를 왼쪽 구석에 표시
            cv2.imshow("Image", img2)        ###### 영상 획득 기본 코드 ######
            winsound.Beep(frequency, duration) # 비프음 재생
            print("38광땡") 

        elif num == 2:
            h, w, c = overlayList[totalFingers-2].shape  # 오버레이 이미지의 높이, 폭, 채널
            img2[0:h, 0:w] = overlayList[totalFingers-2]  # 오버레이 이미지를 왼쪽 구석에 표시
            cv2.imshow("Image", img2)        ###### 영상 획득 기본 코드 ######  
            print("장땡")
        
        elif num == 3:
            h, w, c = overlayList[totalFingers-1].shape  # 오버레이 이미지의 높이, 폭, 채널
            img2[0:h, 0:w] = overlayList[totalFingers-1]  # 오버레이 이미지를 왼쪽 구석에 표시
            cv2.imshow("Image", img2)        ###### 영상 획득 기본 코드 ###### 
            print("암행어사")

        if keyInput == ord('q'):
            break

    else:
         break
            
time.sleep(1)
capture.release()
cv2.destroyAllWindows()
os._exit(True)