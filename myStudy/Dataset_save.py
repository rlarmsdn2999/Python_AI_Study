# 가위, 바위, 보 훈련이미지 만들기 // 사진의 크기, 이름 상관없음
# 파일 실행후 카메라를 향해 가위모양을 하고 숫자키 1을 누르면 폴더1에 가위 사진저장, 2(바위)는 폴더2, 3(보)은 폴더3
import os
from glob import glob
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import cv2

# Local Path로 변경해 주어야 함
train_dir =r'D:\AI Study\Xylobot AI Project\RockPaperScissors AI Game\dataset\train\\'
train_dir1 = train_dir + '1/'
train_dir2 = train_dir + '2/'
train_dir3 = train_dir + '3/'
data1_Num = 0   #1폴더에 저장되는 파일이름(숫자의 증분으로 표시)
data2_Num = 0   #2폴더에 저장되는 파일이름(숫자의 증분으로 표시)
data3_Num = 0   #3폴더에 저장되는 파일이름(숫자의 증분으로 표시)

# Cam에서 스트림으로 이미지 표시하기
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame=capture.read()
    if ret == True:
        cv2.imshow("VideoFrame", frame)

        keyinput = cv2.waitKey(33)  #33msec 간격(30Hz)으로 키 입력을 받음(영상을 리프레시 함)

        # 숫자키 1,2,3키를 누르면 각각의 폴더에 저장함 (q키를 누르면 캡쳐를 종료함)
        if keyinput == ord('q'): 
            break
        
        elif keyinput == ord('1'): 
            data1_Num += 1
            save = cv2.imwrite(train_dir1 + str(data1_Num) + ".jpg", frame, params= None)
            print('1키가 눌렸습니다. 이 영상을 [가위]에 저장합니다({}).', data1_Num)

        elif keyinput == ord('2'): 
            data2_Num += 1
            save = cv2.imwrite(train_dir2 + str(data2_Num) + ".jpg", frame, params= None)
            print('2키가 눌렸습니다. 이 영상을 [바위]에 저장합니다({}).', data2_Num)

        elif keyinput == ord('3'): 
            data3_Num += 1
            save = cv2.imwrite(train_dir3 + str(data3_Num) + ".jpg", frame, params= None)
            print('3키가 눌렸습니다. 이 영상을 [보]에 저장합니다({}).', data3_Num)
    else:
        break

capture.release()
cv2.destroyAllWindows()
