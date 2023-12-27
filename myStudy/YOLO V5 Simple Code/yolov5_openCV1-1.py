# 카메라로 물체검출을 하고, xyxy안의 다양한 정보를 추출하여 확인해 보자.
# 카메라에 책이 잡히면 멧세지를 표시해 주는 간단한 프로그램 

import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device = 'cpu')
model.conf = 0.3   # confidence를 높이면 검출 감도가 떨어짐. 확실한 것만 잡아냄
# model.classes = [73]  # 73번 label (Book)만 검출하게 됨
# print(model.names)  # 80개의 label명이 표시됨

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    results = model(frame)
    results.render()
    for *box, conf, cls in results.xyxy[0]:  # xyxy[0]안에 있는 다양한 정보를 추출
        # print(cls, *box)
        # print(model.names[int(cls)])       # 몇 번 클래스(레이블명)인가를 표시 
        if model.names[int(cls)] == 'book':  # 책이 보인다면
            print('There are Books')

    cv2.imshow('yolov5', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break