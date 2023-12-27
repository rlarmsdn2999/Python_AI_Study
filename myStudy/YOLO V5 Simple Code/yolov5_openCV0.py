# 카메라와 openCV를 사용하여 직접 물체검출을 하여 보자!

import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device = 'cpu')
model.conf = 0.3   # confidence를 높이면 검출 감도가 떨어짐
# print(model.names)  # 80개의 label명이 표시됨
# model.classes = [73]  # 73번 label (Book)만 검출하게 됨

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    results = model(frame)
    results.render()
    cv2.imshow('yolov5', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break