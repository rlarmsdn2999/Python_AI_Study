# AP모드로 설정된 무선카메라를 이용하여 yolo물체검출을 테스트해 보자!
# 모델은 로컬에 저장된 모델 사용 yolov5_openCV2와 yolov5_openCV3를 합친 것

import cv2
import torch
import requests   ####### HTTP를 통한 URI(Uniform Resource Identifier)를 이용하기 위해
from imutils.video import WebcamVideoStream  # Webcam영상 스트림을 위해 import

model = torch.hub.load('C:/Users/office1/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path='yolov5n.pt', source='local')

esp_ip = "http://192.168.4.1"
host = "{}:81/stream".format(esp_ip)       ### Webcam을 AP로 하여 ip어드레스와 port(81) 지정 (영상 수신용)
cam = WebcamVideoStream(src=host).start()    # 비디오 스트림 시작. capture = cv2.VideoCapture(0) 부분에 해당 

while True:
    frame = cam.read()
    frame = cv2.resize(frame, (640*2, 480*2))  # 이미지를 원하는 사이즈로 설정 가능 
    results = model(frame)
    results.render()
    cv2.imshow('Modi Camera Module', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
