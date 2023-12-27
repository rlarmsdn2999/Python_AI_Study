# AP모드로 설정된 무선카메라를 이용하여 yolo물체검출을 테스트해 보자!
# AP모드는 인터넷연결이 안되니, 동글을 이용한 SoftAP설정이 필요. 
# model을 인터넷을 통해 가져오지만 않는다면, WiFi를 무선카메라 AP로 설정하여 실행가능  -->yolov5_openCV4.py

import cv2
import torch
import requests   ####### HTTP를 통한 URI(Uniform Resource Identifier)를 이용하기 위해
from imutils.video import WebcamVideoStream  # Webcam영상 스트림을 위해 import
import json
import pandas as pd

model = torch.hub.load('C:/Users/User/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path='C:/Users/User/Desktop/rlarmsdn/python/project/YOLO/bubbles/best(1).pt', source='local')
model.conf = 0.6   # confidence를 높이면 검출 감도가 떨어짐. 확실한 것만 잡아냄

# host = "{}:4747/video".format("http://192.168.0.3")        ### Webcam을 AP로 하여 ip어드레스와 port(81) 지정 (영상 수신용)
# cam = WebcamVideoStream(src=host).start()    # 비디오 스트림 시작. capture = cv2.VideoCapture(0) 부분에 해당

cam = cv2.VideoCapture(0)

def obj_cord(img):   # 검출된 객체들의 좌표와 레이블을 반환 (objects coordination)
    results = model(img)
    results.render()
    return json.loads(results.pandas().xyxy[0].to_json(orient='records'))

bubbleNum = 0
color = (255,0,0)

while True:
    ret, frame = cam.read()
    results = model(frame)
    results.render()
    cv2.line(frame, (0,360), (640,360), (color), thickness=2)
    for val in obj_cord(frame):
        modNm = model.names[val['class']]  # class명 획득
        if modNm == 'bubbles':              # 사람이 검출되었다면
            x_min = val['xmin']                # Bounding Box의 x축 minimum 좌표
            x_max = val['xmax']                # Bounding Box의 x축 maximum 좌표
            y_min = val['ymin']                # Bounding Box의 y축 minimum 좌표
            y_max = val['ymax']                # Bounding Box의 y축 maximum 좌표
            x_cnt = (x_min + (x_max-x_min)/2)/640  # Bounding Box의 x축 중심점 계산
            y_cnt = (y_min + (y_max-y_min)/2)/480  # Bounding Box의 y축 중심점 계산
            # 검출된 물체의 이름과 중심점을 0~1사이의 값으로 표시. 소수점 3째자리까지만 표시
            # print('class: {} || x-center: {:.3f}, y-center: {:.3f}'.format(modNm, x_cnt, y_cnt))
            if y_cnt > 0.75:
                bubbleNum += 1
                print('There are bubbles')
                print('bubble is on the line of the screen : ', bubbleNum)
            else:
                pass      
        # Width(x_max-x_min)*Height(y_max-y_min)로 Bounding Box의 크기 계산: 원근을 대략적으로 알 수 있음
        # print('Size of Object: {:.3f}'.format((x_max-x_min)/640*(y_max-y_min)/480)) 

    cv2.imshow('yolov5', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break