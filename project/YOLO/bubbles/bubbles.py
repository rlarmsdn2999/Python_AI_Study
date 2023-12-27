import cv2
import torch
import requests   ####### HTTP를 통한 URI(Uniform Resource Identifier)를 이용하기 위해
from imutils.video import WebcamVideoStream  # Webcam영상 스트림을 위해 import
import numpy as np ## import
import pandas as pd

model = torch.hub.load('C:/Users/User/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path='C:/Users/User/Desktop/rlarmsdn/python/project/YOLO/bubbles/best(1).pt', source='local')

model.conf = 0.6   # confidence를 높이면 검출 감도가 떨어짐. 확실한 것만 잡아냄
# esp_ip = "http://192.168.0.3"
# host = "{}:4747/video".format(esp_ip)       ### Webcam을 AP로 하여 ip어드레스와 port(81) 지정 (영상 수신용)
# cap=WebcamVideoStream(src=host).start() # 비디오 스트림 시작. capture = cv2.VideoCapture(0) 부분에 해당

cap = cv2.VideoCapture(0)
# count=0
# model = torch.hub.load('/usr/local/lib/python3.9/dist-packages/yolov5', 'custom', path,source='local')
b=model.names[0] = 'bubbles'
size=416 ## 초록색 원 인식범위
count=0 ## 카메라 인식 버퍼 값
counter=0 ## 비눗방울 수
color=(0,0,255) ## 카운팅 line 색상
cy1=250 ## 빨간색 기준 선 y축 값, 증가할 수록 아래로 내려감
offset=60 ## 비눗방울과 선 충돌 값
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))  # 이미지를 원하는 사이즈로 설정 가능
    results = model(frame)
    results.render()
    ############################################### 카운트용 line 제작
    count += 1 # 코드 실행 후 count값 증가
    if count % 4 != 0: # "4"의 배수일경우 코드 실행(버퍼값, 중복 카운팅 방지용)
        continue
    cv2.line(frame, (0, 320), (640, 320), (color), 2) # 라인 설정
    results=model(frame,size)
    a=results.pandas().xyxy[0]
    for inedx, row in results.pandas().xyxy[0].iterrows():
        x1=int(row['xmin'])
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['ymax'])
        d=(row['class'])
        if d==0:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            rectx1, recty1=((x1+x2)/2, (y1+y2)/2)
            rectcenter=int(rectx1),int(recty1)
            cx=rectcenter[0]
            cy=rectcenter[1]
            cv2.circle(frame,(cx,cy),30,(0,255,0),-1)
            cv2.putText(frame,str(b),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            if cy<(cy1+offset) and cy>(cy1-offset): #비눗방울 카운팅
                counter+=1
                cv2.line(frame, (79,cy1),(590,cy1),(0,255,0),2)
                print('현재 비눗방울의 수는 총 {}개 입니다.'.format(counter))
                cv2.putText(frame,str(counter),(x2,y2),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow("IMG",frame) #영상 출력
    ###############################################
    if cv2.waitKey(1) & 0xff == ord('q'): #q키 누를경우 종료
        break
print('비눗방울 합계 : {}개.'.format(counter))
cap.release()
cv2.destroyAllWindows()