# 카메라로 물체검출을 하고, xyxy[0]안의 다양한 정보를 추출하여 확인해 보자.
# pandas의 값(클래스, 좌표값 등)들을 json포맷으로 변환 -> 필요값만 가지고 옴
# 검출된 물체의 이름과 중심점 좌표를 읽어오자!

import cv2
import torch
import pandas
import json

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device = 'cpu')
cam = cv2.VideoCapture(0)

def obj_cord(img):   # 검출된 객체들의 좌표와 레이블을 반환 (objects coordination)
    results = model(img)
    results.render()
    return json.loads(results.pandas().xyxy[0].to_json(orient='records'))

while True:
    ret, frame = cam.read()
    for val in obj_cord(frame):
        modNm = model.names[val['class']]  # class명 획득
        x_min = val['xmin']                # Bounding Box의 x축 minimum 좌표
        x_max = val['xmax']                # Bounding Box의 x축 maximum 좌표
        y_min = val['ymin']                # Bounding Box의 y축 minimum 좌표
        y_max = val['ymax']                # Bounding Box의 y축 maximum 좌표
        x_cnt = (x_min + (x_max-x_min)/2)/640  # Bounding Box의 x축 중심점 계산
        y_cnt = (y_min + (y_max-y_min)/2)/480  # Bounding Box의 y축 중심점 계산
        # 검출된 물체의 이름과 중심점을 0~1사이의 값으로 표시. 소수점 3째자리까지만 표시
        print('class: {} || x-center: {:.3f}, y-center: {:.3f}'.format(modNm, x_cnt, y_cnt))
        
        # Width(x_max-x_min)*Height(y_max-y_min)로 Bounding Box의 크기 계산: 원근을 대략적으로 알 수 있음
        # print('Size of Object: {:.3f}'.format((x_max-x_min)/640*(y_max-y_min)/480)) 

    cv2.imshow('yolov5', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break