# 설정된 사진파일의 이미지에서 yolov5로 각각의 물체를 검출한 결과를 확인해 보자!
# runs/exp2 폴더에서 crop된 파트(bus, car, person, truck ...)를 확인해 보자! 

import torch
from PIL import Image  # py -m pip install pillow
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device = 'cpu')  
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device = 'cpu')  # 보다 정밀한 모델 
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', device = 'cpu') # 가장 정밀한 모델

img = Image.open('road_junction.jpg')
results = model(img)
results.crop(save = True)
results.show()