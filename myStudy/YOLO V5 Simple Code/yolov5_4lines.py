# 단 4줄만으로 yolov5 물체검출의 위력을 확인해 보자! 

import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device = 'cpu')
img = 'http://ultralytics.com/images/zidane.jpg'
print(model(img))