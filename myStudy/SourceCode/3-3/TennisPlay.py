# 테니스공을 네트위로 살짝 넘기기 위한 계산용 인공지능
# 입력은 라켓각도, 공을 치는 힘, 네트와의 거리
# 3X3X1의 인공 신경망을 만들어 보자. 
# 입력부에 1을 추가하여 bias대체
# 출력부 활성화 함수로 하이퍼탄젠트함수(tanh function)사용

import numpy as np
from random import random

alpha = 0.3   # 학습률(learning rate)
epoch = 5000

# 가중치의 초기화 (initializing of weight) 
wt = []    # 빈 list를 만들고, 나중에 append를 통해 추가 

for i in range(16):  # 3X3X1에서 총 16개의 가중치값이 필요
    w = np.random.rand()
    wt.append(w)

# sigmoid 활성화 함수
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def tanh(x):
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y

# 입력(input)값과 정답(teaching data)
input_data = np.array([[80,6,10], [45,10,5], [70,8,9], [60,6,8], [60,5,3], [50,5,4], [60,7,7]])
teaching_data = np.array([[-0.3], [1], [0.1], [0.05], [-0.1], [0.5], [0]])

# 입력값과 정답을 통해 학습을 시작
for n in range(1, epoch+1): # 1부터 epoch까지 반복
    for i in range(len(input_data)): 
        x1 = input_data[i][0]/90   # i번째 행의 첫번째 숫자/라켓각도도(최대 90도)
        x2 = input_data[i][1]/10   # i번째 행의 두번째 숫자/힘(최대 10N)
        x3 = input_data[i][2]/12   # i번째 행의 세번째 숫자/네트와의 거리(최대 12m)
        t  = teaching_data[i]      # i번째 행의 숫자 (네트와의 간격)
        ########## 순방향 계산 #########
        u1 = sigmoid(wt[0]*x1 + wt[3]*x2 + wt[6]*x3 + wt[9])
        u2 = sigmoid(wt[1]*x1 + wt[4]*x2 + wt[7]*x3 + wt[10])
        u3 = sigmoid(wt[2]*x1 + wt[5]*x2 + wt[8]*x3 + wt[11])
        y  = tanh(wt[12]*u1 + wt[13]*u2 + wt[14]*u3 + wt[15])
        ######## 역방향 계산 (오차 역전파법) ########
        E = 0.5 * (y - t)**2
        dE_dw_0 = (y-t)*wt[12]*(1+y)*(1-y)*(1-u1)*u1*x1
        dE_dw_1 = (y-t)*wt[13]*(1+y)*(1-y)*(1-u2)*u2*x1
        dE_dw_2 = (y-t)*wt[14]*(1+y)*(1-y)*(1-u3)*u3*x1
        dE_dw_3 = (y-t)*wt[12]*(1+y)*(1-y)*(1-u1)*u1*x2
        dE_dw_4 = (y-t)*wt[13]*(1+y)*(1-y)*(1-u2)*u2*x2
        dE_dw_5 = (y-t)*wt[14]*(1+y)*(1-y)*(1-u3)*u3*x2
        dE_dw_6 = (y-t)*wt[12]*(1+y)*(1-y)*(1-u1)*u1*x3
        dE_dw_7 = (y-t)*wt[13]*(1+y)*(1-y)*(1-u2)*u2*x3
        dE_dw_8 = (y-t)*wt[14]*(1+y)*(1-y)*(1-u3)*u3*x3
        dE_dw_9 = (y-t)*wt[12]*(1+y)*(1-y)*(1-u1)*u1 
        dE_dw_10 = (y-t)*wt[13]*(1+y)*(1-y)*(1-u2)*u2
        dE_dw_11 = (y-t)*wt[14]*(1+y)*(1-y)*(1-u3)*u3
        dE_dw_12 = (y-t)*(1+y)*(1-y)*u1
        dE_dw_13 = (y-t)*(1+y)*(1-y)*u2
        dE_dw_14 = (y-t)*(1+y)*(1-y)*u3
        dE_dw_15 = (y-t)*(1+y)*(1-y)
        ########## 가중치 업데이트 (경사하강법) #########
        wt[0] = wt[0] - alpha * dE_dw_0
        wt[1] = wt[1] - alpha * dE_dw_1
        wt[2] = wt[2] - alpha * dE_dw_2
        wt[3] = wt[3] - alpha * dE_dw_3
        wt[4] = wt[4] - alpha * dE_dw_4
        wt[5] = wt[5] - alpha * dE_dw_5
        wt[6] = wt[6] - alpha * dE_dw_6
        wt[7] = wt[7] - alpha * dE_dw_7
        wt[8] = wt[8] - alpha * dE_dw_8
        wt[9] = wt[9] - alpha * dE_dw_9
        wt[10] = wt[10] - alpha * dE_dw_10
        wt[11] = wt[11] - alpha * dE_dw_11
        wt[12] = wt[12] - alpha * dE_dw_12
        wt[13] = wt[13] - alpha * dE_dw_13
        wt[14] = wt[14] - alpha * dE_dw_14
        wt[15] = wt[15] - alpha * dE_dw_15

    print("{} EPOCH-ERROR: {}".format(n, E))

# Test: 입력값 x에 대하여 본 신경망으로 예측(순방향 계산)
x1 = 87/90          # 라켓의 각도 (많이 세운 상태, 최대90도 중, 80도)
x2 = 10/10          # 공을 치는 힘(가장 큰 힘, 최대 10중, 10N)
x3 = 10/12          # 네트와의 거리 (코트의 뒤쪽, 최대 12중, 10m)
u1 = sigmoid(wt[0]*x1 + wt[3]*x2 + wt[6]*x3 + wt[9])
u2 = sigmoid(wt[1]*x1 + wt[4]*x2 + wt[7]*x3 + wt[10])
u3 = sigmoid(wt[2]*x1 + wt[5]*x2 + wt[8]*x3 + wt[11])
y  = np.tanh(wt[12]*u1 + wt[13]*u2 + wt[14]*u3 + wt[15])
print("공을 쳤습니다. 네트와의 간격은??")
print("각도:{}도, 힘:{}N, 거리:{}m --> 네트와의 간격:{}m".format(x1*90, x2*10, x3*12, y))
if y > 0:
    print("공은 네트위로 넘어갔어요!")
else:
    print("공이 네트에 걸렸네요. ㅠㅠ")
print("")