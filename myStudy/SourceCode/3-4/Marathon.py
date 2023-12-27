# 마라톤에서 좋은 기록을 남기기위한 예측 프로그램
# 초기속도에 따른 최종기록을 예측
# 입력은 처음 15km구간 평균속도, 출력은 다음구간의 속도감소와 최종기록 
# 1X3X3의 인공 신경망을 만들어 보자. 
# 입력부에 1을 추가하여 bias대체
# 시그모이드 활성화 사용. 간단한 정규화로 데이터 입력

import numpy as np
from random import random

alpha = 0.3   # 학습률(learning rate)
epoch = 5000

# 가중치의 초기화 (initializing of weight) 
wt = []    # 빈 list를 만들고, 나중에 append를 통해 추가 

for i in range(18):  # 1X3X3에서 총 18개의 가중치값이 필요
    w = np.random.rand()
    wt.append(w)

# sigmoid 활성화 함수
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# 입력(input)값과 정답(teaching data)
input_data = np.array([[25], [30], [20], [17], [27]])
teaching_data = np.array([[6,6,2.31], [9,12,2.54], [3,3,2.49], [0,3,2.62], [6,9,2.27]])

# 입력값과 정답을 통해 학습을 시작
for n in range(1, epoch+1): # 1부터 epoch까지 반복
    for i in range(len(input_data)): 
        x = input_data[i]*3/10      # i번째 입력 데이터/초기 속도(곱하기3/10으로 정규화)
        t1 = teaching_data[i][0]/15 # i번째 행의 첫번째 숫자/첫번째 구간 속도감소(15로 나누어 정규화)
        t2 = teaching_data[i][1]/15 # i번째 행의 두번째 숫자/두번째 구간 속도감소(15로 나누어 정규화)
        t3 = teaching_data[i][2]-2  # i번째 행의 세번째 숫자/최종기록(빼기2로 정규화)
        ########## 순방향 계산 #########
        u1 = sigmoid(wt[0]*x + wt[3])
        u2 = sigmoid(wt[1]*x + wt[4])
        u3 = sigmoid(wt[2]*x + wt[5])
        y1 = sigmoid(wt[6]*u1 + wt[9]*u2 + wt[12]*u3 + wt[15])
        y2 = sigmoid(wt[7]*u1 + wt[10]*u2 + wt[13]*u3 + wt[16])
        y3 = sigmoid(wt[8]*u1 + wt[11]*u2 + wt[14]*u3 + wt[17])
        ######## 역방향 계산 (오차 역전파법) ########
        E = 0.5*(y1 - t1)**2 + 0.5*(y2 - t2)**2 + 0.5*(y3 - t3)**2
        dE_dw_0 = (y1-t1)*(1-y1)*y1*wt[6]*(1-u1)*u1*x+(y2-t2)*(1-y2)*y2*wt[7]*(1-u1)*u1*x+(y3-t3)*(1-y3)*y3*wt[8]*(1-u1)*u1*x
        dE_dw_1 = (y1-t1)*(1-y1)*y1*wt[9]*(1-u2)*u2*x+(y2-t2)*(1-y2)*y2*wt[10]*(1-u2)*u2*x+(y3-t3)*(1-y3)*y3*wt[11]*(1-u2)*u2*x
        dE_dw_2 = (y1-t1)*(1-y1)*y1*wt[12]*(1-u3)*u3*x+(y2-t2)*(1-y2)*y2*wt[13]*(1-u3)*u3*x+(y3-t3)*(1-y3)*y3*wt[14]*(1-u3)*u3*x
        dE_dw_3 = (y1-t1)*(1-y1)*y1*wt[6]*(1-u1)*u1+(y2-t2)*(1-y2)*y2*wt[7]*(1-u1)*u1+(y3-t3)*(1-y3)*y3*wt[8]*(1-u1)*u1
        dE_dw_4 = (y1-t1)*(1-y1)*y1*wt[9]*(1-u2)*u2+(y2-t2)*(1-y2)*y2*wt[10]*(1-u2)*u2+(y3-t3)*(1-y3)*y3*wt[11]*(1-u2)*u2
        dE_dw_5 = (y1-t1)*(1-y1)*y1*wt[12]*(1-u3)*u3+(y2-t2)*(1-y2)*y2*wt[13]*(1-u3)*u3+(y3-t3)*(1-y3)*y3*wt[14]*(1-u3)*u3
        dE_dw_6 = (y1-t1)*(1-y1)*y1*u1
        dE_dw_7 = (y2-t2)*(1-y2)*y2*u1
        dE_dw_8 = (y3-t3)*(1-y3)*y3*u1
        dE_dw_9 = (y1-t1)*(1-y1)*y1*u2
        dE_dw_10 = (y2-t2)*(1-y2)*y2*u2
        dE_dw_11 = (y3-t3)*(1-y3)*y3*u2
        dE_dw_12 = (y1-t1)*(1-y1)*y1*u3
        dE_dw_13 = (y2-t2)*(1-y2)*y2*u3
        dE_dw_14 = (y3-t3)*(1-y3)*y3*u3
        dE_dw_15 = (y1-t1)*(1-y1)*y1
        dE_dw_16 = (y2-t2)*(1-y2)*y2
        dE_dw_17 = (y1-t3)*(1-y3)*y3
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
        wt[16] = wt[16] - alpha * dE_dw_16
        wt[17] = wt[17] - alpha * dE_dw_17

    print("{} EPOCH-ERROR: {}".format(n, E))

# Test: 입력값 x에 대하여 본 신경망으로 예측(순방향 계산)
x = 28 *3/10          # 처음 15km구간의 평균속도(3/10을 곱해서 정규화)
u1 = sigmoid(wt[0]*x + wt[3])
u2 = sigmoid(wt[1]*x + wt[4])
u3 = sigmoid(wt[2]*x + wt[5])
y1 = sigmoid(wt[6]*u1 + wt[9]*u2 + wt[12]*u3 + wt[15])
y2 = sigmoid(wt[7]*u1 + wt[10]*u2 + wt[13]*u3 + wt[16])
y3 = sigmoid(wt[8]*u1 + wt[11]*u2 + wt[14]*u3 + wt[17])
print("이 선수, 첫구간 15km 평균 속도가 시속 {}km 이네요.".format(x*10/3))
print("다음 15km 구간에서 속도가 {}km/h 만큼 떨어졌어요. ".format(y1*15))
print("저런, 마지막 12km 구간에서 속도가 {}km/h 만큼 떨어졌어요. ".format(y2*15))
print("아아! 최종 기록은 2시간 {}분 입니다!!".format(y3*60))
print("")