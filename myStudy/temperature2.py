import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler   # py -m pip install sklearn
from sklearn.metrics import mean_squared_error

# 데이터 전처리
n_features = 4  # 특성의 개수 (4개의 열이 있다고 가정)
n_steps = 7

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)-n_steps):
        X.append(sequence[i:i+n_steps, :])  # 모든 열들을 입력으로 사용
        y.append(sequence[i+n_steps, -1])  # 마지막 열을 라벨로 사용
    return np.array(X), np.array(y)

# CSV 파일에서 데이터 불러오기
data = pd.read_csv('temperature2.csv')

# 날짜 열은 예측에 영향을 미치지 않으므로 삭제
data = data.drop(columns=['날짜'])

# 최고기온 값을 0과 1 사이의 값으로 스케일링
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

n_steps = 7  # 7일 이전의 최고기온을 기반으로 다음 날의 최고기온을 예측
X, y = split_sequence(data_scaled, n_steps)

n_train = int(0.8 * len(X))  # 전체 데이터 중 80%를 훈련 데이터로 사용

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)

y_pred = y_pred.reshape(-1, n_features)
y_pred = scaler.inverse_transform(y_pred)

y_test = y_test.reshape(-1, n_features)
y_test = scaler.inverse_transform(y_test)

test_dates = data.iloc[n_train + n_steps:].index

# 예측 결과 시각화
plt.figure(figsize=(12, 6))

# Adjust the test_dates to have the same length as y_test
test_dates = test_dates[-len(y_test):]

plt.plot(test_dates, y_test[:, -1], label='Actual Max Temperature', color='blue')
plt.plot(test_dates, y_pred[:, -1], label='Predicted Max Temperature', color='red')
plt.xlabel('Date')
plt.ylabel('Max Temperature (°C)')
plt.title('Actual vs. Predicted Max Temperature')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()