import gym
import numpy as np
import random

env = gym.make('CartPole-v1')  # CartPole gyn환경을 만들어 env에 저장
goal_steps = 100               # 최대 몇번까지로 동작제한

done = False
while not done:                # 막대가 서있는 동안은 goal_steps까지 계속 실행
  obs = env.reset()            # 환경 초기화
  for i in range(goal_steps):  # 매 프레임마다 반복, 최대 goal_steps까지
    obs, reward, done, info = env.step(random.randrange(0, 2))
    # env.render()      # 환경을 Display
env.close()
# reward값을 최대화하여 오랜시간 버티는 신경망 제작
# N:CartPole을 실행해 볼 횟수, K: 그중에 뽑을 데이터 갯수, f: Cart를 어떻게 동작시킬지 정하는 함수
def data_preparation(N, K, f, render=False):
  game_data = []
  for i in range(N):      # N번동안 실행해보고 현재상황과 판단을 저장
    score = 0
    game_steps = []
    obs = env.reset()
    for step in range(goal_steps):
      if render: env.render()
      action = f(obs)
      game_steps.append((obs, action))
      obs, reward, done, _ = env.step(action)
      score += reward
      if done:
        break
    game_data.append((score, game_steps))
  
  game_data.sort(key=lambda s:-s[0])
  
  training_set = []      # N번동안 살행한 것중 상위 K개를 모아 training set을 만듦
  for i in range(K):     # training set은 관측을 담은 길이4의 배열과 행동을 담은 길이2의 배열로 구성 
    for step in game_data[i][1]:
      if step[1] == 0:   # 행동은 [p, q]가 p의 확률로 0, q의 확률로 1을 실행, 즉 [1,0]또는 [0,1]이 됨
        training_set.append((step[0], [1, 0]))
      else:
        training_set.append((step[0], [0, 1]))

  print("{0}/{1}th score: {2}".format(K, N, game_data[K-1][0]))
  if render:
    for i in game_data:
      print("Score: {0}".format(i[0]))
  
  return training_set
# 처음에는 랜덤하게 실행한 후, 나중에 상위 K개만을 모아서 사용
N = 10000
K = 50
training_data = data_preparation(N, K, lambda s: random.randrange(0, 2))

# 신경망 구축: input이 4개, output이 2개
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam  
def build_model():
  model = Sequential()
  model.add(Dense(128, input_dim=4, activation='relu'))
  model.add(Dense(52, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='mse', optimizer=Adam())
  return model
def train_model(model, training_set):
  X = np.array([i[0] for i in training_set]).reshape(-1, 4)
  y = np.array([i[1] for i in training_set]).reshape(-1, 2)
  model.fit(X, y, epochs=10)

# N번 실행중 상위K개만을 가지고 옴
if __name__ == '__main__':
  model = build_model()
  training_data = data_preparation(N, K, lambda s: random.randrange(0, 2))
  train_model(model, training_data)
  
  def predictor(s):
    return np.random.choice([0, 1], p=model.predict(s.reshape(-1, 4))[0] )
    
  data_preparation(100, 100, predictor, True)
# if __name__ == '__main__':
#   N = 1000
#   K = 50
#   self_play_count = 10
#   model = build_model()
#   training_data = data_preparation(N, K, lambda s: random.randrange(0, 2))
#   train_model(model, training_data)
  
#   def predictor(s):
#     return np.random.choice([0, 1], p=model.predict(s.reshape(-1, 4))[0] )

#   for i in range(self_play_count):
#     K = (N//9 + K)//2
#     training_data = data_preparation(N, K, predictor)
#     train_model(model, training_data)
  
#   data_preparation(100, 100, predictor, True)