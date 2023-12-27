import gym
import numpy as np
import random

env = gym.make('CartPole-v1')  # CartPole gyn환경을 만들어 env에 저장
goal_steps = 500               # 최대 몇번까지로 동작제한

done = False
while not done:                # 막대가 서있는 동안은 goal_steps까지 계속 실행
  obs = env.reset()            # 환경 초기화
  for i in range(goal_steps):  # 매 프레임마다 반복, 최대 goal_steps까지
    obs, reward, done, info = env.step(random.randrange(0, 2))
    # obs는 현재상황(카트 위치, 카트 속도, 막대 각도, 막대끝 속도)
    # reward는 보상, 막대가 떨어지지 않으면 1이 주어짐
    # done는 막대가 떨어짐을 의미. 막대의 각도가 12도 이상 기울거나 카트가 2.4칸 이상 움직인 경우
    # if done: break    # 막대가 떨어지면 끝냄
    env.render()      # 환경을 Display
env.close()