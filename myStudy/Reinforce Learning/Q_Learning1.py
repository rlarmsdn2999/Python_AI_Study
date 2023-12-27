# Q Learning는 모델을 사용하지 않는 기계학습임. 같은 알고리즘을 다양하게 적용이 가능
# 특정 환경에서 상태(states)와 행동(actions)으로 분류
# 상태는 환경에서 취득한 관측과 샘플링, 행동은 관측에 기반하여 에이전트가 취한 선택
import gym      # OpenAI GYM을 임포트 py -m pip install gym

env = gym.make("MountainCar-v0")   # 환경을 실행
env.reset()                        # 환경을 초기화 

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

done = False
while not done:
    action = 2  # always go right!  0은 왼쪽으로 push, 1은 정지, 2는 오른쪽으로 push
    new_state, reward, done, _=env.step(action)
    # print(new_state)  # 첫번째는 위치, 두번째는 속도
    env.step(action)    # gym 환경에 대한 반복루프 개시
    env.render()
env.close()