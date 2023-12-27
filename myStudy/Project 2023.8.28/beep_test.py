# 간단한 비프음을 만들어보자

import winsound
frequency = 1000  # 비프 소리의 주파수 (Hz)
duration = 1000   # 비프 소리의 지속 시간 (밀리초)
winsound.Beep(frequency, duration) # 비프음 재생

for i in range(500, 3000, 100): # 500~3000Hz까지 100씩 증가
    winsound.Beep(i, 500) # 0.5초 간격으로 비프음 재생