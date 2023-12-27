from random import *
cnt = 0
for i in range(1,51):
    time = randrange(5,51)
    if 5 <= time <= 15:
        print("[o]{0}번 손님(소요시간 : {1}분)".format(i,time))
        cnt += 1
    else:
        print("[x]{0}번 손님(소요시간 : {1}분)".format(i,time))
print("50명중 {}명의 손님을 태웠습니다.".format(cnt))