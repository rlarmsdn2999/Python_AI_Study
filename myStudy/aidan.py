import aidan_module as am
import random as rd
import drow
a = input("RK단의 영웅 Aidan의 모험")
attackPower = input("Aidan의 공격력을 입력해주세요 : ")
attack = int(attackPower)
hpPower = input("Aidan의 방어력을 입력해주세요 : ")
hp = int(hpPower)
dressPower = input("Aidan의 장비레벨을 입력해주세요 : ")
dress = int(dressPower)
am.combatPower(attack, hp, dress)
f = input("")
drow.drow()
b = input("Aidan이 모험을 떠나다 동굴에서 Diablo를 만났습니다.")
diablo = rd.randint(1,100)
d = input("Diablo의 체력은 {0}입니다.".format(diablo))
aidan = am.compatPower2(attack, hp, dress)
c = input("Aidan({0})이 Diablo({1})를 공격합니다.".format(aidan, diablo))
if aidan >= diablo:
    print("Aidan의 전투력이 Diablo보다 높아 물리쳤습니다.")
else:
    print("Aidan의 전투력이 Diablo보다 낮아 패배하였습니다.")