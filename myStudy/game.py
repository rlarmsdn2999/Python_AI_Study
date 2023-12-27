import random as rd

class Game:
    def __init__(self, coin, betting):
        self.coin = coin
        self.betting = betting

    def win(self):
        self.coin += self.betting
        return self.coin
    
    def lose(self):
        self.coin -= self.betting
        return self.coin
    
a=input("홀짝 맞추기 게임을 시작합니다.")
b=input("당신에게는 100개의 코인이 있습니다.")
c=input("매 경기마다 가지고 있는 코인 내에서 배팅을 합니다.")
d=input("게임에서 이기면 배팅한 코인의 2배를 가져가고, 게임에서 지면 배팅한 코인을 잃습니다.")
coin = int(100)

while True:
    start = int(input("게임을 시작하시겠습니까? 시작 1, 종료 2 : "))
    
    if start == 2 or coin <= 0:
        break

    elif start == 1:
        computer = rd.randint(1, 10)
        betting = int(input("원하는 만큼 배팅을 해주세요(현재 코인은 {}개 입니다.) : ".format(coin)))
        choice = int(input("상대방의 숫자가 홀수이면 1, 짝수이면 2를 입력해주세요 : "))
        print("------------------------")
        print("컴퓨터가 낸 숫자는 {}입니다.".format(computer))
        print("------------------------")

        game = Game(coin, betting)

        if computer % 2 == 0 and choice == 2:
            print("정답입니다!")
            coin = game.win()
            print("게임에서 이겨 코인이 지급됩니다. (현재 코인은 {}개 입니다.)".format(coin))
        elif computer % 2 == 1 and choice == 1:
            print("정답입니다!")
            coin = game.win()
            print("게임에서 이겨 코인이 지급됩니다. (현재 코인은 {}개 입니다.)".format(coin))
        else:
            print("오답입니다!")
            coin = game.lose()
            print("게임에서 패배하여 코인이 차감됩니다. (현재 코인은 {}개 입니다.)".format(coin))

    if coin <= 0:
        print("코인을 다 소진하여 게임이 종료됩니다.")
        break