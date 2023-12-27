# 유전적 알고리즘(Genetic Algorothm)을 이용하여 세일즈맨 문제를 풀어보자.
# 유전적 알고리즘은 생물의 진화를 모방하여 최적의 해를 찾아내는 알고리즘이다.
# 교차(crossover)와 돌연변이(mutation)를 통해 새로운 개체를 생성하고,
# 평가(evaluation)를 통해 적합도를 측정하여 적합도가 높은 개체를 선택하는 방식이다.
# 세일즈맨 문제는 통과할 지점(노드)의 수가 n개일 때, n!개의 경로 중 가장 짧은 경로를 찾는 문제이다.

from datetime import datetime
import random
from decimal import Decimal
import numpy as np
from itertools import zip_longest
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 염색체를 정의
class Chromosome:

    point_info = None
    route = None
    evaluation = None

    def __init__(self, point_info, route, evaluation):
        self.point_info = point_info
        self.route = route
        self.evaluation = evaluation
        
    def getPoint_info(self):
        return self.point_info

    def getEvaluation(self):
        return self.evaluation
    
    def getRoute(self):
        return self.route

    def setPoint_info(self, point_info):
        self.coordinate = point_info

    def setRoute(self, route):
        self.route = route
        
    def setEvaluation(self, evaluation):
        self.evaluation = evaluation

# 통과할 지점(노드)의 수
COORDINATE_NUM = 15
# 유전자 정보의 길이
GENOM_LENGTH = COORDINATE_NUM
# 유전자 집단의 크기
MAX_GENOM_LIST = 100
# 유전자 선택 수
SELECT_GENOM = 10
# 개체 돌연변이 확률
INDIVIDUAL_MUTATION = 0.1
# 반복되는 세대(generation)수
MAX_GENERATION = 40
# 반복을 멈추는 평가값의 역치
THRESSHOLD = 0

# 통과할 지점을 랜덤하게 생성
# 범위(넓이) 지정
x_range = 100
y_range = 100
# 랜덤하게 통과할 지점을 생성
x_coordinate = [random.randint(0, x_range) for _ in range(COORDINATE_NUM)]
y_coordinate = [random.randint(0, y_range) for _ in range(COORDINATE_NUM)]
# [x,y] 2차원 배열을 생성
coordinate = [[x_coordinate[i], y_coordinate[i]] for i in range(COORDINATE_NUM)]

# {key:도시(통과 지점) 번호, value:좌표값}의 딕셔너리를 생성
position_info = {}
for i in range(COORDINATE_NUM):
    position_info[i] = coordinate[i]

def create_Chromosome(point_num, point_info):
    """
    인수로 지정된 자리수의 랜덤한 유전자정보를 생성하고 ChromosomeClass를 통해 반환
    :param point_num: 통과할 지점의 수
    :param point_info: 통과할 지점의 정보
    :return: 생성한 개체집단 ChromosomeClass
    """
    # 초기의 통과 루틴(순서)을 적당히 생성
    select_num = [i for i in range(point_num)]
    route = random.sample(select_num, point_num)

    return Chromosome(position_info, route, 0)

def evaluation(Chromosome):
    """
    평가함수(Evaluation Fn): 유클리드 기하학적 거리의 총합을 계산하여 평가값으로 설정
    :param Chromosome: 평가를 실행하는 ChromosomeClass
    :return: 평가 처리를 한 ChromosomeClass를 반환
    """
    route = Chromosome.getRoute()
    coordinate = Chromosome.getPoint_info()
    evaluate = []
    x_coordinate = [coordinate[route[x]][0] for x in range(len(route))]
    y_coordinate = [coordinate[route[y]][1] for y in range(len(route))]
    
    for i in range(len(route)):
        if i == len(route) - 1:
            distance = math.sqrt(pow((x_coordinate[i] - x_coordinate[0]), 2) + pow((y_coordinate[i] - y_coordinate[0]), 2))
        else:
            distance = math.sqrt(pow((x_coordinate[i] - x_coordinate[i + 1]), 2) + pow((y_coordinate[i] - y_coordinate[i + 1]), 2))
        evaluate.append(distance)
        
    return sum(evaluate)

def elite_select(Chromosome, elite_length):
    """
    선택함수: 엘리트 선택
    평가값이 높은 순서로 sorting을 한 후, 일정값 이상의 염색체를 선택
    :param Chromosome: 선택을 실행하는 ChromosomeClass의 배열
    :param elite_length: 선택할 염색체수
    :return: 선택처리를 한 일정 이상의 엘리트 ChromosomeClass를 반환
    """
    # 현재 세대의 개체집단의 평가를 낮은 순서로 sorting
    sort_result = sorted(Chromosome, reverse=False, key=lambda u: u.evaluation)
    # 일정 이상의 엘리트 개체를 선택
    result = [sort_result.pop(0) for i in range(elite_length)]
    return result

def roulette_select(Chromosome, choice_num):
    """
    선택함수: 룰렛(roulette) 선택, 원판을 돌려 적당히 선택하는 방식
    적응도에 따라 가중치를 부여하여 랜덤하게 선택
    :param Chromosome: 선택을 실행하는 ChromosomeClass의 배열
    :param elite_length: 선택할 염색체수
    :return: 선택처리를 한 염색체 ChromosomeClass을 반환
    """
    # 적응도를 배열화
    fitness_arr = np.array([float(genom.evaluation) for genom in Chromosome])
    
    idx = np.random.choice(np.arange(len(Chromosome)), size=choice_num, p=fitness_arr/sum(fitness_arr))
    result = [Chromosome[i] for i in idx]
    return result

def tournament_select(Chromosome, choice_num):
    """
    선택함수: 토너먼트(tournament) 선택, 엘리트선택과 룰렛선택을 합친 방식
    평가값이 높은 순서로 sorting을 한 후, 일정값 이상의 염색체를 선택
    :param Chromosome: 선택을 실행하는 ChromosomeClass의 배열
    :param elite_length: 선택할 염색체수
    :return: 선택처리를 한 염색체(유전자) ChromosomeClass을 반환
    """
    # 적응도를 배열화
    fitness_arr = [float(genom.evaluation) for genom in Chromosome]
    next_gene_arr = []
    for i in range(choice_num):
        [idx_chosen1, idx_chosen2] = np.random.randint(MAX_GENOM_LIST, size=2)
        if fitness_arr[idx_chosen1] > fitness_arr[idx_chosen1]:
            next_gene_arr.append(Chromosome[idx_chosen1])
        else:
            next_gene_arr.append(Chromosome[idx_chosen2])

    return np.array(next_gene_arr)

def crossover(Chromosome_one, Chromosome_second):
    """
    교차 함수: 순서교차(ordered crossover), 첨부의 그림파일 참조
    :param Chromosome: 교차시킬 ChromosomeClass의 배열
    :param Chromosome_one: 첫번째 개체 (아빠)
    :param Chromosome_second: 두번째 개체 (엄마)
    :return: 두개의 자손 ChromosomeClass를 격납한 리스트를 반환
    """
    # 자손을 격납할 리스트를 생성
    genom_list = []
    progeny_one = []
    progeny_second = []
    # 교차시킬 두점의 위치를 설정
    cross_one = random.randint(0, GENOM_LENGTH)
    cross_second = random.randint(cross_one, GENOM_LENGTH)
    # 교차시킬 유전자를 추출
    one = Chromosome_one.getRoute()
    second = Chromosome_second.getRoute()
    # 교차
    one2 = one[cross_one:cross_second]
    second2 = second[cross_one:cross_second]
    
    for second_i in second:
        if len(progeny_one) == cross_one:
            progeny_one.extend(one2)
            break
        if second_i not in one2:
            progeny_one.append(second_i)
    if len(progeny_one) < len(one):
        for second_i in second:
            if second_i not in progeny_one:
                progeny_one.append(second_i)
        
    for one_i in one:
        if len(progeny_second) == cross_one:
            progeny_second.extend(second2)
            break
        if one_i not in second2:
            progeny_second.append(one_i)
    if len(progeny_second) < len(second):
        for one_i in one:
            if one_i not in progeny_second:
                progeny_second.append(one_i)
    # ChromosomeClass 인스턴스를 생성하여 자손을 리스트에 격납
    genom_list.append(Chromosome(Chromosome_one.getPoint_info(), progeny_one, 0))
    genom_list.append(Chromosome(Chromosome_second.getPoint_info(), progeny_second, 0))
    return genom_list

def mutation(Chromosome, individual_mutation):
    """
    돌연변이 함수: 통과 경로의 순서를 랜덤하게 재배치
    :param Chromosome: 돌연변위를 일으킬 ChromosomeClass
    :param individual_mutation: 고정 유전자에 대한 돌연변이 확률
    :return: 돌연변이 처리를 한 genomClass를 반환
    """
    Chromosome_list = []
    for genom in Chromosome:
        # 각 개체에 대하여 일정 확률로 돌연변이가 일어남
        if individual_mutation > (random.randint(0, 100) / Decimal(100)):
            route = genom.getRoute()
            select_num = [i for i in range(len(route))]
            select_index = random.sample(select_num, 2)
            temp = route[select_index[0]]
            route[select_index[0]] = route[select_index[1]]
            route[select_index[1]] = temp
            genom.setRoute(route)
            Chromosome_list.append(genom)
        else:
            Chromosome_list.append(genom)
            
    return Chromosome_list

def next_generation_gene_create(Chromosome, Chromosome_elite, Chromosome_progeny):
    """
    세대 교체 처리 함수: 다음 세대의 유전자를 생성
    :param Chromosome: 현 세대의 개체집단
    :param Chromosome_elite: 현 세대의 엘리트 집단
    :param Chromosome_progeny: 현 세대의 자손 집단
    :return: 차세대 개체집단
    """
    # 현 세대 개체집단의 평가를 점수가 높은 순으로 sorting
    next_generation_geno = sorted(Chromosome, reverse=True, key=lambda u: u.evaluation)
    # 추가할 엘리트집단과 자손집단의 합계분을 제거
    for i in range(0, len(Chromosome_elite) + len(Chromosome_progeny)):
        next_generation_geno.pop(0)
    # 엘리트 집단과 자손집단을 차세대 집단에 추가
    next_generation_geno.extend(Chromosome_elite)
    next_generation_geno.extend(Chromosome_progeny)
    return next_generation_geno

# 맨 처음의 현재 세대 개체집단을 생성
current_generation_individual_group = []
for i in range(MAX_GENOM_LIST):
    current_generation_individual_group.append(create_Chromosome(COORDINATE_NUM, GENOM_LENGTH))

for count_ in range(1, MAX_GENERATION + 1):
    # 현 세대의 개체집단 유전자를 평가하여 ChromosomeClass에 대입
    for i in range(MAX_GENOM_LIST):
        evaluation_result = evaluation(current_generation_individual_group[i])
        current_generation_individual_group[i].setEvaluation(evaluation_result)
    # 엘리트 개체를 선택
    elite_genes = elite_select(current_generation_individual_group,SELECT_GENOM)
    # 엘리트 유전자를 교차시켜 리스트에 격납
    progeny_gene = []
    for i in range(0, SELECT_GENOM):
        progeny_gene.extend(crossover(elite_genes[i - 1], elite_genes[i]))
    # 차세대 개체집단을 현 세대의 엘리트집단, 자손집단으로 부터 생성
    next_generation_individual_group = next_generation_gene_create(current_generation_individual_group,
                                                                   elite_genes, progeny_gene)
    # 차세대 개체집단 전체(개체)에 돌연변이를 일으킴
    next_generation_individual_group = mutation(next_generation_individual_group,INDIVIDUAL_MUTATION)

    # 1세대 진화에 대한 계산을 종료

    # 각 개체적용도를 배열화
    fits = [i.getEvaluation() for i in current_generation_individual_group]

    # 진화 결과를 평가
    min_ = min(fits)
    max_ = max(fits)
    avg_ = Decimal(sum(fits)) / Decimal(len(fits))

    # 현 세대의 진화 결과를 출력
    print(datetime.now(),
          f'세대수 : {count_}  ',
          f'Min : {min_:.3f} ',
          f'Max : {max_:.3f}  ',
          f'Avg : {avg_:.3f}  '
         )
    # 현 세대와 차세대를 재배치. 차세대 -> 현세대 (진화)
    current_generation_individual_group = next_generation_individual_group
    # 적응도가 역치를 넘어서면 종료
    if THRESSHOLD >= min_:
        print('optimal')
        print(datetime.now(),
          f'세대수 : {count_}  ',
          f'Min : {min_:.3f} ',
          f'Max : {max_:.3f}  ',
          f'Avg : {avg_:.3f}  '
         )
        break
# 최우수 개체 결과 출력
print(f'최우수 개체 정보:{elite_genes[0].getRoute()}')

# 최우수 개체의 통과 경로를 Plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# 최우수 개체를 가지고 옴
genes = elite_genes[0]
route = genes.getRoute()
print(f'position_info {position_info}')
print(f'통과 경로 {route}')
x_coordinate = [position_info[route[x]][0] for x in range(len(route))]
y_coordinate = [position_info[route[y]][1] for y in range(len(route))]
x_coordinate.append(position_info[route[0]][0])
y_coordinate.append(position_info[route[0]][1])
ax.scatter(x_coordinate, y_coordinate)
ax.plot(x_coordinate, y_coordinate)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Best Individual {genes.getEvaluation():.3f}')
plt.show()