absent = [2,4,7]
no_book = [10]
i = 10
for student in range(1,13):
    if student in absent:
        continue
    elif student in no_book:
        print("정신이 있는거야! {}번 학생 왜 교재를 깜빡해!".format(student))
        break
    print("{}번 학생, 교재 {}페이지 읽어봐!".format(student, i))
    i += 1
