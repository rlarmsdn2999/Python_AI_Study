year = input("당신이 태어난 년도 : ")
birth_year = int(year)
animal_no = (birth_year+8)%12
animal = ('쥐','소', '호랑이', '토끼', '용','뱀','말','양','원숭이','닭','개','돼지')
print("당신은",animal[animal_no],"띠입니다.")