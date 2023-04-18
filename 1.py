import iapws


def alpha_determine(t,p2,D):
    #Внутренний диаметр
    dint = 17 #мм
    #Кинематическая вязкость 
    v = iapws.IAPWS95(P = p2, T = t).nu # м2/с
    #Теплопроводность материала труб
    lambda_gen = iapws.IAPWS95(P = p2, T = t).k #Вт/(м*К)
    #Число Рейнольдса 
    Re = (D/3600)*dint/v
    #Число Нуссельта на 
    Nu = 0.023 * Re**0.8 * 5.43**0.3
    #Коэффициент теплоотдачи 
    alpha = Nu*lambda_gen/(dint/1000) #Вт/м*К
    return alpha


#Входные данные 
#
tin = float(input()) + 273.15 #K 
# T before HE id 317
tout = float(input()) + 273.15 #K 
# TO1 id 325
# TO2 id 460
# TO3 id 461
# TO4 id 462
# TO5 id 463
#Давление второго контура
p2 = float(input()) #МПа
# P2 
#Расход теплоносителя 
D = int(input()) #м^3/ч

alpha_in = alpha_determine(tin,p2,D)
alpha_out = alpha_determine(tout,p2,D)
alpha_gen = (alpha_in+alpha_out)/2
print('Коэффиициент теплоотдачи:', alpha_gen)




