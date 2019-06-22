import re
import numpy as np
import csv
import matplotlib.pyplot as plt


# пакеты/сек - пакеты/мин
def PacksPerMin(traffic):
    result = [0]
    for i in range(0, len(traffic)-60, 60):     # суммируем счетчики пакетов за каждые 60 секунд
        result[-1] = sum(traffic[i:i+60])
        result.append(0)                        # создаем следующий счетчик
    return result


# Среднеквадратическое отклонение
def Sigma(traffic, mean):
    result = 0
    for item in traffic:
        result += (item - mean)**2              # сумма квадратов разниц значений и ср. ариф.
    result *= 1/(len(traffic)-1)                # исправленная дисперсия
    return np.sqrt(result)                      # квадратный корень


traffic = [0]
c_in = 0                                        # количество пакетов в текущем отрезке
time = (3*60 + 0)*60 + 0                        # начальное время
with open('C:/4.csv', 'r') as file:   # открываем файл
    reader = csv.DictReader(file)
    for row in reader:                          # построчно считываем файл
        temp = re.search('\d+:\d+:\d+', row['frame.time']).group(0) # время, указанное в записи
        temp = temp.split(':')                  # разбиваем на часы, минуты и секунды
        temp = (int(temp[0])*60+int(temp[1]))*60 + int(temp[2]) # переводим в секунды
        while True:
            if temp - time > 1:                 # если разница между конечным временем отрезка и текущим больше 1
                traffic.append(0)               # добавляем нулевой элемент в конец списка
                time += 1                       # переходим к следующей секунде
            elif temp < time:                   # если разница больше чем на сутки
                temp += 86400                   # подправляем дату
            else:                               # иначе
                traffic[-1] += 1                # увеличиваем счетчик пакетов на 1
                break

traffic_m = PacksPerMin(traffic)                # считаем пакеты в минуту


mean = np.median(traffic_m)                       # среднее кол-во пакетов
sig = Sigma(traffic_m, mean)                    # среднеквадратическое отклонение
thresholds = []

for i in range (10):
    thresholds.append(0.1*(i+1)*sig)


anomalies = [[0 for i in range(len(traffic_m))] for k in range(len(thresholds))]               # аномалии
minutes = [i for i in range(len(traffic_m))]                # время
cnf_matrix = [[[0 for x in range(2)] for y in range(2)] for z in range(len(thresholds))]

FAS = 3                                                     # рамки предсказанных
FAE = 50                                                    # аномалий
SAS = 31878
SAE = 31930


counter = 0
for k in range(len(thresholds)):
    for i in range(0, len(traffic_m)-10, 10):                   # отсчеты по 10 минут
        if abs(sum(traffic_m[i:i+10])/10 - mean) > (3*thresholds[k]):     # по правилу 3-х сигм
            if k == len(thresholds) - 1:
                counter += 10
            for j in range(10):
                anomalies[k][i+j] = traffic_m[i+j]             # заносим аномалии в список

true = [[], [], [], [], [], [], [], [], [], []]
predictions = []

for i in range(len(traffic_m)):
    if FAS <= i <= FAE or SAS <= i <= SAE:
        predictions.append(1)
    else:
        predictions.append(-1)


for k in range(len(thresholds)):
    true.append([])
    for i in range(len(traffic_m)):                         # поминутные массивы аномалий
        if anomalies[k][i] == 0:                                   # (полученных и предсказанных)
            true[k].append(-1)
        else:
            true[k].append(1)

for k in range (len(thresholds)):
    for i in range (0, len(anomalies[k])):
        if true[k][i] == 1 and predictions[i] == 1:
            cnf_matrix[k][0][0] += 1
        elif true[k][i] == -1 and predictions[i] == -1:
            cnf_matrix[k][1][1] += 1
        elif true[k][i] == 1 and predictions[i] == -1:
            cnf_matrix[k][0][1] += 1
        else:
            cnf_matrix[k][1][0] += 1

FPR = [1]
TPR = [1]


for k in range(len(cnf_matrix)):
    TP = cnf_matrix[k][0][0]
    TN = cnf_matrix[k][1][1]
    FP = cnf_matrix[k][0][1]
    FN = cnf_matrix[k][1][0]
    FPR.append(FP/(FP+TN))
    TPR.append(TP/(TP+FN))


precision = TP/(TP+FP)
beta = 1
F = (1 + beta ** 2)*(precision*TPR[len(TPR)-1])/(precision*(beta ** 2)+TPR[len(TPR)-1])

points = []                                            # крайние точки аномалий
for i in range(len(anomalies[len(anomalies)-1])):
    if anomalies[len(anomalies)-1][i] != 0:
        if anomalies[len(anomalies)-1][i-1] != 0:
            points[-1][1] = i
        else:
            points.append([i, i])



print('Длительность дампа в секундах: ', len(traffic))
print('Длительность дампа в минутах: ', len(traffic_m))
print('Среднеквадратическое отклонение: ', sig)
print('Среднее количество пакетов: ', mean)
print('Количество аномалий: ', counter)
print('Временные рамки аномалий (от начала формирования дампа): ', points)
print('-----------------------------------------------------------')
print('Метрики:')
print('Матрица ошибок:')
print(cnf_matrix[len(cnf_matrix)-1][0][0], cnf_matrix[len(cnf_matrix)-1][0][1])
print(cnf_matrix[len(cnf_matrix)-1][1][0], cnf_matrix[len(cnf_matrix)-1][1][1])
print('Precision: ', precision)
print('Полнота (=true positive rate): ', TPR[len(TPR)-1])
print('F-мера: ', F)
# print('Auc:', auc)
print('Значения порогов для ROC-кривой: ', thresholds)
TPR.append(0)
FPR.append(0)
auc = 0
for i in range (1, len(TPR)-1):
    auc += TPR[i]*(FPR[i-1]-FPR[i])
print('Auc:', auc)

plt.figure()                                            # строим фигуру
plt.title('График аномалий в трафике')                  # заголовок
plt.plot(minutes, anomalies[len(anomalies)-1], 'ro', minutes, traffic_m)  # рисуем график
plt.xlabel('t, мин.')                                   # подпись оси абсцисс...
plt.ylabel('количество пакетов/мин')                    # ...и ординат
plt.grid('on')                                          # сетка (для удобства)
plt.show()                                              # вывод на экран

plt.figure()                                            # аналогично строим график
plt.title('ROC-кривая')                                 # с ROC-кривой
plt.plot(FPR, TPR, color='darkorange', label='ROC-кривая')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid('on')
plt.legend(loc='lower right')
plt.show()
