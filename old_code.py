import numpy as np
import math
from math import pi
import matplotlib.pyplot as plt

REAL_RATIO = 0.1

# find highest UCB
def K_best(m):
    Lis = np.argsort(m)[-K:]
    Lis = np.flipud(Lis)
    reward = 1 - np.prod(1 - m[Lis])
    return Lis, reward

# determine alpha attack value
def alpha(arm):
    ret = 0
    for j in List:
        if Ti[j] == 1:
            Ti[j] += 1
        tem = Mu_hat_0[j] - delta0 - 2 * math.sqrt(math.log(pi * pi * L * (Ti[j] - 1) * (Ti[j] - 1) / (3 * delta)) / (2 * (Ti[j] - 1)))
        tem = max(tem, 0)
        rtemp = Mu_hat_0[arm] * (Ti[arm] - 1) + 1 - Attack[arm] - tem * Ti[arm]
        if rtemp <= 1:
            Tstars[arm, j] = tem
        else:
            rtemp = Mu_hat_0[arm] * (Ti[arm] - 1) + 1 - Attack[arm] - Tstars[arm, j] * Ti[arm]
        if rtemp > ret:
            ret = rtemp
    return ret


L, K, T = 16, 8, 100000
delta0, delta = 0.1, 0.1
Repeat = 20
M = 9
List = [6, 7, 8, 9, 10, 11, 12, 13] #attack? arm list
Cost1, CostT, CostK = np.array([np.zeros(Repeat)] * T), np.array([np.zeros(Repeat)] * T), np.array([np.zeros(Repeat)] * T)
Time1, TimeT, TimeK = np.array([np.zeros(Repeat)] * T), np.array([np.zeros(Repeat)] * T), np.array([np.zeros(Repeat)] * T)
A = np.zeros(T) #attack value in each round
Target = np.zeros(T) #


FLAG = False
Mu = np.zeros(L)

#store pre-attack optimal arms
optK, optR = K_best(Mu)
print("Optimal Arms:\t", end='')
for item in optK:
    print(Mu[item], end=" ")
print()
for item in Mu:
    print(item)


file = open('mu.txt', 'w')
file.write(str(Mu))
file.close()


for counter in range(Repeat):
    print()
    print("Counter: " + str(counter))


    Radius = np.zeros(L)
    Ti = np.ones(L)
    Mu_hat = (np.random.rand(L) < Mu).astype(float)
    Mu_hat_0 = np.zeros(L)
    Attack = np.zeros(L)
    Ntimes = np.zeros(L)
    Tstars = np.zeros((L, L)) # store data from previous rounds


    for t in range(T):
        Radius = np.sqrt(3 * np.log(t + 1) / (2 * Ti))
        UCB = np.clip(Mu_hat + Radius, a_min=0, a_max=1)
        Rec, _ = K_best(UCB)

        for item in Rec:
            Ntimes[item] += 1 #if it is in optimal arm we can increase by 1

        flag = False
        flip = np.random.rand()
        if flip < REAL_RATIO:
            for item in Rec:
                Ti[item] += 1 
                temp = (np.random.rand() < Mu[item]).astype(int)
                Mu_hat_0[item] += (temp - Mu_hat_0[item]) / Ti[item] 
                Mu_hat[item] += (temp - Mu_hat[item]) / Ti[item]
                if temp == 1:
                    break
        else:
            for item in Rec:
                Ti[item] += 1 #check if recommended?
                temp = (np.random.rand() < Mu[item]).astype(int) #click result?
                if flag:
                    if item not in List:
                        temp = 0
                    else:
                        temp = 1
                        Attack[item] += 1
                if item not in List:
                    if temp == 1:
                        Mu_hat_0[item] += (1 - Mu_hat_0[item]) / Ti[item]
                        Alpha = alpha(item)
                        if Alpha > 0:
                            temp = 0
                            Attack[item] += 1
                            flag = True
                    else:
                        Mu_hat_0[item] += (0 - Mu_hat_0[item]) / Ti[item]
                else:
                    Mu_hat_0[item] += (temp - Mu_hat_0[item]) / Ti[item]
                Mu_hat[item] += (temp - Mu_hat[item]) / Ti[item]
                if temp == 1:
                    break
                
        Target[t] = (Target[t] * counter + Ntimes[M]) / (counter + 1)
        A[t] = (A[t] * counter + sum(Attack)) / (counter + 1)
        Cost1[t][counter] = sum(Attack)
        Time1[t][counter] = Ntimes[M] / (t + 1)


Ratio = np.zeros(T)
for t in range(T):
    Ratio[t] = Target[t] / (t + 1)


AT = np.zeros(T)
TargetT = np.zeros(T)


for counter in range(Repeat):
    print()
    print("Counter: " + str(counter))


    Radius = np.zeros(L)
    Ti = np.ones(L)
    Mu_hat = (np.random.rand(L) < Mu).astype(float)
    Attack = np.zeros(L)
    Ntimes = np.zeros(L)
    Tstars = np.zeros((L, L))

    for t in range(T):
        Radius = np.sqrt(3 * np.log(t + 1) / (2 * Ti))
        UCB = np.clip(Mu_hat + Radius, a_min=0, a_max=1)
        Rec, _ = K_best(UCB)

        for item in Rec:
            Ntimes[item] += 1

        flag = False
        for item in Rec:
            Ti[item] += 1
            temp = (np.random.rand() < Mu[item]).astype(int)
            if flag:
                temp = 0
            if item is not M:
                if temp == 1:
                    temp = 0
                    Attack[item] += 1
                    flag = True
            Mu_hat[item] += (temp - Mu_hat[item]) / Ti[item]
            if temp == 1:
                break

        TargetT[t] = (TargetT[t] * counter + Ntimes[M]) / (counter + 1)
        AT[t] = (AT[t] * counter + sum(Attack)) / (counter + 1)
        CostT[t][counter] = sum(Attack)
        TimeT[t][counter] = Ntimes[M] / (t + 1)


RatioT = np.zeros(T)
for t in range(T):
    RatioT[t] = TargetT[t] / (t + 1)


AK = np.zeros(T)
TargetK = np.zeros(T)


for counter in range(Repeat):
    print()
    print("Counter: " + str(counter))


    Radius = np.zeros(L)
    Ti = np.ones(L)
    Mu_hat = (np.random.rand(L) < Mu).astype(float)
    Attack = np.zeros(L)
    Ntimes = np.zeros(L)
    Tstars = np.zeros((L, L))

    for t in range(T):
        Radius = np.sqrt(3 * np.log(t + 1) / (2 * Ti))
        UCB = np.clip(Mu_hat + Radius, a_min=0, a_max=1)
        Rec, _ = K_best(UCB)

        for item in Rec:
            Ntimes[item] += 1

        for item in Rec:
            Ti[item] += 1
            temp = (np.random.rand() < Mu[item]).astype(int)
            if item not in List:
                if temp == 1:
                    temp = 0
                    Attack[item] += 1
            Mu_hat[item] += (temp - Mu_hat[item]) / Ti[item]
            if temp == 1:
                break

        TargetK[t] = (TargetK[t] * counter + Ntimes[M]) / (counter + 1)
        AK[t] = (AK[t] * counter + sum(Attack)) / (counter + 1)
        CostK[t][counter] = sum(Attack)
        TimeK[t][counter] = Ntimes[M] / (t + 1)


RatioK = np.zeros(T)
for t in range(T):
    RatioK[t] = TargetK[t] / (t + 1)


VarCost1, VarCostT, VarCostK = np.zeros(T), np.zeros(T), np.zeros(T)
VarTime1, VarTimeT, VarTimeK = np.zeros(T), np.zeros(T), np.zeros(T)
for index in range(T):
    VarCost1[index] = np.std(Cost1[index])
    VarCostT[index] = np.std(CostT[index])
    VarCostK[index] = np.std(CostK[index])
    VarTime1[index] = np.std(A[index])
    VarTimeT[index] = np.std(AT[index])
    VarTimeK[index] = np.std(AK[index])


plt.figure(figsize=(12, 8), dpi=100)
plt.grid(True)
x = np.linspace(1, T, T)
ymin = min(min(Ratio), min(RatioK), min(RatioT))
plt.ylim(ymin - 0.1, 1.1)
plt.xlim(0, 100)
plt.plot(x, Ratio, label="Our attack", color='red', lw=2.8)
plt.plot(x, Ratio + VarTime1, color='pink', lw=0.8)
plt.plot(x, Ratio - VarTime1, color='pink', lw=0.8)
plt.fill_between(x, Ratio + VarTime1, Ratio - VarTime1, alpha=0.25, color='pink')

plt.plot(x, RatioT, label="Trivial₁", color='blue', lw=2.4)
plt.plot(x, RatioT + VarTimeT, color='skyblue', lw=0.8)
plt.plot(x, RatioT - VarTimeT, color='skyblue', lw=0.8)
plt.fill_between(x, RatioT + VarTimeT, RatioT - VarTimeT, alpha=0.25, color='skyblue')

plt.plot(x, RatioK, label="Trivialₖ₋₁", color='green', lw=2)
plt.plot(x, RatioK + VarTimeK, color='lightgreen', lw=0.8)
plt.plot(x, RatioK - VarTimeK, color='lightgreen', lw=0.8)
plt.fill_between(x, RatioK + VarTimeK, RatioK - VarTimeK, alpha=0.25, color='lightgreen')

plt.xlabel("t", fontsize=18)
plt.ylabel("Chosen ratio", fontsize=18)
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
plt.savefig("Ratio1.png", bbox_inches='tight')
plt.show()


plt.figure(figsize=(12, 8), dpi=100)
plt.grid(True)
x = np.linspace(1, T, T)
ymax = max(max(A), max(AK))
plt.ylim(0, ymax + 10)

plt.plot(x, A, label="Our attack", color='red', lw=2.4)
plt.plot(x, A + VarCost1, color='pink', lw=0.8)
plt.plot(x, A - VarCost1, color='pink', lw=0.8)
plt.fill_between(x, A + VarCost1, A - VarCost1, alpha=0.25, color='pink')

plt.plot(x, AT, label="Trivial₁", color='blue', lw=2.4)
plt.plot(x, AT + VarCostT, color='skyblue', lw=0.8)
plt.plot(x, AT - VarCostT, color='skyblue', lw=0.8)
plt.fill_between(x, AT + VarCostT, AT - VarCostT, alpha=0.25, color='skyblue')

plt.plot(x, AK, label="Trivialₖ₋₁", color='green', lw=2.4)
plt.plot(x, AK + VarCostK, color='lightgreen', lw=0.8)
plt.plot(x, AK - VarCostK, color='lightgreen', lw=0.8)
plt.fill_between(x, AK + VarCostK, AK - VarCostK, alpha=0.25, color='lightgreen')

plt.xlabel("t", fontsize=18)
plt.ylabel("Cost", fontsize=18)
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
plt.savefig("Cost1.png", bbox_inches='tight')
plt.show()
