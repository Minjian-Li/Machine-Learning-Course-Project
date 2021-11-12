import pandas as pd
import numpy as np

# loading data
rawData1 = pd.read_csv('Data/x_test.csv')
rawData2 = pd.read_csv('Data/y_test.csv')
rawData3 = pd.read_csv('Data/x_validation.csv')
rawData4 = pd.read_csv('Data/y_validation.csv')
rawData5 = pd.read_csv('Data/x_train.csv')
rawData6 = pd.read_csv('Data/y_train.csv')
x_test = np.array(rawData1.values)
y_test = np.array(rawData2.values)
x_validation = np.array(rawData3.values)
y_validation = np.array(rawData4.values)
x_train = np.array(rawData5.values)
y_train = np.array(rawData6.values)

# Compute the prior probabilities P(male), P(female), P(brand) and P(unknown)
male = 0
female = 0
brand = 0
unknown = 0

for i in range(len(y_train)):
    if y_train[i][0] == 1:
        brand += 1
    if y_train[i][0] == 2:
        female += 1
    if y_train[i][0] == 3:
        male += 1
    if y_train[i][0] == 4:
        unknown += 1

P_male = male / len(y_train)
P_female = female / len(y_train)
P_brand = brand / len(y_train)
P_unknown = unknown / len(y_train)

# Compute the P(feature | class) for all features
# probability_TBL0 = [[fn0_c1,fn0_c2,fn0_c3,fn0_c4]...]
# probability_TBL1 = [[fn1_c1,fn1_c2,fn1_c3,fn1_c4]...]
probability_TBL0 = [0] * np.shape(x_train)[1]
probability_TBL1 = [0] * np.shape(x_train)[1]

for j in range(np.shape(x_train)[1]):
    f0_c1 = 0
    f0_c2 = 0
    f0_c3 = 0
    f0_c4 = 0
    f1_c1 = 0
    f1_c2 = 0
    f1_c3 = 0
    f1_c4 = 0
    P_f0_c1 = 0
    P_f0_c2 = 0
    P_f0_c3 = 0
    P_f0_c4 = 0
    P_f1_c1 = 0
    P_f1_c2 = 0
    P_f1_c3 = 0
    P_f1_c4 = 0
    for k in range(np.shape(x_train)[0]):
        if x_train[k][j] == 0:
            if y_train[k][0] == 1:
                f0_c1 += 1
            if y_train[k][0] == 2:
                f0_c2 += 1
            if y_train[k][0] == 3:
                f0_c3 += 1
            if y_train[k][0] == 4:
                f0_c4 += 1
        if x_train[k][j] != 0:
            if y_train[k][0] == 1:
                f1_c1 += x_train[k][j]
            if y_train[k][0] == 2:
                f1_c2 += x_train[k][j]
            if y_train[k][0] == 3:
                f1_c3 += x_train[k][j]
            if y_train[k][0] == 4:
                f1_c4 += x_train[k][j]

    if f0_c1 == 0:
        P_f0_c1 = (1 / len(y_train)) / ((brand + 2) / len(y_train))
    else:
        P_f0_c1 = (f0_c1 / len(y_train)) / P_brand
    if f0_c2 == 0:
        P_f0_c2 = (1 / len(y_train)) / ((female + 2) / len(y_train))
    else:
        P_f0_c2 = (f0_c2 / len(y_train)) / P_female
    if f0_c3 == 0:
        P_f0_c3 = (1 / len(y_train)) / ((male + 2) / len(y_train))
    else:
        P_f0_c3 = (f0_c3 / len(y_train)) / P_male
    if f0_c4 == 0:
        P_f0_c4 = (1 / len(y_train)) / ((unknown + 2) / len(y_train))
    else:
        P_f0_c4 = (f0_c4 / len(y_train)) / P_unknown

    if f1_c1 == 0:
        P_f1_c1 = 1 / (brand + 2)
    else:
        P_f1_c1 = (f1_c1 / len(y_train)) / P_brand
    if f1_c2 == 0:
        P_f1_c2 = 1 / (female + 2)
    else:
        P_f1_c2 = (f1_c2 / len(y_train)) / P_female
    if f1_c3 == 0:
        P_f1_c3 = 1 / (male + 2)
    else:
        P_f1_c3 = (f1_c3 / len(y_train)) / P_male
    if f1_c4 == 0:
        P_f1_c4 = 1 / (unknown + 2)
    else:
        P_f1_c4 = (f1_c4 / len(y_train)) / P_unknown

    probability_TBL0[j] = [P_f0_c1, P_f0_c2, P_f0_c3, P_f0_c4]
    probability_TBL1[j] = [P_f1_c1, P_f1_c2, P_f1_c3, P_f1_c4]

print(probability_TBL0)
print(probability_TBL1)


# Prediction function

def prediction(x, y):
    mistake = 0
    for l in range(np.shape(x)[0]):
        p1 = P_brand
        p2 = P_female
        p3 = P_male
        p4 = P_unknown
        p_bold = [0] * 4

        for m in range(np.shape(x)[1]):
            if x[l][m] == 0:
                p1 *= probability_TBL0[m][0]
                p2 *= probability_TBL0[m][1]
                p3 *= probability_TBL0[m][2]
                p4 *= probability_TBL0[m][3]

            if x[l][m] != 0:
                p1 *= probability_TBL1[m][0]
                p2 *= probability_TBL1[m][1]
                p3 *= probability_TBL1[m][2]
                p4 *= probability_TBL1[m][3]

        p_bold[0] = p1
        p_bold[1] = p2
        p_bold[2] = p3
        p_bold[3] = p4
        yhat = np.argmax(p_bold) + 1
        if yhat != y[l][0]:
            mistake += 1
    accuracy = 1 - (mistake / np.shape(y)[0])
    print(accuracy)


print("Training accuracy:")
prediction(x_train, y_train)
print("Validation accuracy:")
prediction(x_validation, y_validation)
print("Testing accuracy:")
prediction(x_test, y_test)
