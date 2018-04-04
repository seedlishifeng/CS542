#!/usr/bin/python
import pandas as pd
import math
import sys
import numpy
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def normalize(x):
    re = pd.DataFrame()
    for j in range(0, x.shape[1]):
        if j == 1 or j == 2 or j == 7 or j == 10 or j == 13 or j == 14:
            m = x[j].mean()
            s = numpy.std(x[j], ddof=1)
            for i in range(0, x.shape[0]):
                re.loc[i, j] = (float(x.loc[i, j]) - m) / s
        else:
            for i in range(0, x.shape[0]):
                re.loc[i, j] = x.loc[i, j]
    return re

def normalize1(x):
    re = pd.DataFrame()
    for j in range(0, x.shape[1]):
        m = x[j].mean()
        s = numpy.std(x[j], ddof=1)
        for i in range(0, x.shape[0]):
            re.loc[i, j] = (float(x.loc[i, j]) - m) / s
    return re

def knn(train, test, k):
    dis = pd.DataFrame()
    for i in range(0, train.shape[0]):
        s = 0
        for j in range(0, test.shape[0]):
            if j == 1 or j == 2 or j == 7 or j == 10 or j == 13 or j == 14:
                s += math.pow((float(train.ix[i, j]) - float(test.ix[j, 0])), 2)
            else:
                if test.ix[j, 0] == train.ix[i, j]:
                    s += 0
        dis.loc[i, 0] = int(i)
        dis.loc[i, 1] = float(math.sqrt(s))

    sort_dis = dis.sort_values(by=1, ascending=True).head(k)
    return sort_dis

def knn1(train, test, k):
    dis = pd.DataFrame()
    for i in range(0, train.shape[0]):
        s = 0
        for j in range(0, test.shape[0]):
            s += math.pow((float(train.ix[i, j]) - float(test.ix[j, 0])), 2)
        dis.loc[i, 0] = int(i)
        dis.loc[i, 1] = float(math.sqrt(s))

    sort_dis = dis.sort_values(by=1, ascending=True).head(k)
    return sort_dis

def reg(x, dis):
    c_p = 0
    c_m = 0
    for value in dis[0]:
        if x.ix[value, 0] == '+':
            c_p += 1
        else:
            c_m += 1
    if c_p > c_m:
        return '+'
    else:
        return '-'
def reg1(x, dis):
    c = [0]*3
    for value in dis[0]:
        if x.ix[value, 0] == 1:
            c[0] += 1
        if x.ix[value, 0] == 2:
            c[1] += 1
        if x.ix[value, 0] == 3:
            c[2] += 1
    if c[0]==max(c):
        return 1
    if c[1]==max(c):
        return 2
    if c[2]==max(c):
        return 3

def main(k, ftrain, ftest):
    test = pd.read_csv(ftest, header=None)
    train = pd.read_csv(ftrain, header=None)

    # get data
    label_x = train[train.shape[1] - 1]
    x_train = train[list(range(train.shape[1] - 1))]
    x_test = test[list(range(test.shape[1] - 1))]

    # normalize
    x_train_norm = normalize(x_train)
    x_test_norm = normalize(x_test)

    # knn
    label_y = pd.DataFrame()
    for i in range(0, x_test_norm.shape[0]):
        dis_k = knn(x_train_norm, x_test_norm.ix[i, :], k)
        label = reg(label_x, dis_k)
        label_y.loc[i, 0] = label

    # append output
    test[16] = label_y
    name = 'knn_output.csv'
    count = 0
    for i in range(0, test.shape[0]):
        if test.ix[i, 15] == test.ix[i,16]:
            count = count + 1
        else:
            count = count + 0
    accuracy = float(count/test.shape[0])
    print(accuracy)
    test.to_csv(name, header=None, index=None)

def main2(k,ftrain,ftest):
    test = pd.read_csv(ftest, header=None)
    train = pd.read_csv(ftrain, header=None)

    # get data
    label_x = train[train.shape[1] - 1]
    x_train = train[list(range(train.shape[1] - 1))]
    x_test = test[list(range(test.shape[1] - 1))]

    # normalize
    x_train_norm = normalize1(x_train)
    x_test_norm = normalize1(x_test)

    # knn
    label_y = pd.DataFrame()
    for i in range(0, x_test_norm.shape[0]):
        dis_k = knn1(x_train_norm, x_test_norm.ix[i, :], k)
        label = reg1(label_x, dis_k)
        label_y.loc[i, 0] = label

    # append output
    test[5] = label_y
    name = 'knn_output1.csv'
    print(test)
    count = 0
    for i in range(0, test.shape[0]):
        if test.ix[i, 4] == test.ix[i, 5]:
            count = count + 1
        else:
            count = count + 0
    accuracy = float(count / test.shape[0])
    print(accuracy)
    test.to_csv(name, header=None, index=None)


def justfyimplement(k,train,test):
    df = pd.read_csv(train, header=None)
    if is_number(df.ix[0,0]):
        main2(k,train,test)
    else:
        main(k,train,test)


k = int(sys.argv[1])
train = sys.argv[2]
test = sys.argv[3]
justfyimplement(k, train,test)
#justfyimplement(4,'lenses.training.processed.csv', 'lenses.testing.processed.csv')
#justfyimplement(5,'crx.data.training.processed.csv','crx.data.testing.processed.csv')