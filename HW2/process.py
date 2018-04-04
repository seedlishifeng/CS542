#!/usr/bin/python
import pandas as pd
import sys
from collections import Counter
def process(path):
    df = pd.read_csv(path, header=None)
    for j in range(0, df.shape[1]):
        if j == 1 or j == 2 or j == 7 or j == 10 or j == 13 or j == 14:
            s = 0
            count = 0
            for i in range(0, df.shape[0]):
                if df.ix[i, j] != '?'and df.ix[i,15]=='+':
                    s += float(df.ix[i, j])
                    count += 1
            mean = s / count
            for i in range(0, df.shape[0]):
                if df.ix[i, j] == '?'and df.ix[i,15]=='+':
                    df.ix[i, j] = mean
            s1 = 0
            count1 = 0
            for i in range(0, df.shape[0]):
                if df.ix[i, j] != '?'and df.ix[i,15]=='-':
                    s1 += float(df.ix[i, j])
                    count1 += 1
            mean1 = s1 / count1
            for i in range(0, df.shape[0]):
                if df.ix[i, j] == '?' and df.ix[i,15]=='-':
                    df.ix[i, j] = mean1
        else:
            pos = []
            neg =[]
            for i in range(0, df.shape[0]):
                if df.ix[i, j] != '?'and df.ix[i, 15] == '+':
                    pos.append(df.ix[i, j])
            for i in range(0, df.shape[0]):
                if df.ix[i, j] != '?'and df.ix[i, 15] == '-':
                    neg.append(df.ix[i, j])
            mode = Counter(pos).most_common(1)
            for i in range(0, df.shape[0]):
                if df.ix[i, j] == '?'and df.ix[i, 15] == '+':
                    df.ix[i, j] = mode[0][0]
            mode1 = Counter(pos).most_common(1)
            for i in range(0, df.shape[0]):
                if df.ix[i, j] == '?'and df.ix[i, 15] == '-':
                    df.ix[i, j] = mode1[0][0]

    name = path+'.processed.csv'
    df.to_csv(name, header=None, index=None)

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
def process2(path):
    df = pd.read_csv(path, header=None)
    for j in range(0, df.shape[1]):
        s = 0
        count = 0
        for i in range(0, df.shape[0]):
            if df.ix[i, j] != '?' and df.ix[i, 4] == 1:
                s += float(df.ix[i, j])
                count += 1
        mean = s / count
        for i in range(0, df.shape[0]):
            if df.ix[i, j] == '?' and df.ix[i, 4] == 1:
                df.ix[i, j] = mean
        s1 = 0
        count1 = 0
        for i in range(0, df.shape[0]):
            if df.ix[i, j] != '?' and df.ix[i, 4] == 2:
                s1 += float(df.ix[i, j])
                count1 += 1
        mean1 = s1 / count1
        for i in range(0, df.shape[0]):
            if df.ix[i, j] == '?' and df.ix[i, 4] == 2:
                df.ix[i, j] = mean1
        s2 = 0
        count2 = 0
        for i in range(0, df.shape[0]):
            if df.ix[i, j] != '?' and df.ix[i, 4] == 3:
                s2 += float(df.ix[i, j])
                count2 += 1
        mean2 = s2 / count2
        for i in range(0, df.shape[0]):
            if df.ix[i, j] == '?' and df.ix[i, 4] == 3:
                df.ix[i, j] = mean2
        name = path + '.processed.csv'
        df.to_csv(name, header=None, index=None)

def justfyprocess(path):
    df = pd.read_csv(path, header=None)
    if is_number(df.ix[0,0]):
        process2(path)
    else:
        process(path)



def main(path1, path2):
    justfyprocess(path1)
    justfyprocess(path2)

argv1 = sys.argv[1]
argv2 = sys.argv[2]

main(argv1, argv2)
#main('crx.data.testing', 'crx.data.training')
#main('lenses.testing', 'lenses.training')
