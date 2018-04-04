import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import scipy.io
import numpy as np
import itertools
from collections import OrderedDict
from pandas_ml import ConfusionMatrix
from Function import *

mat = scipy.io.loadmat('MNIST_data.mat')
X_train = np.array(mat['train_samples'])
y_train = np.array(mat['train_samples_labels']).reshape((mat['train_samples_labels'].shape[0],))

X_test = np.array(mat['test_samples'])
y_test = np.array(mat['test_samples_labels']).reshape((mat['test_samples_labels'].shape[0],))
def data_cluster(X_train, y_train):
    #set label from 0 to 9
    X_train0 = []
    X_train1 = []
    X_train2 = []
    X_train3 = []
    X_train4 = []
    X_train5 = []
    X_train6 = []
    X_train7 = []
    X_train8 = []
    X_train9 = []

    for i in range(X_train.shape[0]):
        if y_train[i] == 0:
            X_train0.append(X_train[i])
        elif y_train[i] == 1:
            X_train1.append(X_train[i])
        elif y_train[i] == 2:
            X_train2.append(X_train[i])
        elif y_train[i] == 3:
            X_train3.append(X_train[i])
        elif y_train[i] == 4:
            X_train4.append(X_train[i])
        elif y_train[i] == 5:
            X_train5.append(X_train[i])
        elif y_train[i] == 6:
            X_train6.append(X_train[i])
        elif y_train[i] == 7:
            X_train7.append(X_train[i])
        elif y_train[i] == 8:
            X_train8.append(X_train[i])
        elif y_train[i] == 9:
            X_train9.append(X_train[i])
    return np.array(X_train0), np.array(X_train1), np.array(X_train2), np.array(X_train3), np.array(X_train4), np.array(X_train5), np.array(X_train6), np.array(X_train7), np.array(X_train8), np.array(X_train9)

def generate_data(X1, X2):
    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2)) * -1
    return y1 , y2

def transform(predict, plus, minus):
    for i in range(predict.shape[0]):
        if predict[i] == 1 :
            predict[i] = plus
        elif predict[i] == -1 :
            predict[i] = minus

    return predict


def decision_tree(classes, data_values ):

    number = classes
    combination = list(itertools.combinations(classes, 2))


    if len(combination) > 1 :

        if data_values[combination[0]] < 0 :
            number.pop(0)
            return decision_tree(number, data_values)

        else:
            number.pop(1)
            return decision_tree(number, data_values)

    elif len(combination) == 1 :

        if data_values[combination[0]] < 0:
            return combination[0][1]

        else:
            return combination[0][0]



def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=6):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def dag():

    X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9 = data_cluster(X_train, y_train)


    prediction = []
    numpy_list = []

    numpy_predict = [X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9]

    combination  = list(itertools.combinations([0,1,2,3,4,5,6,7,8,9], 2))


    for pair in combination:

        y1, y2 = generate_data(numpy_predict[pair[0]], numpy_predict[pair[1]])

        training_data = np.vstack((numpy_predict[pair[0]] , numpy_predict[pair[1]]))
        test_data = np.hstack((y1, y2))

        clf = SVM(C=0.1)
        clf.train(training_data, test_data)

        y_predict = clf.compute(X_test)
        numpy_list.append(y_predict)
    numpy_list = np.array(numpy_list)
    transpose = np.transpose(numpy_list)
    mix = list(itertools.combinations([0,1,2,3,4,5,6,7,8,9], 2))
    for row in transpose:
        newdict = {}
        for i in range(len(mix)):
            newdict[mix[i]] = row[i]
        result = decision_tree([0,1,2,3,4,5,6,7,8,9], newdict)
        prediction.append(result)
    prediction = np.array(prediction)
    correct = np.sum(prediction == y_test)
    confusion_matrix = ConfusionMatrix(y_test, prediction)
    print("Confusion matrix:\n%s" % confusion_matrix)
    size = len(y_predict)
    accuracy = (correct/float(size)) * 100
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    print ("The accuracy in percentage is  ")
    print(accuracy)

dag()
