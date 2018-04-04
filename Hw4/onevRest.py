import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import scipy.io
import numpy as np
from pandas_ml import ConfusionMatrix
from Function import *
mat = scipy.io.loadmat('MNIST_data.mat')


X_train = np.array(mat['train_samples'])
y_train = np.array(mat['train_samples_labels']).reshape((mat['train_samples_labels'].shape[0],))

X_test = np.array(mat['test_samples'])
y_test = np.array(mat['test_samples_labels']).reshape((mat['test_samples_labels'].shape[0],))


def data_clustering(X_train, y_train):
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


def join_cluster(X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9, number):

    if number == 0:
        X_train_rest = np.vstack((X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train0 = np.ones(len(X_train0))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train0 , X_train_rest, y_train0, y_train_rest

    elif number == 1:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train1 = np.ones(len(X_train1))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train1 , X_train_rest, y_train1, y_train_rest

    elif number == 2:
        X_train_rest = np.vstack((X_train0, X_train1, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train2 = np.ones(len(X_train2))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train2 , X_train_rest, y_train2, y_train_rest

    elif number == 3:
        X_train_rest = np.vstack((X_train0, X_train2, X_train1, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train3 = np.ones(len(X_train3))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train3 , X_train_rest, y_train3, y_train_rest

    elif number == 4:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train1, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train4 = np.ones(len(X_train4))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train4 , X_train_rest, y_train4, y_train_rest

    elif number == 5:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train1, X_train6, X_train7, X_train8, X_train9))
        y_train5 = np.ones(len(X_train5))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train5 , X_train_rest, y_train5, y_train_rest

    elif number == 6:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train1, X_train7, X_train8, X_train9))
        y_train6 = np.ones(len(X_train6))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train6 , X_train_rest, y_train6, y_train_rest

    elif number == 7:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train6, X_train1, X_train8, X_train9))
        y_train7 = np.ones(len(X_train7))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train7 , X_train_rest, y_train7, y_train_rest

    elif number == 8:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train1, X_train9))
        y_train8 = np.ones(len(X_train8))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train8 , X_train_rest, y_train8, y_train_rest

    elif number == 9:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train1))
        y_train9 = np.ones(len(X_train9))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train9 , X_train_rest, y_train9, y_train_rest



def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=6):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def onevRest():

    X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9 = data_clustering(X_train, y_train)

    numpy_predict = []


    for number in range(10):

        train_number, train_rest, test_number, test_rest = join_cluster(X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9, number)

        training_data = np.vstack((train_number, train_rest))
        test_data = np.hstack((test_number, test_rest))

        clf = SVM(C=0.1)
        clf.train(training_data, test_data)

        y_predict = clf.compute(X_test)
        numpy_predict.append(y_predict)



    prediction = np.argmax(np.array(numpy_predict), axis = 0 )

    correct = np.sum(prediction == y_test)

    confusion_matrix = ConfusionMatrix(y_test, prediction)
    print("Confusion matrix:\n%s" % confusion_matrix)

    size = len(y_predict)
    accuracy = (correct/float(size)) * 100

    print ("%d out of %d predictions correct" % (correct, len(y_predict)))
    print ("The accuracy in percentage is  ")
    print(accuracy)


onevRest()
