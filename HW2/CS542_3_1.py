import pandas as pd
import numpy as np
import sys
import copy
import csv
import math

def openfile(path):
	data = []
	file = open(path)
	try:
		array = csv.reader(file)
		for i in array:
			data.append(i)
	finally:
		file.close()
	return data

data_training = openfile('crx.data.training')

data_testing = openfile('crx.data.testing')


def replaceChar(x):
	x -= 1
	list = []
	tongji = []

	for i in range(len(data_training)):
		list.append(data_training[i][x])
	myset = set(list)
	for item in myset:
		tongji.append([item, list.count(item)])
	tongjinum=[]
	tongjiitem=[]
	for i in tongji:
		tongjinum.append(i[1])
		tongjiitem.append(i[0])
	idxb= tongjiitem[tongjinum.index(max(tongjinum))]

	for i in range(len(data_training)):
		if data_training[i][x] == '?':
			data_training[i][x] = copy.copy(idxb)

	return data_training

num1 = [1,4,5,6,7,9,10,12,13]
for x in num1:
	replaceChar(x)


def replaceNum():
	num2 = [1,2,7,10,13,14]

	for i in num2:
		counter1 = 0
		counter2 = 0
		counter3 = 0
		counter4 = 0
		diff1 = 0
		diff2 = 0
		he1 = 0
		he2 = 0
		for j in range(len(data_training)):
			if data_training[j][i] == '?':
				if data_training[j][15] == '+':
					counter1 += 1
					data_training[j][i] = 12345
				elif data_training[j][15] == '-':
					counter2 += 1
					data_training[j][i] = 54321
			if data_training[j][15] == '+':
				counter3 += 1
			elif data_training[j][15] == '-':
				counter4 += 1
		for j in range(len(data_training)):
			if data_training[j][15] == '+':
				he1 = he1 + float(data_training[j][i])
			elif data_training[j][15] == '-':
				he2 = he2 + float(data_training[j][i])
		he1 = he1-12345*counter1
		he2 = he2 -54321*counter2
		mean1 = he1 / (counter3 - counter1)
		mean2 = he2 / (counter4 - counter2)
		for k in range(len(data_training)):
			if data_training[k][i] == 12345:
				data_training[k][i] = copy.copy(mean1)
			elif data_training[k][i] == 54321:
				data_training[k][i] = copy.copy(mean2)
		for j in range(len(data_training)):
			if data_training[j][15] == '+':
				diff1 = diff1 + pow((float(data_training[j][i]) - mean1), 2)
			elif data_training[j][15] == '-':
				diff2 = diff2 + pow((float(data_training[j][i]) - mean2), 2)
		sigma1 = math.sqrt(diff1 / counter3)
		sigma2 = math.sqrt(diff2 / counter4)
		for j in range(len(data_training)):
			if data_training[j][15] == '+':
				data_training[j][i] = (float(data_training[j][i]) - mean1) / sigma1
			elif data_training[j][15] == '-':
				data_training[j][i] = (float(data_training[j][i]) - mean2) / sigma2

	return data_training

replaceNum()


for i in data_training:
	print(i)
