#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: simpleBP.py
@time: 2018/09/10 23:21
@description:

"""
import numpy as np


def func(x):
    return np.sin(x[0]) + np.cos(x[1]) + np.exp(x[0]+x[1]) + 1

def multiply(w,x):
    return w.dot(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def generate_data(N = 100):
    data = np.random.random((2,N))
    labels = np.zeros((1,N))
    for i in range(N):
        labels[0,i] = func(data[:,i])

    data = np.concatenate((data,labels),0)
    return data

def simpleBP(data,input_shape = 2,lr = 0.01,iterations = 100,
             hidden_units = (5,3,1)):
    x,labels = data[:-1,:],data[-1,:]
    W1 = np.random.random((hidden_units[0],input_shape))
    b1 = np.random.random((hidden_units[0],1))
    W2 = np.random.random((hidden_units[1],hidden_units[0]))
    b2 = np.random.random((hidden_units[1],1))
    W3 = np.random.random((hidden_units[2],hidden_units[1]))
    b3 = np.random.random((hidden_units[2],1))
    variables_index = [(W1.shape[0],W1.shape[1]),(b1.shape[0],1),
                       (W2.shape[0],W2.shape[1]),(b2.shape[0],1),
                       (W3.shape[0], W3.shape[1]), (b3.shape[0], 1)
                       ]
    variables_num = sum([x[0]*x[1] for x in variables_index])
    variables_list = np.random.uniform(0,5,variables_num)
    losses = []
    transformed_variables = transform(variables_list, variables_index)
    for i in range(1,iterations+1):
        if i % 100 == 0:
            lr /= 2
        grads = gradient(ann_func,x,variables_list,variables_index,labels)
        variables_list -= lr*grads
        transformed_variables = transform(variables_list,variables_index)
        y_predict = ann_func(x,*transformed_variables)
        loss = cal_loss(y_predict,labels)
        losses.append(loss)
    return transformed_variables,losses


def cal_loss(predict,labels):
    return np.sqrt(np.square(predict-labels).mean())


def gradient(ann_func,x,variables_list,variables_index,labels,delta =0.01):
    grads = np.zeros_like(variables_list)
    variables_num = len(variables_list)
    for i in range(variables_num):
        variables_list_delta = np.copy(variables_list)
        variables_list_delta[i] += delta
        y = ann_func(x,*transform(variables_list,variables_index))
        y1 = ann_func(x,*transform(variables_list_delta,variables_index))
        grad = (cal_loss(y1,labels) -cal_loss(y,labels))/delta
        grads[i] = grad
    return grads


def transform(variables_list,variables_index):
    variables = []
    index = 0
    for shape in variables_index:
        size = shape[0]*shape[1]
        variables.append(variables_list[index:index+size].reshape(shape))
        index += size
    return variables



def ann_func(x,W1,b1,W2,b2,W3,b3):
    y1 = sigmoid(W1.dot(x)+b1)
    y2 = sigmoid(W2.dot(y1)+b2)
    y3 = W3.dot(y2)+b3
    return y3






if __name__ == '__main__':
    data = generate_data()
    transformed_variables, losses = simpleBP(data,iterations=2000,lr=2)
    for loss in losses:
        print(loss)

