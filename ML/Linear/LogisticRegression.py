#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: LogisticRegression.py
@time: 2018/09/08 09:46
@description:
a simple logistic regression with sgd
"""
import numpy as np
import matplotlib.pyplot as plt

N = 100

def generate_data():
    global N
    positive_data = np.random.uniform(0,1,(2,N))
    negative_data = np.random.uniform(3,4,(2,N))
    labels = np.array([1]*N + [0]*N,dtype=np.float32).reshape(1,-1)
    data = np.concatenate((positive_data,negative_data),1)
    data = np.concatenate((data,labels),0)
    data = data.T
    np.random.shuffle(data)
    return data.T

def sigmoid(x,w):
    return 1/(1+np.exp(-(w.T).dot(x)))



def lr_with_sgd():
    global N
    lr = 0.01
    iterations = 500
    count = 0
    eps = 0.01
    reg = 10
    w = np.array([1,1,3],dtype=np.float32).reshape(-1,1)
    data = generate_data()
    for i in range(iterations):
        r = np.random.choice(2*N)
        x = data[:2,r]
        label = data[2,r]
        x = np.concatenate((x,np.array([1])),0).reshape(-1,1)
        dw = (label-sigmoid(x,w))*x + reg*w
        if np.sqrt(((lr*dw)**2).sum()) < eps:
            break
        w -= lr*dw
        count += 1

    print("lr with sgd iterates %d iterations!" % count)
    return w

def lr_with_gd():
    global N
    lr = 0.01
    iterations = 100
    count = 0
    eps = 0.01
    reg = 10
    w = np.array([1,1,3],dtype=np.float32).reshape(-1,1)
    data = generate_data()
    for i in range(iterations):
        dw = 0
        for r in range(2*N):
            x = data[:2,r]
            label = data[2,r]
            x = np.concatenate((x,np.array([1])),0).reshape(-1,1)
            dw += (label-sigmoid(x,w))*x
        dw += reg*w
        if np.sqrt(((lr*dw)**2).sum()) < eps:
            break
        w -= lr*dw
        count += 1
    print("lr with gd iterates %d iterations!" % count)
    return w

def cal_accuracy(data,w,eps = 0.5):
    num = data.shape[1]
    count = 0
    for i in range(num):
        x = np.concatenate((data[:2,i],np.array([1])),0).reshape(-1,1)
        label = data[2,i]
        if label == 1:
            if sigmoid(x,w) >= eps:
                count += 1
        else:
            if sigmoid(x,w) < eps:
                count += 1
    return count/num


if __name__ == '__main__':
    w = lr_with_sgd()
    data = generate_data()
    print("w:",w)
    plt.plot(data[0,:],data[1,:],'r.')
    x0,x1 = -1,5
    y0,y1 = (-w[2,0]-w[0,0]*x0)/w[1,0],(-w[2,0]-w[0,0]*x1)/w[1,0]
    plt.plot([x0,x1],[y0,y1],'d-')
    plt.show()
    print("accuracy:",cal_accuracy(data,w))



