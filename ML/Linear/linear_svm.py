#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: linear_svm.py
@time: 2018/09/08 11:22
@description:
a simple linear svm with SGD
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(N = 100):
    positive_data = np.random.uniform(0,1,(2,N))
    negative_data = np.random.uniform(3,4,(2,N))
    labels = np.array([1]*N + [-1]*N,dtype=np.float32).reshape(1,-1)
    data = np.concatenate((positive_data,negative_data),1)
    data = np.concatenate((data,labels),0)
    data = data.T
    np.random.shuffle(data)
    return data.T

def linear_svm(data,lr = 0.1,reg = 0.5,iterations = 100,eps = 0.1):
    w = np.random.random((3,1))
    data_num = data.shape[1]
    iteration = 0
    for i in range(iterations):
        dw = 0
        for j in range(data_num):
            x = np.concatenate((data[:2,j],np.array([1])),0).reshape(-1,1)
            y = data[2,j]
            if 1-y*((w.T).dot(x)) > 0:
                dw += -y*x
        dw += reg*w
        if np.sqrt(((lr*dw)**2).sum()) < eps:
            break
        w -= lr*dw
        iteration += 1
    print("iterate %d iterations!" % iteration)
    return w


if __name__ == '__main__':
    data = generate_data()
    lr = 0.05
    reg = 0.5
    iterations = 500
    eps = 0.1
    w = linear_svm(data,lr=lr,reg=reg,iterations=iterations,eps=eps)
    print("w:", w)
    plt.plot(data[0, :], data[1, :], 'r.')
    x0, x1 = -1, 5
    y0, y1 = (-w[2, 0] - w[0, 0] * x0) / w[1, 0], (-w[2, 0] - w[0, 0] * x1) / w[1, 0]
    plt.plot([x0, x1], [y0, y1], 'd-')
    plt.show()



