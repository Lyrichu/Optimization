#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: newton.py
@time: 2018/09/06 10:30
@description:
newton method,reference:https://blog.csdn.net/itplus/article/details/21896453
f(x1,x2) = (x1-2)^2 + (x1*x2-4)^2 + 3
"""
import numpy as np
from scipy import linalg


def func(x):
    x1, x2 = x[0][0], x[1][0]
    return (x1-2)**2 +(x1*x2-4)**2 + 3

def cal_df(x):
    x1, x2 = x[0][0], x[1][0]
    df_x1 = 2*(x1-2) + 2*(x1*x2-4)*x2
    df_x2 = 2*(x1*x2-4)*x1
    return np.array([df_x1,df_x2],dtype=np.float32).reshape(2,1)

def cal_hessian(x):
    x1, x2 = x[0][0],x[1][0]
    hessian = np.zeros((2,2))
    hessian[0][0] = 2+2*x2**2
    hessian[0][1] = -8 + 4*x1*x2
    hessian[1][0] = -8 + 4*x1*x2
    hessian[1][1] = 2*x1**2
    return hessian

def find_best_step(x,d,min_step = -1,max_step = 1,size = 0.05):
    best_step = min_step
    best_res = func(x-best_step*d)
    for step in np.arange(min_step,max_step,size):
        cur_res = func(x-step*d)
        if cur_res < best_res:
            best_res = cur_res
            best_step = step
    return best_step




def newton_sulotion():
    x = np.random.uniform(1.5,2,(2,1))
    steps = 100
    eps = 0.001
    iteration = 0
    for i in range(steps):
        g = cal_df(x)
        if np.sqrt((g**2).sum()) < eps:
            break
        d = linalg.inv(cal_hessian(x)).dot(g)
        x -= d
        iteration += 1
    print("newton_sulotion iteration num:",iteration)
    return x

def damping_newton():
    x = np.random.uniform(0,1, (2, 1))
    steps = 100
    eps = 0.001
    iteration = 0
    for i in range(steps):
        g = cal_df(x)
        if np.sqrt((g ** 2).sum()) < eps:
            break
        d = linalg.inv(cal_hessian(x)).dot(g)
        best_step = find_best_step(x, d)
        x -= best_step*d
        iteration += 1
    print("damping_newton iteration num:", iteration)
    return x

def dfp():
    x = np.random.uniform(-1, 1, (2, 1))
    eps = 0.001
    D = np.eye(2,2)
    steps = 100
    iteration = 0
    for i in range(steps):
        g1 = cal_df(x)
        d = D.dot(g1) # D是对hessain矩阵逆矩阵的近似
        best_step = find_best_step(x,d)
        s = best_step*d
        x -= s
        iteration += 1
        g2 = cal_df(x)
        if np.sqrt((g2**2).sum()) <eps:
            break
        y = g2 -g1
        # D的更新
        D += s.dot(s.T)/((s.T).dot(y)) - D.dot(y).dot(y.T).dot(D)/((y.T).dot(D).dot(y))
    print("dfp iteration num:",iteration)
    return x

def bfgs():
    x = np.random.uniform(-1, 1, (2, 1))
    eps = 0.001
    B = np.eye(2, 2)
    steps = 100
    iteration = 0
    for i in range(steps):
        g1 = cal_df(x)
        d = linalg.inv(B).dot(g1)  # D是对hessain矩阵逆矩阵的近似
        best_step = find_best_step(x, d)
        s = best_step * d
        x -= s
        iteration += 1
        g2 = cal_df(x)
        if np.sqrt((g2 ** 2).sum()) < eps:
            break
        y = g2 - g1
        # B的更新
        B += y.dot(y.T)/((y.T).dot(s)) - (B.dot(s).dot(s.T).dot(B))/(s.T.dot(B).dot(s))
    print("bfgs iteration num:", iteration)
    return x











if __name__ == '__main__':
    x = newton_sulotion()
    print("newton_solution:",x)
    x = damping_newton()
    print("damping_newton:",x)
    x = dfp()
    print("dfp:",x)
    x = bfgs()
    print("bfgs:", x)









