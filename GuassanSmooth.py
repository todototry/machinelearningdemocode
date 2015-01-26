# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 22:45:05 2013

@author: fandyst
"""
import scipy.stats as sta
import numpy as np

def wfit(num,r):
    ww = []
    for i in range(num):
        ww.append(sta.norm.pdf(i,num/2,r))
    return ww


def adjustGus(num,w,yy):
    for i in range(num/2,len(yy)-num/2):
        yy[i] = yy[i-num/2:i+num/2+1] * np.asmatrix(w).T
    return yy


y = [6, 13, 16, 43, 43]
x = np.linspace(0,100,num=5)

y0 = np.linspace(y[0],y[1],25)

y1 = np.linspace(y[1],y[2],25)

y2 = np.linspace(y[2],y[3],25)

y3 = np.linspace(y[3],y[4],25)

yy = y0
yy = yy.tolist()
yy.extend(y1)
yy.extend(y2)
yy.extend(y3)

ylist = np.copy(yy)


w = wfit(9,2)
print w,sum(w)
zz = adjustGus(9,w,ylist)
plot(range(100),yy,color='red')
print "new"
plot(range(100),zz,color='green')


