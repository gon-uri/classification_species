# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:29:02 2017

lee datos en un file ïnput.txt¨
0,0,0,0,0,0,0,0,1,0
0,0,0,0,0,0,0,0,1,1
0,0,0,0,0,0,0,1,0,0
0,0,0,0,0,0,0,1,0,1
2,0,2,1,0,2,0,0,0,0
2,0,2,1,1,2,2,0,0,1
1,0,1,2,2,1,1,0,0,2
1,0,1,1,1,1,1,0,1,1

@author: gabrielmindlin
"""
import numpy as np
from sklearn import svm

fin = open('input.txt','r')
X=[]
for line in fin.readlines():
    X.append( [ float (x) for x in line.split(',') ] )

y = [0, 0, 0, 0, 2, 2, 1, 1]

clf = svm.SVC()
clf.fit(X, y) 

print(clf.predict([[0.2, 1.1, 0, 1, 1, 0, 1, 1,2,1]]))