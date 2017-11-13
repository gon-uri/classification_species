# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:29:02 2017

@author: gabrielmindlin
"""

from sklearn import svm

X = [[0, 0], [1, 1], [2,3], [3.1,3.8]]
y = [0, 1, 2, 3]

clf = svm.SVC()
clf.fit(X, y) 

print(clf.predict([[5.2, 5.1]]))