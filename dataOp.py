#coding=utf-8
'''
***** dataOp *****
author : zsz
email  : zhengsz@pku.edu.cn
last modify: 2017-11-26
description: operations on data
copyright: the codes following are free, hope these codes can be helpful
***** ******** *****
'''

import numpy as np

def fetchData(filename, x_beg, x_size, y_beg, y_size):
    raw = np.loadtxt(filename)
    x = raw[:,x_beg:x_beg + x_size]
    y = raw[:,y_beg:y_beg + y_size]
    ave = np.average(y)
    y = [[1] if p[0] > ave else [-1] for p in y]
    y = np.array(y)
    return x,y