import scipy.io
import numpy as np
import random
import time
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import floor
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt

mat = scipy.io.loadmat('monkeydata_training.mat')
trial = mat['trial']

def flatten(x):
    return [item for sublist in x for item in sublist]

def rsmeXY(pre_flatx,pre_flaty,flat_x,flat_y):
    squared_numbersx = [number ** 2 for number in pre_flatx]
    squared_numbersy = [number ** 2 for number in pre_flaty]
    sum_list1 = [a + b for a, b in zip(squared_numbersx, squared_numbersy)]
    s11 = np.sqrt(sum_list1)

    fsquared_numbersx = [number ** 2 for number in flat_x]
    fsquared_numbersy = [number ** 2 for number in flat_y]
    sum_list2 = [a + b for a, b in zip(fsquared_numbersx, fsquared_numbersy)]
    s22 = np.sqrt(sum_list2)

    rmsall = sqrt(mean_squared_error(s11, s22))
    return rmsall


# to times
bin = 20

for label in range(trial.shape[1]):
    time = []
    x = []
    y = []
    for ind in range(trial.shape[0]):
        timeLength = trial[ind][label][1].shape[1]
        xVal = trial[ind][label][2][0].tolist()
        yVal = trial[ind][label][2][1].tolist()
        #print(timeLength)
        if timeLength>640:
            timeLength = 640
        if timeLength< 640:
            lenX = len(xVal)
            num_of_missing = 640 -lenX

            last_pointX = xVal[lenX-1]
            list_of_last_eleX = [last_pointX] * num_of_missing
            xVal = xVal + list_of_last_eleX

            lenY = len(yVal)
            last_pointY = yVal[lenY-1]
            list_of_last_eleY = [last_pointY] * num_of_missing
            yVal = yVal + list_of_last_eleY
            #print(len(xVal))
            timeLength = 640

        tm = []
        xx = []
        yy = []

        for timeFlag in range(floor(timeLength/bin)):
            binTime = bin*timeFlag
            tm.append(binTime)
            xx.append(xVal[binTime])
            yy.append(yVal[binTime])
        time.append(tm)
        x.append(xx)
        y.append(yy)



    #print(time[0])

    lr_x0 = linear_model.LinearRegression()
    lr_x0.fit(time, x)
    lr_y0 = linear_model.LinearRegression()
    lr_y0.fit(time, y)
    #
    predictX = lr_x0.predict(time)
    predictY = lr_y0.predict(time)

    flat_x = flatten(x)
    pre_flatx = flatten(predictX)
    flat_y = flatten(y)
    pre_flaty = flatten(predictY)

    rmseall = rsmeXY(pre_flatx,pre_flaty,flat_x,flat_y)
    print("rmse xy :", rmseall)

    plt.scatter(pre_flatx,pre_flaty, s = 7, label = "predict")
    plt.scatter(flat_x,flat_y,alpha = 0.4, s = 0.5, label = "true")
    plt.legend()
    plt.show()
    plt.close()
