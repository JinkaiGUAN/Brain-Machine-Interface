# -*-coding = utf-8 -*-
# @Time : 27/04/2022 17:59
# @Author : ZHONGJIE ZHANG
# @File :linear.py
# @Software:PyCharm
import scipy.io
import numpy as np
from numpy.linalg import pinv
import random
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import floor
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
window_size = 300
step = 20
mat = scipy.io.loadmat('monkeydata_training.mat')
def data(mat,start=0,end=51):
    trial = mat['trial']
    single_angle_fr = {}
    single_angle_position_xy = {}
    for angle_inx in range(trial.shape[1]):
        single_angle_fr[angle_inx] = np.zeros([99,1])
        single_angle_position_xy[angle_inx] = np.zeros([2,1])
        for trial_inx in range(start,end):
            trial_length = trial[trial_inx][angle_inx][1].shape[1]
            step_number = len(range(320,trial_length,step))
            fire_rate = np.zeros([99,step_number])
            position_xy = np.zeros([2,step_number])
            for time_inx in range(step_number):
                real_time = time_inx*step + 320
                fire_rate[:-1,time_inx] = np.sum(trial[trial_inx][angle_inx][1][:,real_time-window_size:real_time],
                                   axis=1)
                fire_rate[-1, time_inx] = time_inx
                position_xy[0,time_inx] = trial[trial_inx][angle_inx][2][0][real_time]
                position_xy[1, time_inx] = trial[trial_inx][angle_inx][2][1][real_time]
            single_angle_fr[angle_inx] = np.concatenate((single_angle_fr[angle_inx],fire_rate),axis = 1)
            single_angle_position_xy[angle_inx] = np.concatenate((single_angle_position_xy[angle_inx],
                                                              position_xy),
                                                             axis=1)
        single_angle_fr[angle_inx] = single_angle_fr[angle_inx][:, 1:]
        single_angle_position_xy[angle_inx] = single_angle_position_xy[angle_inx][:, 1:]
    return single_angle_fr,single_angle_position_xy
def linear_fit(X,y):
    print(X.shape,y.shape)
    return ((pinv((X.T).dot(X))).dot(X.T)).dot(y)

def linear_predict(B,X):
    return X.dot(B)

train_fr,train_xy = data(mat)
test_fr,test_xy = data(mat,start=51,end=100)
model = {}
pre_xy ={}
for label in range(8):
    model[label] = LinearRegression()
    model[label].fit(train_fr[label].T,train_xy[label].T)
    #model[label] = linear_fit(train_fr[label].T,train_xy[label].T)
    #pre_xy[label] = linear_predict(model[label],test_fr[label].T)
    pre_xy[label] = model[label].predict(test_fr[label].T)

    #print(error.shape)
    #print(train_xy[label][1,0:20])
    plt.plot(pre_xy[label][:,0],pre_xy[label][:,1])
plt.show()
