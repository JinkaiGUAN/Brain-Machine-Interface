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


def rsmeXY(pre_flatx,pre_flaty,flat_x,flat_y):
    # squared_numbersx = [number ** 2 for number in pre_flatx]
    # squared_numbersy = [number ** 2 for number in pre_flaty]
    # sum_list1 = [a + b for a, b in zip(squared_numbersx, squared_numbersy)]
    # s11 = np.sqrt(sum_list1)
    #
    # fsquared_numbersx = [number ** 2 for number in flat_x]
    # fsquared_numbersy = [number ** 2 for number in flat_y]
    # sum_list2 = [a + b for a, b in zip(fsquared_numbersx, fsquared_numbersy)]
    # s22 = np.sqrt(sum_list2)
    #
    # rmsall = sqrt(mean_squared_error(s11, s22))
    #return rmsall

    pre_flatx = np.array(pre_flatx)
    pre_flaty = np.array(pre_flaty)
    flat_x = np.array(flat_x)
    flat_y = np.array(flat_y)

    x_error = pre_flatx - flat_x
    y_error = pre_flaty - flat_y

    squared_numbersx = np.square(x_error)
    squared_numbersy = np.square(y_error)

    rmse = np.sqrt((np.sum(squared_numbersx) + np.sum(squared_numbersy))/len(flat_x))
    return rmse




# ------------------方法2
bin = 20
ss = []
ll = []
xx = []
yy = []

foookarman = []

for label in range(trial.shape[1]):
    s1 = []
    l1 = []
    x1 = []
    y1 = []
    x11 = []
    y11 = []
    for ind in range(trial.shape[0]):
        spikes = trial[ind][label][1]
        xVal = trial[ind][label][2][0]
        yVal = trial[ind][label][2][1]
        length = spikes.shape[1] - 300

        x11.append(xVal[0])
        y11.append(yVal[0])

        foo = 0
        for bar in range(floor(length/bin)):

            binStart = bar * bin
            binEnd = 300 + bar * bin

            spikes_bin = spikes[:,binStart:binEnd]
            sum_spike = np.sum(spikes_bin,axis=1)
            x = xVal[binEnd]
            y = yVal[binEnd]
            s1.append(sum_spike)
            x1.append(x)
            y1.append(y)
            l1.append(label)


            foo += 1

        foookarman.append(foo)


    ss.append(s1)
    xx.append(x1)
    yy.append(y1)
    ll.append(l1)
#linear regression linear_model
#0
spike_train0 = ss[0]
x_train0 = xx[0]
y_train0 = yy[0]

lr_x0 = linear_model.LinearRegression()
lr_x0.fit(spike_train0, x_train0)
lr_y0 = linear_model.LinearRegression()
lr_y0.fit(spike_train0, y_train0)


#linear regression linear_model
#1
spike_train1 = ss[1]
x_train1 = xx[1]
y_train1 = yy[1]

lr_x1 = linear_model.LinearRegression()
lr_x1.fit(spike_train1, x_train1)
lr_y1 = linear_model.LinearRegression()
lr_y1.fit(spike_train1, y_train1)

#linear regression linear_model
#2
spike_train2 = ss[2]
x_train2 = xx[2]
y_train2 = yy[2]

lr_x2 = linear_model.LinearRegression()
lr_x2.fit(spike_train2, x_train2)
lr_y2 = linear_model.LinearRegression()
lr_y2.fit(spike_train2, y_train2)

#linear regression linear_model
#3
spike_train3 = ss[3]
x_train3 = xx[3]
y_train3 = yy[3]

lr_x3 = linear_model.LinearRegression()
lr_x3.fit(spike_train3, x_train3)
lr_y3 = linear_model.LinearRegression()
lr_y3.fit(spike_train3, y_train3)

#linear regression linear_model
#4
spike_train4 = ss[4]
x_train4 = xx[4]
y_train4 = yy[4]

lr_x4 = linear_model.LinearRegression()
lr_x4.fit(spike_train4, x_train4)
lr_y4 = linear_model.LinearRegression()
lr_y4.fit(spike_train4, y_train4)

#linear regression linear_model
#5
spike_train5 = ss[5]
x_train5 = xx[5]
y_train5 =yy[5]

lr_x5 = linear_model.LinearRegression()
lr_x5.fit(spike_train5, x_train5)
lr_y5 = linear_model.LinearRegression()
lr_y5.fit(spike_train5, y_train5)

#linear regression linear_model
#6
spike_train6 = ss[6]
x_train6 = xx[6]
y_train6 = yy[6]

lr_x6 = linear_model.LinearRegression()
lr_x6.fit(spike_train6, x_train6)
lr_y6 = linear_model.LinearRegression()
lr_y6.fit(spike_train6, y_train6)

#linear regression linear_model
#7
spike_train7 = ss[7]
x_train7 = xx[7]
y_train7 = yy[7]

lr_x7 = linear_model.LinearRegression()
lr_x7.fit(spike_train7, x_train7)
lr_y7 = linear_model.LinearRegression()
lr_y7.fit(spike_train7, y_train7)



testx = lr_x0.predict(ss[0])
testy = lr_y0.predict(ss[0])

predx = []
predy = []

flat_label = [item for sublist in ll for item in sublist]
flat_x= [item for sublist in xx for item in sublist]
flat_y = [item for sublist in yy for item in sublist]
flat_spike_data = [item for sublist in ss for item in sublist]

for ind in range(len(flat_label)):
    label = flat_label[ind]
    if label == 0:
        testx = lr_x0.predict([flat_spike_data[ind]])
        testy = lr_y0.predict([flat_spike_data[ind]])
        predx.append(testx)
        predy.append(testy)
    if label == 1:
        testx = lr_x1.predict([flat_spike_data[ind]])
        testy = lr_y1.predict([flat_spike_data[ind]])
        predx.append(testx)
        predy.append(testy)
    if label == 2:
        testx = lr_x2.predict([flat_spike_data[ind]])
        testy = lr_y2.predict([flat_spike_data[ind]])
        predx.append(testx)
        predy.append(testy)
    if label == 3:
        testx = lr_x3.predict([flat_spike_data[ind]])
        testy = lr_y3.predict([flat_spike_data[ind]])
        predx.append(testx)
        predy.append(testy)
    if label == 4:
        testx = lr_x4.predict([flat_spike_data[ind]])
        testy = lr_y4.predict([flat_spike_data[ind]])
        predx.append(testx)
        predy.append(testy)
    if label == 5:
        testx = lr_x5.predict([flat_spike_data[ind]])
        testy = lr_y5.predict([flat_spike_data[ind]])
        predx.append(testx)
        predy.append(testy)
    if label == 6:
        testx = lr_x6.predict([flat_spike_data[ind]])
        testy = lr_y6.predict([flat_spike_data[ind]])
        predx.append(testx)
        predy.append(testy)
    if label == 7:
        testx = lr_x7.predict([flat_spike_data[ind]])
        testy = lr_y7.predict([flat_spike_data[ind]])
        predx.append(testx)
        predy.append(testy)

#---------------------calculate rmse
print(len(predx))
pre_flatx = [item for sublist in predx for item in sublist]
pre_flaty = [item for sublist in predy for item in sublist]

rmsx = sqrt(mean_squared_error(flat_x, pre_flatx))
rmsy = sqrt(mean_squared_error(flat_y, pre_flaty))

rmseall2 = rsmeXY(pre_flatx,pre_flaty,flat_x,flat_y)

print('-------')
print(rmseall2)
print('-------')
