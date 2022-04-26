# -*- coding: UTF-8 -*-
"""
@Project : BMI-coursework
@File    : Regression.py
@IDE     : PyCharm
@Author  : Haonan Zhou
@Date    : 18/04/2022 21:11
@Brief   :
"""
import numpy as np
import scipy.io as scio
from sklearn.linear_model import LinearRegression
from base_model import BaseModelRegression


data_path = 'F:/Users/27339/Desktop/IC/Modules/BMI/Brain-Machine-Interface/report_src/monkeydata_training.mat'
short_data_path = './monkeydata_training.mat'


class RegressionData:

    def __init__(self,dataPath: str, label: int, winWidth=300, bin=20) -> None:
        self.data = scio.loadmat(dataPath).get('trial')
        self.winWidth = winWidth
        self.bin = bin
        self.setShape = self.data.shape
        self.label = label
        self.data_fr, self.handPos = self.dataExtract()
        self.mean_fr = np.mean(self.data_fr, axis=1)
        self.std_fr = np.std(self.data_fr, axis=1)
        self.mean_handPos = np.mean(self.handPos, axis=1)
        self.std_handPos = np.std(self.handPos, axis=1)

    def dataExtract(self):
        fr_Data = np.sum(self.data[0, 0][1][:, 300 - self.winWidth: 300], axis=1).reshape([98, 1])
        handPos = self.data[0, 0][2][0:2, 300].reshape([2, 1])

        if self.label == 0:
            for c in range(self.setShape[1]):
                for trail in range(self.setShape[0]):
                    length = self.data[trail, c][2].shape[1]
                    initialPos = self.data[trail, c][2][0:2, 300].reshape([2, 1])
                    for t in range(300, length, self.bin):
                        fr_d = np.sum(self.data[trail, c][1][:, t - self.winWidth: t], axis=1).reshape([98, 1])
                        fr_Data = np.append(fr_Data, fr_d, axis=1)
                        hand_p = self.data[trail, c][2][0:2, t].reshape([2, 1]) - initialPos
                        handPos = np.append(handPos, hand_p, axis=1)
        else:
            c = self.label - 1
            for trail in range(self.setShape[0]):
                length = self.data[trail, c][2].shape[1]
                initialPos = self.data[trail, c][2][0:2, 300].reshape([2, 1])
                for t in range(300, length, self.bin):
                    fr_d = np.sum(self.data[trail, c][1][:, t - self.winWidth: t], axis=1).reshape([98,1])
                    fr_Data = np.append(fr_Data, fr_d, axis=1)
                    hand_p = self.data[trail, c][2][0:2, t].reshape([2, 1]) - initialPos
                    handPos = np.append(handPos, hand_p, axis=1)

        fr_Data = fr_Data[:, 1:]
        handPos = handPos[:, 1:]
        return fr_Data.T, handPos.T

    def getATrail(self, trailIndex, label):
        fr_Data = np.sum(self.data[0, 0][1][:, 300 - self.winWidth: 300], axis=1).reshape([98, 1])
        handPos = self.data[0, 0][2][0:2, 300].reshape([2, 1])

        for c in label - 1:
            for trail in trailIndex:
                length = self.data[trail, c][2].shape[1]
                initialPos = self.data[trail, c][2][0:2, 300].reshape([2, 1])
                for t in range(300, length, self.bin):
                    fr_d = np.sum(self.data[trail, c][1][:, t - self.winWidth: t], axis=1).reshape([98, 1])
                    fr_Data = np.append(fr_Data, fr_d, axis=1)
                    hand_p = self.data[trail, c][2][0:2, t].reshape([2, 1]) - initialPos
                    handPos = np.append(handPos, hand_p, axis=1)

        fr_Data = fr_Data[:, 1:]
        handPos = handPos[:, 1:]
        Spikes = self.data[trailIndex, label-1][1]

        return Spikes.T, fr_Data.T, handPos.T


class RegressionModel(BaseModelRegression):

    def __init__(self, dataPath: str,  winWidth=300, bin=20, approach = 'LinearRegression'):
        self.data = {'1': RegressionData(dataPath, 1, winWidth=winWidth, bin=bin),
                     '2': RegressionData(dataPath, 2, winWidth=winWidth, bin=bin),
                     '3': RegressionData(dataPath, 3, winWidth=winWidth, bin=bin),
                     '4': RegressionData(dataPath, 4, winWidth=winWidth, bin=bin),
                     '5': RegressionData(dataPath, 5, winWidth=winWidth, bin=bin),
                     '6': RegressionData(dataPath, 6, winWidth=winWidth, bin=bin),
                     '7': RegressionData(dataPath, 7, winWidth=winWidth, bin=bin),
                     '8': RegressionData(dataPath, 8, winWidth=winWidth, bin=bin),
                     }
        self.approach = approach
        self.winSize = winWidth
        self.bin = bin
        # self.model = self.set_model()
        if self.approach == 'LinearRegression':
            self.models = {'1': self.set_model(),
                           '2': self.set_model(),
                           '3': self.set_model(),
                           '4': self.set_model(),
                           '5': self.set_model(),
                           '6': self.set_model(),
                           '7': self.set_model(),
                           '8': self.set_model()
                           }

    def set_model(self):
        if self.approach == 'LinearRegression':
            return LinearRegression()

    def fit(self):
        if self.approach == 'LinearRegression':
            for label in range(8):
                l = str(label+1)
                self.models[l].fit(self.data[l].data_fr, self.data[l].handPos)


    def predict(self, spikes: np.ndarray, label: int, initial_position: np.ndarray):
        fireRate = spikes

        if self.approach == 'LinearRegression':
            l = str(label)
            # fireRate = fireRate.reshape([98, 1])
            posPredict = np.array(self.models[l].predict(fireRate))
            return posPredict

    def getFR(self, Spikes):
        fireRate = np.sum(Spikes[- (self.winSize+1): -1, :], axis=0).reshape([98, 1])
        return fireRate.T


if __name__ == '__main__':
    regressionAgent = RegressionModel(short_data_path)
    regressionAgent.fit()

    testSpike = np.ones((320, 98))  # The spike data need to be tested, 320 is an example of the time length
    testFR = regressionAgent.getFR(testSpike) # get the fire rate through Spike data
    # Label = Classification(testSpike) # get the label by input the spike data
    # pre_pos = regressionAgent.predict(regressionAgent.data['1'].data_fr[50:80, :], 1) # predict the position
    pre_pos = regressionAgent.predict(regressionAgent.data['1'].data_fr[50:80, :], 1)
    real_pos = regressionAgent.data['1'].handPos[50:80, :]
    print(pre_pos - real_pos)
    print('Done')






