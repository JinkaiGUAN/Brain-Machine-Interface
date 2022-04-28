# -*-coding = utf-8 -*-
# @Time : 26/04/2022 16:48
# @Author : ZHONGJIE ZHANG
# @File :KalmanRegression.py
# @Software:PyCharm
# -*-coding = utf-8 -*-
# @Time : 25/04/2022 16:29
# @Author : ZHONGJIE ZHANG
# @File :KalmanRegression.py
# @Software:PyCharm
import numpy as np
import scipy.io as scio
from sklearn.linear_model import LinearRegression
import typing as t


data_path = 'F:/Users/27339/Desktop/IC/Modules/BMI/Brain-Machine-Interface/report_src/monkeydata_training.mat'
short_data_path = './monkeydata_training.mat'

class RegressionData:

    def __init__(self,dataPath: t.Union[str, np.ndarray], label: int, winWidth=300, bin=20) -> None:
        # self.data = scio.loadmat(dataPath).get('trial')
        self.data = dataPath
        self.winWidth = winWidth
        self.bin = bin
        self.setShape = self.data.shape
        self.label = label
        self.data_fr, self.handPos = self.dataExtract()
        self.fr_Data, self.all_hand_states = self.Kalmandata()
        self.mean_fr = np.mean(self.data_fr, axis=1)
        self.std_fr = np.std(self.data_fr, axis=1)
        self.mean_handPos = np.mean(self.handPos, axis=1)
        self.std_handPos = np.std(self.handPos, axis=1)

    def dataExtract(self):
        data_fr = np.sum(self.data[0, 0][1][:, 300 - self.winWidth: 300], axis=1).reshape([98, 1])
        handPos = self.data[0, 0][2][0:2, 300].reshape([2, 1])

        if self.label == 0:
            for c in range(self.setShape[1]):
                for trail in range(self.setShape[0]):
                    length = self.data[trail, c][2].shape[1]
                    initialPos = self.data[trail, c][2][0:2, 300].reshape([2, 1])
                    for t in range(300, length, self.bin):
                        fr_d = np.sum(self.data[trail, c][1][:, t - self.winWidth: t], axis=1).reshape([98, 1])
                        fr_Data = np.append(data_fr, fr_d, axis=1)
                        hand_p = self.data[trail, c][2][0:2, t].reshape([2, 1]) - initialPos
                        handPos = np.append(handPos, hand_p, axis=1)
        else:
            c = self.label - 1
            for trail in range(self.setShape[0]):
                length = self.data[trail, c][2].shape[1]
                initialPos = self.data[trail, c][2][0:2, 0].reshape([2, 1])
                for t in range(300, length, self.bin):
                    fr_d = np.sum(self.data[trail, c][1][:, t - self.winWidth: t], axis=1).reshape([98,1])
                    data_fr = np.append(data_fr, fr_d, axis=1)
                    hand_p = self.data[trail, c][2][0:2, t].reshape([2, 1]) - initialPos
                    handPos = np.append(handPos, hand_p, axis=1)

        data_fr = data_fr[:, 1:]
        handPos = handPos[:, 1:]

        return data_fr.T, handPos.T

    #get the data for kalman filtering
    def Kalmandata(self):
        fr_Data = np.sum(self.data[0, 0][1][:, 300 - self.winWidth: 300], axis=1).reshape([98, 1])
        handPos = self.data[0, 0][2][0:2, 300].reshape([2, 1])
        if self.label == 0:
            for c in range(self.setShape[1]):
                for trail in range(self.setShape[0]):
                    length = self.data[trail, c][2].shape[1]
                    initialPos = self.data[trail, c][2][0:2, 300].reshape([2, 1])
                    bin_number = len(range(300, length, self.bin))
                    for t in range(300, length, self.bin):
                        fr_d = np.sum(self.data[trail, c][1][:, t - self.winWidth: t], axis=1).reshape([98, 1])
                        fr_Data = np.append(fr_Data, fr_d, axis=1)
                        hand_p = self.data[trail, c][2][0:2, t].reshape([2, 1]) - initialPos
                        handPos = np.append(handPos, hand_p, axis=1)
        else:
            c = self.label - 1
            all_hand_states = np.zeros([6,1])
            for trail in range(self.setShape[0]):
                length = self.data[trail, c][2].shape[1]
                initialPos = self.data[trail, c][2][0:2, 300].reshape([2, 1])
                bin_number = len(range(280, length, self.bin))
                hand_p = np.zeros([2,bin_number])
                hand_state = np.zeros([6,bin_number-2])
                x = 0
                for t in range(280, length, self.bin):
                    fr_d = np.sum(self.data[trail, c][1][:, t - self.winWidth: t], axis=1).reshape([98,1])
                    if t > 300:
                        fr_Data = np.append(fr_Data, fr_d, axis=1)
                    hand_p[:,x] = (self.data[trail, c][2][0:2, t].reshape([2, 1]) - initialPos).flatten()
                    x += 1
                hand_state[0:2,:] = hand_p[:,2:]
                hand_v = np.diff(hand_p)
                hand_a = np.diff(hand_v)
                hand_state[2:4,:] = hand_v[:, 1:]
                hand_state[4:6,:] = hand_a
                all_hand_states = np.append(all_hand_states, hand_state, axis=1)
        fr_Data = fr_Data[:, 1:]
        all_hand_states = all_hand_states[:, 1:]
        return fr_Data.T, all_hand_states.T

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

from base_model import BaseModelRegression

class RegressionModel(BaseModelRegression):

    def __init__(self, dataPath: t.Union[str, np.ndarray],  winWidth=300, bin=20, approach = 'LinearRegression'):
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
        self.models = {'1': self.set_model(),
                        '2':self.set_model(),
                        '3': self.set_model(),
                        '4': self.set_model(),
                        '5': self.set_model(),
                        '6': self.set_model(),
                        '7': self.set_model(),
                        '8': self.set_model()
                        }
        self.R = {'1': 0,
                        '2':0,
                        '3': 0,
                        '4': 0,
                        '5': 0,
                        '6': 0,
                        '7': 0,
                        '8': 0
                        }

    def set_model(self):
        if self.approach == 'LinearRegression':
            return LinearRegression()
        elif self.approach == 'KalmanRegression':
            return LinearRegression()

    def Kalman_filter(self,ObsZ, R, nDataPoints):
        tStep = self.bin
        a = np.eye(2)
        A = np.array([[a, a, 0.5*tStep**(2)*a],[np.zeros([2,2]), a, tStep*a],[np.zeros([2,2]),np.zeros([2,2]),a]])
        A = A.reshape([6,6])
        States = np.zeros([6, nDataPoints])
        States[:,0] = np.zeros(6)
        SigmaQ = 0.3
        Q = np.array([[SigmaQ **(6/ 36)  , 0 , SigmaQ **(5 / 12)  , 0 , SigmaQ **(4 / 6) , 0],
             [0, SigmaQ **(6/ 36),  0,  SigmaQ **(5 / 12),  0,  SigmaQ **(4 / 6)],
             [SigmaQ **(5/12),  0,  SigmaQ**(4/4),  0,  SigmaQ**(3/2), 0],
             [0, SigmaQ **(5/12),  0,  SigmaQ**(4/4),  0,  SigmaQ**(3/2)],
             [0, SigmaQ **(5/12),  0,  SigmaQ**(4/4),  0,  SigmaQ**(3/2)],
             [0, SigmaQ **(4 / 6),   0,   SigmaQ**(3/2),  0,  SigmaQ**2]])
        Sigma = np.eye(6) * 1
        C = np.eye(6)
        for t in range(1,nDataPoints):
            StatePrior = np.dot(A,States[:,t-1])
            SigmaPrior = np.dot(np.dot(A,Sigma),A.T)+Q
            K = np.dot(np.linalg.pinv(np.dot(SigmaPrior,C.T)),np.dot(np.dot(C,SigmaPrior),C.T) + R)
            States[:,t] = StatePrior + np.dot(K ,(ObsZ[:,t] - np.dot(C, StatePrior)))
            Sigma = np.eye(6) - np.dot(np.dot(K,C), SigmaPrior)
        PositionFinal = States[0:2,-1].reshape([2,1])
        return PositionFinal

    def fit(self):
        if self.approach == 'LinearRegression':
            for label in range(8):
                l = str(label+1)
                self.models[l].fit(self.data[l].data_fr, self.data[l].handPos)
        elif self.approach == 'KalmanRegression':
            for label in range(8):
                l = str(label+1)
                #print(self.data[l].all_hand_states.shape)
                #print(self.data[l].fr_Data.shape)
                self.models[l].fit(self.data[l].fr_Data,self.data[l].all_hand_states)
                r = self.models[l].predict(self.data[l].fr_Data) - self.data[l].all_hand_states
                self.R[l] = np.cov(r.T)

    def predict(self,spikes: np.ndarray, label: int, initial_position: np.ndarray):
        fireRate = self.getFR(spikes.T)
        if self.approach == 'LinearRegression':
            l = str(label+1)
            # fireRate = fireRate.reshape([98, 1])
            posPredict = np.array(self.models[l].predict(fireRate))
            posPredict = posPredict.T + initial_position
            return posPredict[0].flatten(),posPredict[1].flatten()

        elif self.approach == 'KalmanRegression':
            l = str(label+1)
            # fireRate = fireRate.reshape([98, 1])
            nDataPoints = int((spikes.shape[1] - 280) / 20)
            states = self.Kalman_filter(fireRate,self.R[l],nDataPoints)
            states +=  initial_position
            return states

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