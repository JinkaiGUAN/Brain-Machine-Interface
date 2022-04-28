# -*-coding = utf-8 -*-
# @Time : 27/04/2022 15:12
# @Author : ZHONGJIE ZHANG
# @File :linear_regression.py
# @Software:PyCharm
import typing as t
from collections import defaultdict

import numpy as np
from numpy.linalg import pinv,norm
import scipy.io as scio
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from preprocess import Trial
from base_model import BaseModelRegression


class SingleAngleData:
    def __init__(self, firing_rate, hand_position_x, hand_position_y):
        self.firing_rate = firing_rate
        self.hand_position_x = hand_position_x
        self.hand_position_y = hand_position_y


class RegressionData:
    def __init__(self, data_path: t.Union[np.ndarray, str], bin_width: int = 20, window_width: int = 300):

        if isinstance(data_path, str):
            self.data = scio.loadmat(data_path).get('trial')
        if isinstance(data_path, np.ndarray):
            self.data = data_path
        self.bin_width = bin_width
        self.window_width = window_width

        # retrieve data information
        self.trail_num = self.data.shape[0]
        self.angle_num = self.data.shape[1]
        self.neuro_num = self.data[0, 0][1].shape[0]

    def simple_linear_data(self,normalize : bool = False):
        step = self.bin_width
        window_size = self.window_width
        trial = self.data
        single_angle_fr = {}
        single_angle_position_xy = {}
        normal_fr = {}
        normal_xy = {}
        for angle_inx in range(trial.shape[1]):
            single_angle_fr[angle_inx] = np.zeros([98, 1])
            single_angle_position_xy[angle_inx] = np.zeros([2, 1])
            for trial_inx in range(self.trail_num):
                trial_length = trial[trial_inx][angle_inx][1].shape[1]
                step_number = len(range(320, trial_length, step))
                fire_rate = np.zeros([98, step_number])
                position_xy = np.zeros([2, step_number])
                for time_inx in range(step_number):
                    real_time = time_inx * step + 320
                    fire_rate[:, time_inx] = np.sum(
                        trial[trial_inx][angle_inx][1][:, real_time - window_size:real_time],
                        axis=1)
                    #fire_rate[-1, time_inx] = time_inx
                    position_xy[0, time_inx] = trial[trial_inx][angle_inx][2][0][real_time]- trial[trial_inx][angle_inx][2][0][300]
                    position_xy[1, time_inx] = trial[trial_inx][angle_inx][2][1][real_time] - trial[trial_inx][angle_inx][2][0][300]
                single_angle_fr[angle_inx] = np.concatenate((single_angle_fr[angle_inx], fire_rate), axis=1)
                single_angle_position_xy[angle_inx] = np.concatenate((single_angle_position_xy[angle_inx],
                                                                      position_xy),
                                                                     axis=1)
            single_angle_fr[angle_inx] = single_angle_fr[angle_inx][:,1:]
            single_angle_position_xy[angle_inx] = single_angle_position_xy[angle_inx][:,1:]
            #lda = LDA(n_components=2)
            #lda.fit(single_angle_position_xy[angle_inx], y)
            #single_angle_position_xy[angle_inx] = lda.transform(X)
            if normalize:
                #mean_fr = np.mean(single_angle_fr[angle_inx],axis = 1).reshape(98,1)
                #norm_fr = np.std(single_angle_fr[angle_inx],axis = 1).reshape(98,1)
                normal_fr[angle_inx] = np.sqrt(np.sum(single_angle_fr[angle_inx] ** 2))
                single_angle_fr[angle_inx] = single_angle_fr[angle_inx] / normal_fr[angle_inx]
                #single_angle_fr[angle_inx] = single_angle_fr[angle_inx]/norm_fr
                #mean_xy = np.mean(single_angle_position_xy[angle_inx],axis = 1).reshape(2,1)
                #norm_xy = np.std(single_angle_position_xy[angle_inx],axis = 1).reshape(2,1)
                normal_xy[angle_inx] = np.sqrt(np.sum(single_angle_position_xy[angle_inx] ** 2))
                single_angle_position_xy[angle_inx] = single_angle_position_xy[angle_inx]/ normal_xy[angle_inx]
                #print(np.isnan(single_angle_fr[angle_inx]).sum())
        return single_angle_fr, single_angle_position_xy, normal_fr, normal_xy


    def segmented_linear_data(self,normalize : bool = False):
        step = self.bin_width
        window_size = self.window_width
        trial = self.data
        motion_single_angle_fr = {}
        motion_single_angle_position_xy = {}
        rest_single_angle_fr = {}
        rest_single_angle_position_xy = {}
        for angle_inx in range(trial.shape[1]):
            motion_single_angle_fr[angle_inx] = np.zeros([98, 1])
            motion_single_angle_position_xy[angle_inx] = np.zeros([2, 1])
            rest_single_angle_fr[angle_inx] = np.zeros([98, 1])
            rest_single_angle_position_xy[angle_inx] = np.zeros([2, 1])
            for trial_inx in range(self.trail_num):
                trial_length = trial[trial_inx][angle_inx][1].shape[1]
                step_number = len(range(320, trial_length, step))
                fire_rate = np.zeros([98, step_number])
                position_xy = np.zeros([2, step_number])
                for time_inx in range(step_number):
                    real_time = time_inx * step + 320
                    fire_rate[:, time_inx] = np.sum(
                        trial[trial_inx][angle_inx][1][:, real_time - window_size:real_time],
                        axis=1)
                    #fire_rate[-1, time_inx] = time_inx
                    position_xy[0, time_inx] = trial[trial_inx][angle_inx][2][0][real_time]- trial[trial_inx][angle_inx][2][0][300]
                    position_xy[1, time_inx] = trial[trial_inx][angle_inx][2][1][real_time] - trial[trial_inx][angle_inx][2][0][300]
                motion_single_angle_fr[angle_inx] = np.concatenate((motion_single_angle_fr[angle_inx],
                                                                    fire_rate[:,:-5]), axis=1)
                motion_single_angle_position_xy[angle_inx] = np.concatenate((motion_single_angle_position_xy[angle_inx],
                                                                      position_xy[:,:-5]),
                                                                     axis=1)

                rest_single_angle_fr[angle_inx] = np.concatenate((rest_single_angle_fr[angle_inx],
                                                                    fire_rate[:,-5:]), axis=1)
                rest_single_angle_position_xy[angle_inx] = np.concatenate((rest_single_angle_position_xy[angle_inx],
                                                                      position_xy[:,-5:]),
                                                                     axis=1)
            motion_single_angle_fr[angle_inx] = motion_single_angle_fr[angle_inx][:,1:]
            motion_single_angle_position_xy[angle_inx] = motion_single_angle_position_xy[angle_inx][:,1:]
            rest_single_angle_fr[angle_inx] = rest_single_angle_fr[angle_inx][:,1:]
            rest_single_angle_position_xy[angle_inx] = rest_single_angle_position_xy[angle_inx][:,1:]
            if normalize:
                mean_fr = np.mean(single_angle_fr[angle_inx],axis = 1).reshape(98,1)
                std_fr = np.std(single_angle_fr[angle_inx],axis = 1).reshape(98,1)
                single_angle_fr[angle_inx] = (single_angle_fr[angle_inx]-mean_fr)/(std_fr**2)
                mean_xy = np.mean(single_angle_position_xy[angle_inx],axis = 1).reshape(2,1)
                std_xy = np.std(single_angle_position_xy[angle_inx],axis = 1).reshape(2,1)
                single_angle_position_xy[angle_inx] = (single_angle_position_xy[angle_inx] - mean_xy) / (std_xy ** 2)
        return motion_single_angle_fr, motion_single_angle_position_xy, rest_single_angle_fr, rest_single_angle_position_xy



class Linear_Regression(BaseModelRegression):
    def __init__(self, data_path: t.Union[np.ndarray, str], bin_width: int = 20, window_width: int = 300,isRidge: bool = False):
        """

        Args:
            data_path ():
            bin_width ():
            window_width ():
            isRidge (): Set it to False to use linear regression, otherwise it would be Ridge regression.
        """
        data_generator = RegressionData(data_path)
        self.fr, self.position_xy, self.norm_fr, self.norm_xy = data_generator.simple_linear_data(normalize = False)
        self.bin_width = bin_width
        self.window_width = window_width
        self.isRidge = isRidge
        self.models = {}
        #if isRidge:
            #self.models = {i: Ridge(alpha=0.9,normalize = True) for i in range(data_generator.angle_num)}
        #else:
            #self.models = {i: LinearRegression() for i in range(data_generator.angle_num)}

    def linear_predict(self, B, X):
        return X.dot(B)

    def linear_fit(self, X, y):
        return ((pinv((X.T).dot(X))).dot(X.T)).dot(y)

    def fit(self):
        for label in range(8):
            if self.isRidge:
                self.models[label] = Ridge(alpha=0.9,normalize = True)
                self.models[label].fit(self.fr[label].T, self.position_xy[label].T)
            else:
                self.models[label] = self.linear_fit(self.fr[label].T,self.position_xy[label].T)

    def predict(self, spikes: np.ndarray, label: int, initial_positions: np.ndarray) -> t.Tuple[float, float]:
        #split_idx = int(self.window_width / 3)
        valid_start = spikes.shape[1] - self.window_width
        firing_rate = np.sum(spikes[:, valid_start: ], axis=1)
        firing_rate = firing_rate
        if self.isRidge:
            hand_pos = self.models[label].predict(firing_rate.reshape([1,98]))
        else:
            hand_pos = self.linear_predict(self.models[label],firing_rate.reshape([1,98]))
        hand_pos = hand_pos
        initial_x, initial_y = initial_positions[0, 0].item(), initial_positions[1, 0].item()
        return hand_pos[0,0] + initial_x, hand_pos[0,1] + initial_y

class Segmented_Linear_Regression(BaseModelRegression):
    def __init__(self, data_path: t.Union[np.ndarray, str], bin_width: int = 20, window_width: int = 300,isRidge: bool = False):
        """
        Args:
            data_path ():
            bin_width ():
            window_width ():
            isRidge (): Set it to False to use linear regression, otherwise it would be Ridge regression.
        """
        data_generator = RegressionData(data_path)
        self.motion_fr, self.motion_xy, self.rest_fr, self.rest_xy = data_generator.segmented_linear_data(normalize = False)
        self.bin_width = bin_width
        self.window_width = window_width
        self.isRidge = isRidge
        self.models = {}
        #if isRidge:
            #self.models = {i: Ridge(alpha=0.9,normalize = True) for i in range(data_generator.angle_num)}
        #else:
            #self.models = {i: LinearRegression() for i in range(data_generator.angle_num)}

    def linear_predict(self, B, X):
        return X.dot(B)

    def linear_fit(self, X, y):
        return ((pinv((X.T).dot(X))).dot(X.T)).dot(y)

    def fit(self):
        for label in range(8):
            if self.isRidge:
                self.models[(label,0)] = Ridge(alpha=0.9,normalize = True)
                self.models[(label,0)].fit(self.motion_fr[label].T, self.motion_xy[label].T)
                self.models[(label,1)] = Ridge(alpha=0.9,normalize = True)
                self.models[(label,1)].fit(self.rest_fr[label].T, self.rest_xy[label].T)
            else:
                self.models[(label,0)] = self.linear_fit(self.motion_fr[label].T,self.motion_xy[label].T)
                self.models[(label, 1)] = self.linear_fit(self.rest_fr[label].T, self.rest_xy[label].T)
    def predict(self, spikes: np.ndarray, label: int,initial_positions: np.ndarray) -> t.Tuple[float, float]:
        #split_idx = int(self.window_width / 3)
        valid_start = spikes.shape[1] - self.window_width
        firing_rate = np.sum(spikes[:, valid_start: ], axis=1)
        firing_rate = firing_rate
        if spikes.shape[1] >480:
            state = 1
        else:
            state = 0
        if self.isRidge:
            hand_pos = self.models[(label, state)].predict(firing_rate.reshape([1,98]))
        else:
            hand_pos = self.linear_predict(self.models[(label, state)],firing_rate.reshape([1,98]))
        hand_pos = hand_pos
        initial_x, initial_y = initial_positions[0, 0].item(), initial_positions[1, 0].item()
        return hand_pos[0,0] + initial_x, hand_pos[0,1] + initial_y

if __name__ == "__main__":
    import os

    src_dir = os.path.join(os.path.abspath(__file__), '..')
    mat_path = os.path.join(src_dir, 'monkeydata_training.mat')

    solution = SPlitRegression(mat_path)
    solution.fit()