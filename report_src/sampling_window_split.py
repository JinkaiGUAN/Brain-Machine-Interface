# -*- coding: UTF-8 -*-
"""
@Project : 3-coursework 
@File    : sampling_window_split.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 25/04/2022 16:31 
@Brief   : 
"""
import typing as t
from collections import defaultdict

import numpy as np
import scipy.io as scio
from sklearn.linear_model import LinearRegression, Ridge

from preprocess import Trial
from base_model import BaseModelRegression


class SingleAngleData:
    def __init__(self, firing_rate, hand_position_x, hand_position_y):
        self.firing_rate = firing_rate
        self.hand_position_x = hand_position_x
        self.hand_position_y = hand_position_y


class RegressionData:
    def __init__(self, data_path: t.Union[np.ndarray, str], bin_width: int = 20, window_width: int = 300,
                 valid_start: int = 0, valid_end: int = 340):

        if isinstance(data_path, str):
            self.data = scio.loadmat(data_path).get('trial')
        if isinstance(data_path, np.ndarray):
            self.data = data_path
        self.bin_width = bin_width
        self.window_width = window_width
        self.valid_start = valid_start
        self.valid_end = valid_end

        # retrieve data information
        self.trail_num = self.data.shape[0]
        self.angle_num = self.data.shape[1]
        self.neuro_num = self.data[0, 0][1].shape[0]

    def generate_data(self) -> t.Dict[int, SingleAngleData]:

        all_data = defaultdict(SingleAngleData)

        for angle_idx in range(self.angle_num):

            firing_rate = []
            hand_position_x = []
            hand_position_y = []

            for trail_idx in range(self.trail_num):
                raw_single_trail = Trial(self.data[trail_idx, angle_idx], 0, -1)

                for _start in range(0, len(raw_single_trail) - self.window_width + 1, self.bin_width):
                    raw_single_trail.valid_start, raw_single_trail.valid_end = _start, _start + self.window_width

                    # Generate firing rate
                    firing_rate.append(raw_single_trail.split_firing_rate)
                    hand_position_x.append(raw_single_trail.hand_pos_x - raw_single_trail.initial_hand_pos_x)
                    hand_position_y.append(raw_single_trail.hand_pos_y - raw_single_trail.initial_hand_pos_y)

            all_data[angle_idx] = SingleAngleData(firing_rate, hand_position_x, hand_position_y)

        return all_data


class SPlitRegression(BaseModelRegression):
    def __init__(self, data_path: t.Union[np.ndarray, str], bin_width: int = 20, window_width: int = 300,
                 isRidge: bool = False):
        """

        Args:
            data_path ():
            bin_width ():
            window_width ():
            isRidge (): Set it to False to use linear regression, otherwise it would be Ridge regression.
        """
        data_generator = RegressionData(data_path)

        self.data: t.Dict[int, SingleAngleData] = data_generator.generate_data()
        self.bin_width = bin_width
        self.window_width = window_width

        if isRidge:
            self.models = {i: Ridge(alpha=1) for i in range(data_generator.angle_num)}
        else:
            self.models = {i: LinearRegression() for i in range(data_generator.angle_num)}

    def fit(self):
        for label, data in self.data.items():
            position = np.concatenate((np.expand_dims(data.hand_position_x, axis=1),
                                       np.expand_dims(data.hand_position_y, axis=1)), axis=1)  # N * 2
            self.models[label].fit(data.firing_rate, position)

    def predict(self, spikes: np.ndarray, label: int, initial_positions: np.ndarray) -> t.Tuple[float, float]:
        split_idx = int(self.window_width / 3)
        valid_start = spikes.shape[1] - self.window_width

        firing_rate = np.concatenate((
            np.sum(spikes[:, valid_start: split_idx], axis=1),
            np.sum(spikes[:, split_idx: 2 * split_idx], axis=1),
            np.sum(spikes[:, 2 * split_idx:], axis=1)))

        hand_pos = self.models[label].predict([firing_rate])
        initial_x, initial_y = initial_positions[0, 0].item(), initial_positions[1, 0].item()

        return hand_pos[0, 0].item() + initial_x, hand_pos[0, 1].item() + initial_y


if __name__ == "__main__":
    import os

    src_dir = os.path.join(os.path.abspath(__file__), '..')
    mat_path = os.path.join(src_dir, 'monkeydata_training.mat')

    solution = SPlitRegression(mat_path)
    solution.fit()
