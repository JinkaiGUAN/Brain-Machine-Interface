# -*- coding: UTF-8 -*-
"""
@Project : 3-coursework 
@File    : preprocess.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 04/04/2022 22:11 
@Brief   : 
"""

import os
import scipy.io as scio
import numpy as np
import typing as t


class Trial:
    """The base data structure for each trail."""
    def __init__(self, origin_data: np.ndarray, valid_start: int = 0, valid_end: int = -1) -> None:
        self._trial_id = origin_data[0].item()
        self._spikes = origin_data[1]
        self._hand_pos = origin_data[2]

        self.valid_start = valid_start
        self.valid_end = valid_end

    @property
    def spikes(self) -> np.ndarray:
        return self._spikes

    @property
    def hand_pos(self) -> np.ndarray:
        return self._hand_pos

    @property
    def firing_rate(self) -> np.ndarray:
        # data size: (98, )
        if self.valid_end == -1:
            return np.mean(self._spikes[self.valid_start:], axis=1)
        else:
            return np.mean(self._spikes[self.valid_start: self.valid_end], axis=1)


class RetrieveData:
    def __init__(self, data_path: t.Union[np.ndarray, str], valid_start: int = 0, valid_end: int = -1) -> None:
        """The data retriever.

        Args:
            data_path (Union[np.ndarray, str]): To be compatible with different input data type, i.e.,
                a string file path or a numpy matrix, we specify the type can be the absolute path of the MAT file
                or the raw numpy data.
            valid_start (int): The start of the time window.
            valid_end (int): The end of the time window.
        """
        if isinstance(data_path, str):
            self.data = scio.loadmat(data_path).get('trial')
        if isinstance(data_path, np.ndarray):
            self.data = data_path

        self.valid_start = valid_start
        self.valid_end = valid_end

        # retrieve data information
        self.trail_num = self.data.shape[0]
        self.angle_num = self.data.shape[1]
        self.neuro_num = self.data[0, 0][1].shape[0]

        # Initialize X and Y, where X stores the firing rate for each trail, and y stores the corresponding reaching
        # angele
        self._X = np.zeros((self.trail_num * self.angle_num, self.neuro_num))
        self._y = np.zeros(self.trail_num * self.angle_num)

        # Fill dataset
        self.assign_dataset()

    def assign_dataset(self):
        """Retrieve data from raw input data."""
        pre_idx = 0
        for trail_idx in range(self.trail_num):
            for angle_idx in range(self.angle_num):
                single_trail = Trial(self.data[trail_idx, angle_idx], self.valid_start, self.valid_end)
                self._X[pre_idx + angle_idx, :] = single_trail.firing_rate
                self._y[pre_idx + angle_idx] = angle_idx

            pre_idx += self.angle_num

    @property
    def X(self) -> np.ndarray:
        """Data size of (trail_num x angle_num, 98)"""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Data size of (trail_num x angle_num, )"""
        return self._y


if __name__ == "__main__":
    src_dir = os.path.join(os.path.abspath(__file__), '..')
    mat_path = os.path.join(src_dir, 'monkeydata_training.mat')

    retrieve_data = RetrieveData(mat_path)
    print(retrieve_data.X.shape)
    print(retrieve_data.y.shape)

