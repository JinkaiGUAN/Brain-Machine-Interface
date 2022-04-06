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
import typing as t
from collections import deque
from collections import defaultdict

import numpy as np
import scipy.io as scio


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
    def hand_pos_x(self) -> np.ndarray:
        """A single numpy item represents the x information of the hand."""
        return self._hand_pos[0, self.valid_end]

    @property
    def hand_pos_y(self) -> np.ndarray:
        """A single numpy item represents the y information of the hand."""
        return self._hand_pos[1, self.valid_end]

    @property
    def hand_pos_all_x(self) -> np.ndarray:
        return self._hand_pos[0, :]

    @property
    def hand_pos_all_y(self) -> np.ndarray:
        return self._hand_pos[1, :]

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

        # Assign the hand positions, which is going to be the label in linear regression.
        self._hand_position_x = []
        self._hand_position_x = []
        self._hand_positions = defaultdict(list)

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

                self._hand_positions['x'].append(single_trail.hand_pos_all_x)
                self._hand_positions['y'].append(single_trail.hand_pos_all_y)

            pre_idx += self.angle_num

    @property
    def X(self) -> np.ndarray:
        """Firing rate, Data size of (trail_num x angle_num, 98)"""
        return self._X

    @X.setter
    def X(self, val: np.ndarray) -> None:
        self._X = val

    @property
    def y(self) -> np.ndarray:
        """Classification true label (i.e., reaching angle indices), Data size of (trail_num x angle_num, )"""
        return self._y

    @y.setter
    def y(self, val: np.ndarray) -> None:
        self._y = val

    @property
    def hand_position_x(self) -> np.ndarray:
        pass

    @property
    def hand_position_y(self) -> np.ndarray:
        pass

    @property
    def hand_positions(self) -> t.Dict:
        """All hand position information."""
        return self._hand_positions


if __name__ == "__main__":
    src_dir = os.path.join(os.path.abspath(__file__), '..')
    mat_path = os.path.join(src_dir, 'monkeydata_training.mat')

    retrieve_data = RetrieveData(mat_path)
    print(retrieve_data.X.shape)
    print(retrieve_data.y.shape)
