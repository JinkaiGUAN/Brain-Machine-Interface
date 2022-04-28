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
from collections import defaultdict

import numpy as np
import scipy.io as scio


class Trial:
    """The base data structure for each trail."""

    def __init__(self, origin_data: np.ndarray, valid_start: int = None, valid_end: int = None) -> None:
        self._trial_id = origin_data[0].item()
        self._spikes = origin_data[1]
        self._hand_pos = origin_data[2]

        self._valid_start = valid_start if valid_start is not None else 0
        self._valid_end = valid_end if valid_end is not None else 0

    def __len__(self) -> int:
        """The time length of this trial."""
        return self._spikes.shape[1]

    @property
    def spikes(self) -> np.ndarray:
        return self._spikes

    @property
    def hand_pos_x(self) -> np.ndarray:
        """A single numpy item represents the x information of the hand, where -1 represents the `self._valid_end`-th
        element."""
        return self._hand_pos[0, self._valid_end - 1]

    @property
    def hand_pos_y(self) -> np.ndarray:
        """A single numpy item represents the y information of the hand, where -1 represents the `self._valid_end`-th
        element."""
        return self._hand_pos[1, self._valid_end - 1]

    @property
    def initial_hand_pos_x(self) -> float:
        return self._hand_pos[0, 0].item()

    @property
    def initial_hand_pos_y(self) -> float:
        return self._hand_pos[1, 0].item()

    @property
    def hand_pos_all_x(self) -> np.ndarray:
        return self._hand_pos[0, :]

    @property
    def hand_pos_all_y(self) -> np.ndarray:
        return self._hand_pos[1, :]

    @property
    def valid_start(self) -> int:
        return self._valid_start

    @valid_start.setter
    def valid_start(self, val: int) -> None:
        self._valid_start = val

    @property
    def valid_end(self) -> int:
        return self._valid_end

    @valid_end.setter
    def valid_end(self, val: int) -> None:
        self._valid_end = val

    @property
    def firing_rate(self) -> np.ndarray:
        # data size: (98, )
        # It protects from retrieving firing rate without deploying start and end indices.
        if self._valid_end == 0 and self._valid_start == 0:
            raise NotImplementedError(f"The start and end indices have not been assigned for"
                                      f" {self.__class__.__name__}!")

        if self._valid_end == -1:
            return np.mean(self._spikes[:, self._valid_start:], axis=1)
        else:
            return np.mean(self._spikes[:, self._valid_start: self._valid_end], axis=1)

    @property
    def firing_rate_by_sum(self) -> np.ndarray:
        """Using sum to calculate the firing rate."""
        if self._valid_end == -1:
            return np.sum(self._spikes[:, self._valid_start:], axis=1)
        else:
            return np.sum(self._spikes[:, self._valid_start: self._valid_end], axis=1)

    @property
    def raw_firing_rate(self) -> np.ndarray:
        """Return the firing rate raw data, i.e., the raw spiking data, with data size of (98, n), where n is the
        time steps."""
        if self._valid_end == 0 and self._valid_start == 0:
            raise NotImplementedError(f"The start and end indices have not been assigned for"
                                      f" {self.__class__.__name__}!")

        if self._valid_end == -1:
            return self._spikes[:, self._valid_start:]
        else:
            return self._spikes[:, self._valid_start: self._valid_end]

    @property
    def split_firing_rate(self) -> np.ndarray:
        """Split the firing rate, i.e., split the whole time window into 3 parts, and merge them."""
        if self._valid_end == 0 and self._valid_start == 0:
            raise NotImplementedError(f"The start and end indices have not been assigned for"
                                      f" {self.__class__.__name__}!")

        time_window = self.valid_end - self.valid_start
        split_idx = int(time_window / 3)

        return np.concatenate((np.sum(self._spikes[:, self.valid_start : split_idx], axis=1),
                               np.sum(self._spikes[:, split_idx: 2 * split_idx], axis=1),
                               np.sum(self._spikes[:, 2 * split_idx: self.valid_end], axis=1)), axis=0)


    def get_firing_rate(self) -> np.ndarray:
        """get the firing rate"""
        if self._valid_end == 0 and self._valid_start == 0:
            raise NotImplementedError(f"The start and end indices have not been assigned for"
                                      f" {self.__class__.__name__}!")

        #time_window = self.valid_end - self.valid_start
        return np.sum(self._spikes[:, self.valid_end-300:self.valid_end], axis=1)

    # @property
    # def post_two_blocks(self) -> t.Tuple[np.ndarray, np.ndarray]:
    #     """Split the post neuro signal (i.e., last time window) into two blocks. The first block gives the increasing
    #         distance relationship, and the second part gives us almost still movement since the monkey is trying to stop
    #         the hand.
    #
    #     Notes:
    #
    #     """




class RetrieveData:
    def __init__(self, data_path: t.Union[np.ndarray, str], bin_width: int = 20, window_width: int = 300,
                 valid_start: int = 0, valid_end: int = 340, isClassification: bool = True) -> None:
        """The data retriever.

        Args:
            data_path (Union[np.ndarray, str]): To be compatible with different input data type, i.e.,
                a string file path or a numpy matrix, we specify the type can be the absolute path of the MAT file
                or the raw numpy data.
            bin_width (int): The sampling step.
            window_width (int): The size of time window.
            valid_start (int): The start of the time window, classification used only.
            valid_end (int): The end of the time window, classification used only.
            isClassification (bool): The flag that judges that whether the data is prepared for classification or
                linear regression.
        """
        if isinstance(data_path, str):
            self.data = scio.loadmat(data_path).get('trial')
        if isinstance(data_path, np.ndarray):
            self.data = data_path

        self.bin_width = bin_width
        self.window_width = window_width
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.isClassification = isClassification

        # retrieve data information
        self.trail_num = self.data.shape[0]
        self.angle_num = self.data.shape[1]
        self.neuro_num = self.data[0, 0][1].shape[0]

        # Initialize X and Y, where X stores the firing rate for each trail, and y stores the corresponding reaching
        # angele
        if self.isClassification:
            self._X = np.zeros((self.trail_num * self.angle_num, self.neuro_num))
            self._y = np.zeros(self.trail_num * self.angle_num)
        else:
            self._X = []
            self._X_classification = []
            self._y = []

        # Assign the hand positions, which is going to be the label in linear regression.
        self._hand_position_x = []
        self._hand_position_y = []
        self._hand_positions = defaultdict(list)

        # Fill dataset
        if self.isClassification:
            self.assign_dataset()
        else:
            self.assign_dataset_v2()

    def assign_dataset(self):
        """Retrieve data from raw input data, which should be used for KNN classification."""

        pre_idx = 0
        for trail_idx in range(self.trail_num):
            for angle_idx in range(self.angle_num):
                single_trail = Trial(self.data[trail_idx, angle_idx], self.valid_start, self.valid_end)
                self._X[pre_idx + angle_idx, :] = single_trail.firing_rate
                self._y[pre_idx + angle_idx] = angle_idx

                self._hand_positions['x'].append(single_trail.hand_pos_all_x)
                self._hand_positions['y'].append(single_trail.hand_pos_all_y)

            pre_idx += self.angle_num

    def assign_dataset_v2(self):
        """In this function, bin will be used to sample the data within the same time sliding window."""
        for trail_idx in range(self.trail_num):
            for angle_idx in range(self.angle_num):
                # Retrieve the data by the bins
                single_trail = Trial(self.data[trail_idx, angle_idx])
                for _start in range(0, len(single_trail) - self.window_width + 1, self.bin_width):
                    # The start is from 0
                    single_trail.valid_start, single_trail.valid_end = 0, _start + self.window_width
                    # Assign firing rate and reaching angles

                    spikes = single_trail.raw_firing_rate
                    time_length = spikes.shape[1]
                    time_length = time_length if time_length <= 320 else 320

                    sum_spike = np.sum(spikes[:, 0:time_length], axis=1)
                    self._X.append(sum_spike)
                    self._y.append(angle_idx)

                    # self._X.append(single_trail.firing_rate.tolist())
                    # self._X.append(single_trail.raw_firing_rate)
                    # self._y.append(angle_idx)

                    # retrieve hand positions, using float value
                    # self._hand_position_x.append(single_trail.hand_pos_x.item())
                    # self._hand_position_y.append(single_trail.hand_pos_y.item())

    @property
    def X(self) -> np.ndarray:
        """Firing rate, Data size of (trail_num x angle_num, 98)."""
        return np.asarray(self._X)

    @X.setter
    def X(self, val: np.ndarray) -> None:
        self._X = val

    @property
    def X_classification(self) -> np.ndarray:
        """The firring rate for classification use only."""
        return np.asarray(self._X_classification)

    @property
    def y(self) -> np.ndarray:
        """Classification true label (i.e., reaching angle indices), Data size of (trail_num x angle_num, )."""
        return np.asarray(self._y)

    @y.setter
    def y(self, val: np.ndarray) -> None:
        self._y = val

    @property
    def hand_position_x(self) -> np.ndarray:
        """Hand position of x-axis that would be used for linear training."""
        return np.ndarray(self._hand_position_x)

    @property
    def hand_position_y(self) -> np.ndarray:
        """Hand position of y-axis that would be used for linear training."""
        return np.ndarray(self._hand_position_y)

    @property
    def hand_positions(self) -> t.Dict:
        """All hand position information."""
        return self._hand_positions



if __name__ == "__main__":
    src_dir = os.path.join(os.path.abspath(__file__), '..')
    mat_path = os.path.join(src_dir, 'monkeydata_training.mat')

    retrieve_data = RetrieveData(mat_path, isClassification=False)
    retrieve_data.assign_dataset_v2()
    # print(retrieve_data.X.shape)
    # print(retrieve_data.y.shape)
