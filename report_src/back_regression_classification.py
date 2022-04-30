# -*- coding: UTF-8 -*-
"""
@Project : 3-coursework 
@File    : back_regression_classification.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 27/04/2022 15:38 
@Brief   : Classify the post regression data.
"""
from __future__ import annotations
import typing as t
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import scipy.io as scio

from preprocess import Trial
from configuration import Configuration

config = Configuration()


class PostClassificationData:
    """This dataset is used to classify the still state and moving state at the end of training."""

    class PostClassificationEntry:
        def __init__(self, firing_rate: t.List[np.ndarray], still_labels: t.List, hand_position_x: t.List[float],
                     hand_position_y: t.List[float]) -> None:
            self.firing_rate = firing_rate
            self.still_labels = still_labels
            self.hand_position_x = hand_position_x
            self.hand_position_y = hand_position_y

    def __init__(self, data_path: t.Union[np.ndarray, str], bin_width: int = 20, window_width: int = 300,
                 valid_start: int = 0, valid_end: int = 340):
        # super(, self).__init__(data_path)
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

    def __generate_data__(self, split_idx: int = 100, still_label: int = 0) -> t.Dict[int,
                                                                    PostClassificationData.PostClassificationEntry]:
        """

        Args:
            split_idx ():
            still_label (): Represents the hand is still or moving. O is moving, 1 is still.

        Returns:

        """

        all_data = defaultdict(PostClassificationData.PostClassificationEntry)

        for angle_idx in range(self.angle_num):

            firing_rate = []
            still_labels = []  # if the window edge touches the dead region, it should be 1. Otherwise, it is 0.
            hand_position_x = []
            hand_position_y = []
            for trail_idx in range(self.trail_num):
                raw_single_trail = Trial(self.data[trail_idx, angle_idx], 0, -1)
                time_length = raw_single_trail.spikes.shape[1]

                for _start in range(0, len(raw_single_trail) - self.window_width + 1, self.bin_width):
                    raw_single_trail.valid_start, raw_single_trail.valid_end = _start, _start + self.window_width

                    # still_label == 0
                    if still_label == 0 and raw_single_trail.valid_end >= (time_length - split_idx):
                        # We suppose the hand is still
                        continue

                    # still_label == 1
                    if still_label == 1 and raw_single_trail.valid_end < (time_length - split_idx):
                        continue

                    still_labels.append(still_label)

                    # Generate firing rate
                    firing_rate.append(raw_single_trail.firing_rate_by_sum)

                    hand_position_x.append(raw_single_trail.hand_pos_x - raw_single_trail.initial_hand_pos_x)
                    hand_position_y.append(raw_single_trail.hand_pos_y - raw_single_trail.initial_hand_pos_y)

            all_data[angle_idx] = PostClassificationData.PostClassificationEntry(firing_rate,
                                                                                 still_labels,
                                                                                 hand_position_x,
                                                                                 hand_position_y)

        return all_data

    def generate_data(self) -> t.Tuple[t.Dict, t.Dict]:
        split_idx = config.split_idx

        non_still_data = self.__generate_data__(split_idx, still_label=0)
        still_data = self.__generate_data__(split_idx, still_label=1)

        return non_still_data, still_data


class BackRegressionTraining:
    def __init__(self, data: PostClassificationData):

        self.data = data
        self.model = KNeighborsClassifier()

    def fit(self):
        X, y = self.__merge_data__()

        self.model.fit(X, y)

    def predict(self, spikes: np.ndarray) -> int:
        time_length = spikes.shape[1]

        firing_rate = np.sum(spikes[:, time_length - self.data.window_width:], axis=1)

        return int(self.model.predict([firing_rate.tolist()])[0].item())
    #
    def __merge_data__(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """Merge the data into non-still and still part for training."""
        non_still_firing_rate = []
        still_firing_rate = []
        non_still_labels = []
        still_labels = []

        non_still_data, still_data = self.data.generate_data()
        for (_, non_still), (_, still) in zip(non_still_data.items(), still_data.items()):
            non_still_firing_rate += non_still.firing_rate
            non_still_labels += non_still.still_labels
            still_firing_rate += still.firing_rate
            still_labels += still.still_labels

        X = np.concatenate((non_still_firing_rate, still_firing_rate), axis=0)
        y = np.concatenate((non_still_labels, still_labels), axis=0)

        return X, y




