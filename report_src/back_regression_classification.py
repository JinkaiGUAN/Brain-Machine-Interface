# -*- coding: UTF-8 -*-
"""
@Project : 3-coursework 
@File    : back_regression_classification.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 27/04/2022 15:38 
@Brief   : Classify the post regression data.
"""
import typing as t
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from __future__ import annotations

import numpy as np
import scipy.io as scio

from preprocess import Trial


class PostClassificationData:
    """This dataset is used to classify the still state and moving state at the end of training."""

    class PostClassificationEntry:
        def __init__(self, firing_rate: t.List[np.ndarray], still_label: t.List, hand_position_x: t.List[float],
                     hand_position_y: t.List[float]) -> None:
            self.firing_rate = firing_rate
            self.still_label = still_label
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

        non_still_data = self.__generate_data__(100, still_label=0)
        still_data = self.__generate_data__(100, still_label=1)

        return non_still_data, still_data


class BackRegressionTraining:
    def __init__(self, data: PostClassificationData):

        self.data = data
        self.model = KNeighborsClassifier()

    def fit(self):




