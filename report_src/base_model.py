# -*- coding: UTF-8 -*-
"""
@Project : 3-coursework 
@File    : base_model.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 26/04/2022 15:38 
@Brief   : 
"""
from abc import abstractmethod

import typing as t
import numpy as np


class BaseModelRegression:

    @abstractmethod
    def predict(self, spikes: np.ndarray, label: int, initial_position: np.ndarray) -> t.Tuple[float, float]:
        """ The predict function for regression model.

        Args:
            spikes (): The original spikes without any preprocess. However, the time length of this spikes can be
                varying.
            label (): The label given by the classification model.
            initial_position (): The initial hand positin with size of 2 by 1.

        Returns:
            Return a tuple, i.e., the x and y position of given spikes. And the position should be transferred to
            float value.
        """
        raise NotImplementedError("Please implement predict function!")

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError("Please implement fit function!")


