# -*- coding: UTF-8 -*-
"""
@Project : 3-coursework 
@File    : configuration.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 26/04/2022 15:24 
@Brief   : 
"""


class Configuration:
    def __init__(self):
        # Time window parameter
        self.bin_width = 20
        self.time_window_width = 300

        # the index where we want to separate the window
        self.split_idx = 100

        # model configurations
        self.split_regression = "split_regression"
        self.ridge_regression = "ridge_regression"
        self.simple_linear_regression = "linear_regression"

        self.cnn_classification = "cnn_classification"
        self.knn_classification = "knn_classification"

        # model chosen
        self.model_name = self.split_regression
        self.classifier_name = self.cnn_classification



