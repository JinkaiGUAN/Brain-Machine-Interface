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
        # The training and testing dataset.
        self.split_ratio = 0.5

        # Time window parameter
        self.bin_width = 20
        self.time_window_width = 300

        # the index where we want to separate the window
        self.split_idx = 140

        # weather convert to polar coordinate, default: False
        self.polar = False
        # model configurations
        self.split_regression = "split_regression"
        self.split_ridge_regression = "split_ridge_regression"
        self.simple_linear_regression = "linear_regression"
        self.simple_ridge_regression = "simple_ridge_regression"
        self.segmented_linear_regression = "segmented_linear_regression"
        self.segmented_ridge_regression = "segmented_ridge_regression"
        self.all_linear_regression = "all_linear_regression"
        self.all_ridge_regression = "all_ridge_regression"
        self.state_linear_regression = "state_linear_regression"
        self.state_ridge_regression = "state_ridge_regression"
        # model chosen
        self.cnn_classification = "cnn_classification"
        self.knn_classification = "knn_classification"

        # model chosen
        self.classifier_name = self.knn_classification
        self.model_name = self.segmented_ridge_regression


