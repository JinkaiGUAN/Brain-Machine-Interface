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
        self.split_idx = 200

        # ####################### model configurations #######################
        #  Regressors
        self.split_regression = "split_regression"
        self.split_ridge_regression = "split_ridge_regression"
        self.simple_linear_regression = "linear_regression"
        self.simple_ridge_regression = "simple_ridge_regression"
        self.segmented_linear_regression = "segmented_linear_regression"
        self.segmented_ridge_regression = "segmented_ridge_regression"

        # Classifiers for angle index chosen
        self.cnn_classification = "cnn_classification"
        self.knn_classification = "knn_classification"

        # model chosen
        self.classifier_name = self.cnn_classification
        self.model_name = self.segmented_ridge_regression



