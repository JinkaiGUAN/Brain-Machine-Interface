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

        # model configurations
        self.split_regression = "split_regression"
        self.ridge_regression = "ridge_regression"
        self.simple_linear_regression = "linear_regression"

        # model chosen
        self.model_name = self.ridge_regression

