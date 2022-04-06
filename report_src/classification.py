# -*- coding: UTF-8 -*-
"""
@Project : 3-coursework 
@File    : classification.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 04/04/2022 23:10 
@Brief   : 
"""
import os
import random
import typing as t

import numpy as np
import scipy.io as scio
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

from preprocess import RetrieveData


class AvgMeter:
    def __init__(self, name: str):
        self._name = name
        self._vals = []

    def push(self, val: float) -> None:
        self._vals.append(val)

    @property
    def mean(self) -> float:
        return np.mean(self._vals).item()

    def clear_cash(self) -> None:
        self._vals.clear()


class Classifier:
    def __init__(self, params: t.Dict = None) -> None:
        """In this project, accuracy can be more important than MSE, thus, in the comparison of accuracy of two
        naive classifiers, we concluded that KNN is more robust for firring rate data."""
        params = {
            "n_neighbors": 30,
            "algorithm": "ball_tree"
        }
        self.model = KNeighborsClassifier(**params)

        # self.model = GaussianNB()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class Trainer:
    def __init__(self, data_path: str) -> None:
        self.data = scio.loadmat(data_path).get('trial')

        self.training_data = RetrieveData(self.data[:51, :], valid_start=0, valid_end=340)
        self.test_data = RetrieveData(self.data[51:, :], valid_start=0, valid_end=340)

        self.model = Classifier()
        self.pca = PCA(n_components=95)

    def k_fold_cv(self, k: int = 10) -> None:
        """Implements K-fold cross-validation"""
        accuracy_meter = AvgMeter("accuracy")
        mse_meter = AvgMeter("MSE")

        for k in range(k + 1):
            # print("{:-^60}".format(f" {k}-th fold training "))

            trial_num = self.data.shape[0]
            idxes = list(range(trial_num))
            random.shuffle(idxes)

            self.training_data = RetrieveData(self.data[idxes[:51], :])
            self.test_data = RetrieveData(self.data[idxes[51:], :])

            # Deploy PCA
            self.pca.fit(self.training_data.X)
            self.training_data.X = self.pca.transform(self.training_data.X)
            self.test_data.X = self.pca.transform(self.test_data.X)

            self.model.fit(self.training_data.X, self.training_data.y)
            y_pred = self.model.predict(self.test_data.X)
            # print("Accuracy: {:.2f}".format(metrics.accuracy_score(self.test_data.y, y_pred)))
            # print("MSE: {:.2f}".format(metrics.mean_squared_error(self.test_data.y, y_pred)))
            accuracy_meter.push(metrics.accuracy_score(self.test_data.y, y_pred))
            mse_meter.push(metrics.mean_squared_error(self.test_data.y, y_pred))

        print("AVG -- Acc: {:.2f} MSE: {:.2f}".format(accuracy_meter.mean, mse_meter.mean))

    def initial_position_checker(self):
        """Check the initial position of two hands to see whether this can improve the model performance or not."""
        pass

    def run(self):
        """Parameters tuning"""

        for n_component in [98, 95, 90, 85, 80, 75, 70]:
            self.pca = PCA(n_components=n_component)
            print("{:-^60}".format(f" n_component = {n_component} "))
            self.k_fold_cv()


if __name__ == "__main__":
    src_dir = os.path.join(os.path.abspath(__file__), '..')
    mat_path = os.path.join(src_dir, 'monkeydata_training.mat')

    # classifier = Classifier(mat_path)

    trainer = Trainer(mat_path)
    # trainer.k_fold_cv(10)
    trainer.run()
