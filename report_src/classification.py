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
import torch
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from preprocess import RetrieveData
from preprocess import Trial


# Configure the global configuration for plotting
plot_config = {
    "font.family": 'Times New Roman',
    "font.size": 24,
    "mathtext.fontset": 'stix',
    "axes.titlesize": 30
}
rcParams.update(plot_config)


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


class KNN_Classifier:
    def __init__(self, model_name: str = None, params: t.Dict = None) -> None:
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

    def predict(self, X: np.ndarray) -> int:
        time_length = X.shape[1]

        threshold = 340
        # time_step = threshold if time_length > threshold else -1
        if time_length <= threshold:
            X = np.mean(X, axis=1)
        else:
            X = np.mean(X[:, :threshold], axis=1)

        return int(self.model.predict(np.asarray([X.tolist()])).item())


class Trainer:
    def __init__(self, data_path: str) -> None:
        self.data = scio.loadmat(data_path).get('trial')

        self.training_data = RetrieveData(self.data[:51, :], valid_start=0, valid_end=340, isClassification=True)
        self.test_data = RetrieveData(self.data[51:, :], valid_start=0, valid_end=340, isClassification=True)

        self.model = KNN_Classifier()
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

            # end =
            self.training_data = RetrieveData(self.data[idxes[:51], :], valid_start=0, valid_end=-1,
                                              isClassification=True)
            self.test_data = RetrieveData(self.data[idxes[51:], :], valid_start=0, valid_end=-1, isClassification=True)

            # Deploy PCA
            # self.pca.fit(self.training_data.X)
            # self.training_data.X = self.pca.transform(self.training_data.X)
            # self.test_data.X = self.pca.transform(self.test_data.X)

            self.model.fit(self.training_data.X, self.training_data.y)
            y_pred = self.model.predict(self.test_data.X)
            # print("Accuracy: {:.2f}".format(metrics.accuracy_score(self.test_data.y, y_pred)))
            # print("MSE: {:.2f}".format(metrics.mean_squared_error(self.test_data.y, y_pred)))
            accuracy_meter.push(metrics.accuracy_score(self.test_data.y, y_pred))
            mse_meter.push(metrics.mean_squared_error(self.test_data.y, y_pred))

        print("AVG -- Acc: {:.2f} MSE: {:.2f}".format(accuracy_meter.mean, mse_meter.mean))

    def initial_position_checker(self):
        """Check the initial position of two hands to see whether this can improve the model performance or not."""
        # the initial position of all data
        self.model.fit(self.training_data.X, self.training_data.y)

        time_idx = 400
        x_0 = np.asarray([item[time_idx] for item in self.training_data.hand_positions['x']])
        y_0 = np.asarray([item[time_idx] for item in self.training_data.hand_positions['y']])

        labels = np.unique(self.training_data.y)
        fig = plt.figure(figsize=(10, 10))
        for label in labels:
            indices = np.where(self.test_data.y == label)
            plt.scatter(x_0[indices[0]], y_0[indices[0]], label=label)

        y = self.model.predict(np.asarray([self.test_data.X[0, :]]))
        print(y)
        plt.scatter(self.test_data.hand_positions['x'][0][0], self.test_data.hand_positions['y'][0][0])

        plt.legend()
        plt.show()
        test_data = [[xs[0], ys[0]] for xs, ys in zip(self.test_data.hand_positions['x'], self.test_data.hand_positions[
            'y'])]
        y_pred = self.model.predict(np.asarray(test_data))
        print(metrics.accuracy_score(self.test_data.y, y_pred))

    def run(self):
        """Parameters tuning"""

        for n_component in [98, 95, 90, 85, 80, 75, 70]:
            self.pca = PCA(n_components=n_component)
            print("{:-^60}".format(f" n_component = {n_component} "))
            self.k_fold_cv()

    def classification_linear(self):
        pass

    def velocity_checker(self):
        # self.training_data.hand_position_x
        pass

    def test(self):
        """Test function for all."""
        pass


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 70)
        self.layer3 = nn.Linear(70, 35)
        self.layer4 = nn.Linear(35, 8)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class CNN_Classifier:
    def __init__(self, data_path: t.Union[np.ndarray, str], bin_width: int = 20, window_width: int = 300):
        if isinstance(data_path, str):
            self.data = scio.loadmat(data_path).get('trial')
        if isinstance(data_path, np.ndarray):
            self.data = data_path

        self.bin_width = bin_width
        self.window_width = window_width

        self.classification_training_data = RetrieveData(self.data, bin_width=self.bin_width,
                                                         window_width=self.window_width,
                                                         valid_start=0, valid_end=340,
                                                         isClassification=False)

        self.epoch_num = 200
        self.model = Model(self.classification_training_data.X.shape[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, x=None, y=None) -> None:
        X_train = Variable(torch.from_numpy(self.classification_training_data.X.astype(np.float64))).float()
        y_train = Variable(torch.from_numpy(self.classification_training_data.y)).long()

        # loss_list = np.zeros((self.epoch_num,))
        # accuracy_list = np.zeros((self.epoch_num,))

        for epoch in range(self.epoch_num):
            y_pred = self.model(X_train)
            loss = self.loss_fn(y_pred, y_train)
            # loss_list[epoch] = loss.item()

            # Zero gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X: np.ndarray) -> int:
        # time_length = X.shape[1]
        #
        # threshold = 340
        # # time_step = threshold if time_length > threshold else -1
        # if time_length <= threshold:
        #     X = np.mean(X, axis=1)
        # else:
        #     X = np.mean(X[:, :threshold], axis=1)

        time_length = X.shape[1]
        time_length = time_length if time_length <= 320 else 320

        sum_spike = np.sum(X[:, 0:time_length], axis=1)

        X = sum_spike

        with torch.no_grad():
            x = Variable(torch.from_numpy(np.asarray([X.tolist()]))).float()
            y_pred = self.model(x)

        return int(torch.argmax(y_pred, dim=1).item())


if __name__ == "__main__":
    src_dir = os.path.join(os.path.abspath(__file__), '..')
    mat_path = os.path.join(src_dir, 'monkeydata_training.mat')

    # classifier = Classifier(mat_path)

    # trainer = Trainer(mat_path)
    # trainer.k_fold_cv(1)
    # # trainer.run()
    # # trainer.initial_position_checker()
    #
    # trainer.velocity_checker()

    solution = CNN_Classifier(mat_path)
    solution.fit()
