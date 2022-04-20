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
import matplotlib.pyplot as plt
from matplotlib import rcParams

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


class Classifier:
    def __init__(self, model_name: str = None, params: t.Dict = None) -> None:
        """In this project, accuracy can be more important than MSE, thus, in the comparison of accuracy of two
        naive classifiers, we concluded that KNN is more robust for firring rate data."""

        # todo: add more techniques in model name;
        params = {
            "n_neighbors": 30,
            "algorithm": "ball_tree"
        }
        self.model = KNeighborsClassifier(**params)

        # self.model = GaussianNB()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> int:
        return int(self.model.predict(X).item())


class Trainer:
    def __init__(self, data_path: str) -> None:
        self.data = scio.loadmat(data_path).get('trial')

        self.training_data = RetrieveData(self.data[:51, :], valid_start=0, valid_end=340, isClassification=True)
        self.test_data = RetrieveData(self.data[51:, :], valid_start=0, valid_end=340, isClassification=True)

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
        self.training_data.hand_position_x

    def test(self):
        """Test function for all."""
        pass


class Estimation:
    def __init__(self, data_path: str):
        # bin windows
        self.window_width = 300
        self.bin_width = 30

        self.data = scio.loadmat(data_path).get('trial')

        params = {
            "n_neighbors": 30,
            "algorithm": "ball_tree",
        }
        self.classifier = Classifier(model_name='KNN', params=params)

        # classification data
        self.classification_training_data = RetrieveData(self.data[:51, :], valid_start=0, valid_end=340,
                                                         isClassification=True)

        # retrieve data information
        self.trail_num = self.data.shape[0]
        self.angle_num = self.data.shape[1]
        self.neuro_num = self.data[0, 0][1].shape[0]

        # color configurations
        self.colors = [plt.cm.tab20(i) for i in range(self.angle_num)]
        self.angle_mapping = [30, 70, 110, 150, 190, 230, 310, 350]

    def train_model(self) -> None:
        # train the classification model
        self.classifier.fit(self.classification_training_data.X, self.classification_training_data.y)

    def classifier_predict(self, x: np.ndarray) -> int:
        time_length = x.shape[1]

        threshold = 340
        # time_step = threshold if time_length > threshold else -1
        if time_length <= threshold:
            x = np.mean(x, axis=1)
        else:
            x = np.mean(x[:, :threshold], axis=1)

        label = self.classifier.predict(np.asarray([x.tolist()]))

        return label

    def test(self):

        fig = plt.figure(figsize=(13, 10))

        # helper parameters for accuracy calculation use
        correct_count = 0
        sampling_data_num = 0
        angles_set = set()

        for trail_idx in range(self.trail_num):
            for angle_idx in range(self.angle_num):
                raw_single_trail = Trial(self.data[trail_idx, angle_idx], 0, -1)

                # predict hand position
                hand_positions_x = []
                hand_positions_y = []
                for _start in range(0, len(raw_single_trail) - self.window_width + 1, self.bin_width):
                    raw_single_trail.valid_start, raw_single_trail.valid_end = 0, _start + self.window_width

                    # The all spikes for this specified time window
                    spikes = raw_single_trail.raw_firing_rate

                    # predict label
                    label = self.classifier_predict(spikes)
                    # hand position
                    hand_pos_x_pred = 1
                    hand_pos_y_pred = 2
                    hand_positions_x.append(hand_pos_x_pred)
                    hand_positions_y.append(hand_pos_y_pred)

                    # calculate classification accuracy
                    if label == angle_idx:
                        correct_count += 1
                    sampling_data_num += 1

                # plot the graph
                if angle_idx not in angles_set:
                    angles_set.add(angle_idx)
                    plt.plot(raw_single_trail.hand_pos_all_x, raw_single_trail.hand_pos_all_y, c=self.colors[angle_idx],
                             label=f"{self.angle_mapping[angle_idx]}$^\circ$")
                plt.plot(raw_single_trail.hand_pos_all_x, raw_single_trail.hand_pos_all_y, c=self.colors[angle_idx])
                plt.plot(hand_positions_x, hand_positions_y, c=self.colors[angle_idx])

        plt.xlabel("Distance along x-axis")
        plt.ylabel("Distance along y-axis")
        plt.title("Monkey hand position distribution")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.show()
        print("classification accuracy: ", np.round(correct_count / sampling_data_num, 3))

    def run(self):
        """Main function to run the whole process"""
        # Train the model
        # self.train_model()

        # test the model
        self.test()


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
    estimation = Estimation(mat_path)
    estimation.run()