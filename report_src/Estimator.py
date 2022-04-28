# -*- coding: UTF-8 -*-
"""
@Project : 3-coursework
@File    : estimation.py
@IDE     : PyCharm
@Author  : Peter
@Date    : 19/04/2022 16:59
@Brief   :
"""
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error

from classification import Classifier
from configuration import Configuration
from preprocess import RetrieveData
from preprocess import Trial
from sampling_window_split import SPlitRegression
from linear_regression import Linear_Regression,Segmented_Linear_Regression
from KalmanRegression import RegressionModel

# Configure the global configuration for plotting
plot_config = {
    "font.family": 'Times New Roman',
    "font.size": 24,
    "mathtext.fontset": 'stix',
    "axes.titlesize": 30
}
rcParams.update(plot_config)

config = Configuration()


class Estimation:
    def __init__(self, data_path: str):
        # bin windows
        self.window_width = config.time_window_width
        self.bin_width = config.bin_width

        self.data = scio.loadmat(data_path).get('trial')

        params = {
            "n_neighbors": 30,
            "algorithm": "ball_tree",
        }
        self.classifier = Classifier(model_name='KNN', params=params)

        if config.model_name == config.split_regression:
            self.regressor = SPlitRegression(self.data[:51, :], bin_width=self.bin_width,
                                         window_width=self.window_width, isRidge=False)
        elif config.model_name == config.split_ridge_regression:
            self.regressor = SPlitRegression(self.data[:51, :], bin_width=self.bin_width,
                                         window_width=self.window_width, isRidge=True)
        elif config.model_name == config.simple_linear_regression:
            # self.regressor = RegressionModel(data_path)
            self.regressor = Linear_Regression(self.data[:51, :],isRidge = False)
        elif config.model_name == config.simple_ridge_regression:
            # self.regressor = RegressionModel(data_path)
            self.regressor = Linear_Regression(self.data[:51, :],isRidge = True)
        elif config.model_name == config.segmented_linear_regression:
            # self.regressor = RegressionModel(data_path)
            self.regressor = Segmented_Linear_Regression(self.data[:51, :],isRidge = False)
        elif config.model_name == config.segmented_ridge_regression:
            # self.regressor = RegressionModel(data_path)
            self.regressor = Segmented_Linear_Regression(self.data[:51, :],isRidge = True)

        # classification data
        self.classification_training_data = RetrieveData(self.data[:51, :], bin_width=self.bin_width,
                                                         window_width=self.window_width, valid_start=0, valid_end=340,
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
        self.regressor.fit()

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

    def regression_predict(self, spikes: np.ndarray, label: int, initial_position: np.ndarray) -> t.Tuple[float, float]:
        """ The prediction part using regression.

        Args:
            spikes ():
            label ():
            initial_position (np.ndarray): initial position of hands, with size of 2 by 1.

        Returns:

        """
        # Using splitting window for data
        pos_x, pos_y = self.regressor.predict(spikes, label, initial_position)

        return pos_x, pos_y

    def rsme_xy(self, pre_flatx: t.List, pre_flaty: t.List, flat_x: t.List, flat_y: t.List) -> float:
        squared_numbersx = [number ** 2 for number in pre_flatx]
        squared_numbersy = [number ** 2 for number in pre_flaty]
        sum_list1 = [a + b for a, b in zip(squared_numbersx, squared_numbersy)]
        s11 = np.sqrt(sum_list1)
        fsquared_numbersx = [number ** 2 for number in flat_x]
        fsquared_numbersy = [number ** 2 for number in flat_y]
        sum_list2 = [a + b for a, b in zip(fsquared_numbersx, fsquared_numbersy)]
        s22 = np.sqrt(sum_list2)

        rmsall = np.sqrt(mean_squared_error(s11, s22)).item()
        return rmsall

    def test(self):

        fig = plt.figure(figsize=(13, 10))

        # helper parameters for accuracy calculation use
        correct_count = 0
        sampling_data_num = 0
        angles_set = set()
        flag = False

        # Collect x and y position information
        raw_flat_x = []
        raw_flat_y = []
        pre_flat_x = []
        pre_flat_y = []
        for trail_idx in range(51, self.trail_num,1):
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
                    initial_position = np.asarray(
                        [[raw_single_trail.initial_hand_pos_x], [raw_single_trail.initial_hand_pos_y]])
                    hand_pos_x_pred, hand_pos_y_pred = self.regression_predict(spikes, label, initial_position)
                    # hand position
                    #if abs(hand_pos_x_pred) < 200 and abs(hand_pos_y_pred<200):
                    hand_positions_x.append(float(hand_pos_x_pred))
                    hand_positions_y.append(float(hand_pos_y_pred))
                    #else:
                        #print(angle_idx)
                        #print(hand_pos_y_pred,hand_pos_x_pred)

                    # calculate classification accuracy
                    if label == angle_idx:
                        correct_count += 1
                    sampling_data_num += 1

                    # Collect the raw hand position data
                    raw_flat_x.append(raw_single_trail.hand_pos_x)
                    raw_flat_y.append(raw_single_trail.hand_pos_y)

                pre_flat_x += hand_positions_x
                pre_flat_y += hand_positions_y

                # # plot the graph
                if not flag:
                    flag = True
                    plt.plot(raw_single_trail.hand_pos_all_x, raw_single_trail.hand_pos_all_y, c="black", alpha=0.5,
                             label="Original path")
                plt.plot(raw_single_trail.hand_pos_all_x, raw_single_trail.hand_pos_all_y, c="black", alpha=0.5)

                if angle_idx not in angles_set:
                    angles_set.add(angle_idx)
                    plt.plot(np.asarray(hand_positions_x), np.asarray(hand_positions_y), c=self.colors[angle_idx],
                             label=f"{self.angle_mapping[angle_idx]}$^\circ$")
                plt.plot(np.asarray(hand_positions_x), np.asarray(hand_positions_y), c=self.colors[angle_idx])

        rmse_val = self.rsme_xy(pre_flat_y, pre_flat_x, raw_flat_x, raw_flat_y)
        print(f"The RMSE value is {rmse_val}.")
        plt.xlabel("Distance along x-axis")
        plt.ylabel("Distance along y-axis")
        plt.title("Monkey hand position distribution")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/prediction.svg", format='svg', dpi=1600, bbox_inches='tight')
        plt.show()
        # plt.savefig("../figures/prediction.svg", format='svg', dpi=1600, bbox_inches='tight')
        print("classification accuracy: ", np.round(correct_count / sampling_data_num, 3))

    def run(self):
        """Main function to run the whole process"""
        # Train the model
        self.train_model()

        # test the model
        self.test()


if __name__ == "__main__":
    import os

    src_dir = os.path.join(os.path.abspath(__file__), '..')
    mat_path = os.path.join(src_dir, 'monkeydata_training.mat')
    # mat_path = '/Users/huangkexin/Desktop/BMI_compition/Brain-Machine-Interface/report_src/monkeydata_training.mat'
    print(mat_path)

    # classifier = Classifier(mat_path)

    # trainer = Trainer(mat_path)
    # trainer.k_fold_cv(1)
    # # trainer.run()
    # # trainer.initial_position_checker()
    #
    # trainer.velocity_checker()
    estimation = Estimation(mat_path)
    estimation.run()

    # from sklearn.linear_model import LinearRegression
    # LinearRegression()
