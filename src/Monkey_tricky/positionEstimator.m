function [x,y] = positionEstimator(test_data, modelParameters)
%Teamname: Monkey Tricky
%Author: Kexin Huang; Zhongjie Zhang;  Peter Guan; Haonan Zhou.
%linear regression estimator
%     spikes = test_data.spikes;
%     t = size(spikes,2);
%     if t <= 340
%         firingRate = sum(test_data.spikes(:, 1:t), 2)';
%     else 
%         firingRate = sum(test_data.spikes(:, 1:340), 2)';
%     end
%%% Knn
    firingRate = sum(test_data.spikes, 2); % 98 by 1
%     predictLabel = KnnEstimator(firingRate, modelParameters.Knn, 12);
    predictLabel = FCNNReLU(firingRate);

    x = predictLabel;
    y = 1;
% 
%     if t <= 360
%         x = ployPredict(t,modelParameters.ployx{predictLabel,3});
%         y = ployPredict(t,modelParameters.ployy{predictLabel,3});
%     elseif t <= modelParameters.endTime(predictLabel)
%         x = ployPredict(t,modelParameters.ployx{predictLabel,1});
%         y = ployPredict(t,modelParameters.ployy{predictLabel,1});
%     else
%         x = ployPredict(t,modelParameters.ployx{predictLabel,2});
%         y = ployPredict(t,modelParameters.ployy{predictLabel,2});
%     end
%     
% 
%     if t > 640
%         if x >(modelParameters.meanend(1,predictLabel)+ modelParameters.meanend(1,predictLabel))
%             x = modelParameters.meanend(1,predictLabel);
%         end
%         if y >(modelParameters.meanend(2,predictLabel)+ modelParameters.meanend(2,predictLabel))
%             y = modelParameters.meanend(2,predictLabel);
%         end
%     end
% 
%     x = x + test_data.startHandPos(1);
%     y = y + test_data.startHandPos(2);

end

function [y] = ployPredict(x,beta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    n = length(beta);
    y = 0;
    for i = 1:n
        y = y + beta(i)*x^(n-i);
    end
end

%% Knn distance
function [predictLabel] = KnnEstimator(firingRate, modelParameters, K)
% Knn estimator 

%%% Inputs 
% - firingRate (List): The firing rate of the raw testing data. It can be
%       dimension of [n, 98], where n is the rows of testing data and 98 is 
%       the number of neuros.
% - modelParamsters (struct): 
%       This parameters should be gotten directly from function `KnnTrainer`.
%       X (List): The neuro information generated in function `KnnTrainer`.
%       y (List): The label of the corresponding trials. However, they are true
%           labels, i.e., 30 degress for instance. 

%%% Output
% - predictLabel (List): The predicted class for testing data, i.e., firring
%       Rate. It has a dimension of [n, 98], where n is the trail numbers, and
%       98 is the number of neuros. In this project, n default to 1. 

% Create mapping from index to true reaching angles.
keySet = {30, 70, 110, 150, 190, 230, 310, 350};
valueSet = [1, 2, 3, 4, 5, 6, 7, 8];
reachingAngleMapping = containers.Map(keySet,valueSet);

M.train = size(modelParameters.X, 1);  % Training trail number
[M.test, N] = size(firingRate);        % Dimension of testing data

% Calculate the distance between test dataset and training dataset
for n = 1 : M.test
    % n controls the index of each trail in the test dataset.
    for i = 1 : M.train
        % i controls the index of each trial in the training dataset.
        %%% This part is going to check the distance between trail `n` and
        %%% trail `i` for N neuros.
        distance1 = 0;
        for j = 1 : N
            distance1 = distance1 + (firingRate(n, j) - modelParameters.X(i, j)).^2;
        end
        Dis(i, 1) = distance1.^0.5;
    end

    % Get the minimun k distance
    [~, index] = sort(Dis);
    for k = 1 : K
        temp(k) = modelParameters.y(index(k));
    end
    % Samping the data
    table = tabulate(temp);
    % Get the maxinum number of happening
    maxCount = max(table(:, 2, :));
    % Get the label of maxCount
    labelIdx = find(table(:, 2, :) == maxCount);
    % Here we only select the first index
    predictLabel(n) = reachingAngleMapping(table(labelIdx(1), 1));
end
end