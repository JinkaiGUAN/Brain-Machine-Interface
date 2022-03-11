function [x, y] = positionEstimator(test_data, modelParameters)

% **********************************************************
%
% You can also use the following function header to keep your state
% from the last iteration
%
% function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
%                 ^^^^^^^^^^^^^^^^^^
% Please note that this is optional. You can still use the old function
% declaration without returning new model parameters.
%
% *********************************************************

% - test_data:
%     test_data(m).trialID
%         unique trial ID
%     test_data(m).startHandPos
%         2x1 vector giving the [x y] position of the hand at the start
%         of the trial
%     test_data(m).decodedHandPos
%         [2xN] vector giving the hand position estimated by your
%         algorithm during the previous iterations. In this case, N is
%         the number of times your function has been called previously on
%         the same data sequence.
%     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
%     in this case, t goes from 1 to the current time in steps of 20
%     Example:
%         Iteration 1 (t = 320):
%             test_data.trialID = 1;
%             test_data.startHandPos = [0; 0]
%             test_data.decodedHandPos = []
%             test_data.spikes = 98x320 matrix of spiking activity
%         Iteration 2 (t = 340):
%             test_data.trialID = 1;
%             test_data.startHandPos = [0; 0]
%             test_data.decodedHandPos = [2.3; 1.5]
%             test_data.spikes = 98x340 matrix of spiking activity
% - modelParameters (struct):
% 


% ... compute position at the given timestep.

% Return Value:

% - [x, y]:
%     current position of the hand

%%% Classify the test data
firingRate = sum(test_data.spikes(:, 1:320), 2)';
%%% Knn
predictLabel = KnnEstimator(firingRate, modelParameters.Knn, 10);
%   predictLabel = bayesPredictor(firingRate, ...
%       modelParameters.bayes.statisticsSummaries, modelParameters.bayes.Num);
%
%   x = predictLabel;
%   y = 1;

%%% Linear regression
[x,y] = linearRegressionEstimatorV2(test_data, modelParameters.regression, predictLabel);


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

%% Lineare regression estimation

function [x,y] = linearRegressionEstimator(test_data, modelParameters)
%Teamname: Monkey Tricky
%Author: Kexin Huang; Zhongjie Zhang;  Peter Guan; Haonan Zhou.
%linear regression estimator

spikes = test_data.spikes;
t = size(spikes,2);
n =  (t-320)/20;
bins = 300;
FR = [sum(spikes(:,t-bins:t),2)];
pX = [FR;1];
xy = pX.'*modelParameters;
xyP = xy + test_data.startHandPos.';
%     load handposition to esimator for testing
%     xyP = test_data.handposition(:,t);   
x = xyP(1);
y = xyP(2);
end

function [x,y] = linearRegressionEstimatorV2(test_data, modelParameters, label)
%Teamname: Monkey Tricky
%Author: Kexin Huang; Zhongjie Zhang;  Peter Guan; Haonan Zhou.
%linear regression estimator
    spikes = test_data.spikes;
    t = size(spikes,2);
    n =  (t-320)/20;
    bins = 300;
    FR = [sum(spikes(:,t-bins:t),2)];
    pX = [FR;1];
    xy = modelParameters.linearRegression{label}*pX;
    xyP = xy + test_data.startHandPos.';
%     load handposition to esimator for testing
%     xyP = test_data.handposition(:,t);   
    x = xyP(1);
    y = xyP(2);
%     if x^2+y^2>10000
%         x = x*(x/sqrt(x^2+y^2))
%         y = y*(y/sqrt(x^2+y^2))
end