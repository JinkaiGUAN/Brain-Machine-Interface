% function [modelParameters] = positionEstimatorTraining(training_data)
%   % Arguments:
%   
%   % - training_data:
%   %     training_data(n,k)              (n = trial id,  k = reaching angle)
%   %     training_data(n,k).trialId      unique number of the trial
%   %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
%   %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
%   
%   % ... train your model
%   
%   % Return Value:
%   
%   % - modelParameters:
%   %     single structure containing all the learned parameters of your
%   %     model and which can be used by the "positionEstimator" function.
%   modelParameters = 1;
% 
%   
% 
% end

clc;
clear;

load monkeydata_training.mat;

timeRange = [1, 340];

trainingData = trial; %(1:80, :);

%% 1. Preprocess the raw data

% Reaching angle mapping
reachingAngles = [30, 70, 110, 150, 190, 230, 310, 350];

% Num struct stores the data statistics information (i.e., the size).
Num.trial = size(trainingData, 1);
Num.reachingAngle = size(trainingData, 2);
Num.neuro = size(trainingData(1, 1).spikes, 1);

% Initialize the firaing rate: Here using the neuroIdx can retrieve the whole
% firing rate data over reaching angles and trials. This would be treated as
% our training data. 

% Initialize training feature set, with dimension of (100 * 8) by 98.
data.X = zeros(Num.trial * Num.reachingAngle, Num.neuro);
% Initialize training label set, with dimension of (100 * 8) by 1.
data.y = zeros(Num.trial * Num.reachingAngle, 1);

preIdx = 0;  % This is to refer the updating indexes

for reachingAngleIdx = 1 : Num.reachingAngle
    singleClassVec = zeros(Num.trial, Num.neuro + 1);  % Add 1 for the class label

    for trialIdx = 1 : Num.trial
        % Calculate the firing rate for the given sliding window.
        neuroData = trainingData(trialIdx, reachingAngleIdx).spikes(:, ...
            timeRange(1) : timeRange(2));
        % Assign values for training features and labels.
        data.X(preIdx + trialIdx, :) = sum(neuroData, 2)';
        data.y(preIdx + trialIdx) = reachingAngles(reachingAngleIdx);
        
        singleClassVec(trialIdx, :) = cat(2, sum(neuroData, 2)', reachingAngles(reachingAngleIdx));
    end
    % Update preIdx
    preIdx = preIdx + Num.trial;

    % Append the seperated data;
    %%% The seperated data is a cell conatining the data for each label, where
    %%% the index is from 1 to 8 mapping to the reaching angles.
    seperated{reachingAngleIdx} = singleClassVec;
end

data.all = cat(2, data.X, data.y);

%% Summarize the class statistics

for reachingAngleIdx = 1 : length(seperated)
    singleClassVec = seperated{reachingAngleIdx};
    % Calculate the statistics information, i.e.m mean, std and counting.

    neuroSize = size(singleClassVec, 2) - 1;
    
    meanVal = mean(singleClassVec, 1);
    stdVal = std(singleClassVec, 0, 1);
    counting = size(singleClassVec, 1);

    singleClassStatistics = zeros(3, neuroSize);
    singleClassStatistics(1, :) = meanVal(1 : end - 1);
    singleClassStatistics(2, :) = stdVal(1 : end - 1);
    singleClassStatistics(3, :) = counting;

    statisticsSummaries{reachingAngleIdx} = singleClassStatistics;
end

%% Predict the class




rows = data.all(700:800, :);
count = 0;

for i = 1: size(rows, 1)
    row = rows(i, :); %data.all(780, :); %rows(i, :);
%     row = data.all(650, :);
    probabilities = zeros(1, Num.reachingAngle);
    totalNum = size(data.all, 1);
    
    for reachingIdx = 1 : Num.reachingAngle
        classSummaries = statisticsSummaries{reachingIdx};
    
        probabilities(1, reachingIdx) = classSummaries(3, 1) / totalNum;
        
        for neuroIdx = 1 : Num.neuro
            meanVal = classSummaries(1, neuroIdx);
            stdVal = classSummaries(2, neuroIdx);
    
            probabilities(1, reachingIdx) = probabilities(1, reachingIdx) * calGaussianProb(row(1, neuroIdx), meanVal, stdVal);
        end
    end
    
    predictLabel = reachingAngles(find(probabilities == max(probabilities)));
    if row(end) == predictLabel
        count = count + 1;
    end
end

acc = count / size(rows, 1);
disp(['Accuracy: ', num2str(acc * 100), '%']);


function prob = calGaussianProb(neuroFiringRate, meanVal, stdVal)
    if stdVal < 1e-5
        stdVal = 1e-9;
    end
    exponent = exp(-((neuroFiringRate - meanVal)^2 / (2 * stdVal^2 )));
   
    prob =  (1 / (sqrt(2 * pi) * stdVal)) * exponent;
%     disp(prob)
end



