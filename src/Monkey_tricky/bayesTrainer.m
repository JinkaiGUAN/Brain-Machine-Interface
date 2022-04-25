function [statisticsSummaries, Num] = bayesTrainer(trainingData, timeRange)
%%% Input
% trainingData (struct): This should be specified as the name. Note this
%   should be `trial` name in the workspace.
% timeRange (List) : [startTime, endTime], i.e., a range of starting time
%   and end time in the sliding window.

%%% Output
% statisticsSummaries (cell): A cell of statistics information, where
%   each element has a dimention of 3 by 98. And rows represent mean,
%   standard deviarion and number of trials, respectively, which is calculated
%   according to the reaching angle. 
% Num (srtuct): 

    %%% 1. Preprocess the raw data
    
    % Reaching angle mapping
    reachingAngles = [30, 70, 110, 150, 190, 230, 310, 350];
    
    % Num struct stores the data statistics information (i.e., the size).
    Num.trial = size(trainingData, 1);
    Num.reachingAngle = size(trainingData, 2);
    Num.neuro = size(trainingData(1, 1).spikes, 1);
    Num.reachingAngles = reachingAngles;
    
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
            %% todo: Change the calculating mechanism of firing rate.

            % Calculate the firing rate for the given sliding window.
            neuroData = trainingData(trialIdx, reachingAngleIdx).spikes(:, ...
                timeRange(1) : timeRange(2));
            % Assign values for training features and labels.
            data.X(preIdx + trialIdx, :) = sum(neuroData, 2)';
            data.y(preIdx + trialIdx) = reachingAngles(reachingAngleIdx);
            
            singleClassVec(trialIdx, :) = cat(2, sum(neuroData, 2)', ...
                reachingAngles(reachingAngleIdx));
        end
        % Update preIdx
        preIdx = preIdx + Num.trial;
    
        % Append the seperated data;
        %%% The seperated data is a cell conatining the data for each label, where
        %%% the index is from 1 to 8 mapping to the reaching angles.
        seperated{reachingAngleIdx} = singleClassVec;
    end
    
    data.all = cat(2, data.X, data.y);
    
    %%% Summarize the class statistics
    
    for reachingAngleIdx = 1 : length(seperated)
        % retrieve the data for one reaching angle.
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

end







