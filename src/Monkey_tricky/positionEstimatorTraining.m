function [modelParameters] = positionEstimatorTraining(training_data)
% Arguments:

% - training_data:
%     training_data(n,k)              (n = trial id,  k = reaching angle)
%     training_data(n,k).trialId      unique number of the trial
%     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
%     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)

% ... train your model

% Return Value:

% - modelParameters:
%     single structure containing all the learned parameters of your
%     model and which can be used by the "positionEstimator" function.

%% Generate Bayes Parameters
% [statisticsSummaries, Num] = bayesTrainer(training_data, [1, 500]);
% modelParameters.bayes.statisticsSummaries = statisticsSummaries;
% modelParameters.bayes.Num = Num;


%% Generate KNN Parameters
modelParameters.Knn = KnnTrainer(training_data, [1, 320]);

end


function [modelParameters] = KnnTrainer(trainingData, timeRange)
% K-nearest-neighbour algorithm.

%%% Input 
% - training_data (struct):
%       training_data(n,k)              (n = trial id,  k = reaching angle)
%       training_data(n,k).trialId      unique number of the trial
%       training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
%       training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
% - timeRange (List): A list of time stamp with size of 2. The first element
%       is the index of beginning of counting window, and the second one is the
%       ending index of the sliding window.

%%% Returns
% - modelParameters (struct): The parameters that are going to be used when
%       classifying the test data (i.e., a single trial data for any reaching
%       angles). 

% Reaching angle mapping
reachingAngles = [30, 70, 110, 150, 190, 230, 310, 350];

% Num struct stores the data statistics information
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
for trialIdx = 1 : Num.trial
    for reachingAngleIdx = 1 : Num.reachingAngle
        % Calculate the firing rate for the given sliding window.
        neuroData = trainingData(trialIdx, reachingAngleIdx).spikes(:, ...
            timeRange(1) : timeRange(2));
        % Assign values for training features and labels.
        data.X(preIdx + reachingAngleIdx, :) = sum(neuroData, 2)';
        data.y(preIdx + reachingAngleIdx) = reachingAngles(reachingAngleIdx);
    end
    % Update preIdx
    preIdx = preIdx + Num.reachingAngle;
end

% Returns
modelParameters = data;
end