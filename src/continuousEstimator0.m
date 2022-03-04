%%% Team Members: Peter GUAN
%%% BMI Spring 2015 (Update 17th March 2015)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         PLEASE READ BELOW            %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function positionEstimator has to return the x and y coordinates of the
% monkey's hand position for each trial using only data up to that moment
% in time.
% You are free to use the whole trials for training the classifier.

% To evaluate performance we require from you two functions:

% A training function named "positionEstimatorTraining" which takes as
% input the entire (not subsampled) training data set and which returns a
% structure containing the parameters for the positionEstimator function:
% function modelParameters = positionEstimatorTraining(training_data)
% A predictor named "positionEstimator" which takes as input the data
% starting at 1ms and UP TO the timepoint at which you are asked to
% decode the hand position and the model parameters given by your training
% function:

% function [x y] = postitionEstimator(test_data, modelParameters)
% This function will be called iteratively starting with the neuronal data 
% going from 1 to 320 ms, then up to 340ms, 360ms, etc. until 100ms before 
% the end of trial.


% Place the positionEstimator.m and positionEstimatorTraining.m into a
% folder that is named with your official team name.

% Make sure that the output contains only the x and y coordinates of the
% monkey's hand.

%% KNN

data = Knn(trial, [1, 320], 10);

function [data] = Knn(trainingData, timeRange, K)
    % trainingData (struct): This should be specified as the name. Note this
    %   should be `trial` name in the workspace.
    % timeRange (List) : [startTime, endTime], i.e., a range of starting time
    %   and end time in the sliding window.
    % K (int): How many neighbours should be counted or sampled in KNN
    %   algorithm.
    
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
            data.X(preIdx + reachingAngleIdx, :) = mean(neuroData, 2)';
            data.y(preIdx + reachingAngleIdx) = reachingAngles(reachingAngleIdx);
        end
        % Update preIdx
        preIdx = preIdx + Num.reachingAngle;
    end
    
    %%% todo: perform k-fold cross-validation

    % Get the size of all dataset
    [M.all, N] = size(data.X);
    % Shuffle the dataset 
    shuffleIdx = randperm(size(data.X, 1));
    data.X = data.X(shuffleIdx, :);
    data.y = data.y(shuffleIdx, 1);
    % Split the dataset into real training and testing set accoridng to ratio of
    % 0.9, i.e., 10% of all data will be used as testing data, and cross
    % validation will be used to veriy the algorithm.     
    splitIdx = floor(M.all * 0.76);
    data.trainingX = data.X(1 : splitIdx, :);   % traing dataset of features
    data.trainingY = data.y(1 : splitIdx, :);   % training datset of labels
    data.testX = data.X(splitIdx + 1 : end, :); % test dataset of features
    data.testY = data.y(splitIdx + 1 : end, :); % test dataset of labels
    
    % fixme: Visualize the data. Normally, there should be two dimension, x-axis should
    % be the firing rate, and y axis is the label value? Also different sorts of
    % class should be labelled with legend. 
    
    %%% Perform single KNN
    % Get the size of taining and testing dataset
    M.train = size(data.trainingX, 1);  % Training trail number
    M.test = size(data.testX, 1);       % Test trail number

    % Initialize disatance and prediction matrix
    Dis = zeros(M.train, 1);            % Distance matrix
    predictYTest = zeros(M.test, 1);    % Predicted labels for test dataset

    % Calculate the distance between test dataset and training dataset
    for n = 1 : M.test
        % n controls the index of each trail in the test dataset.
        for i = 1 : M.train
            % i controls the index of each trial in the training dataset.
            %%% This part is going to check the distance between trail `n` and
            %%% trail `i` for N neuros. 
            distance1 = 0;
            for j = 1 : N
                distance1 = distance1 + (data.testX(n, j) - data.trainingX(i, j)).^2;
            end
            Dis(i, 1) = distance1.^0.5;
        end

        % Get the minimun k distance 
        [~, index] = sort(Dis);
        for k = 1 : K
            temp(k) = data.trainingY(index(k));
        end
        % Samping the data
        table = tabulate(temp);
        % Get the maxinum number of happening
        maxCount = max(table(:, 2, :)); 
        % Get the label of maxCount 
        labelIdx = find(table(:, 2, :) == maxCount);
        % Here we only select the first index
        %%% todo: How to tackle the problem of the multiple label with the same
        %%% counting number.
        predictYTest(n) = table(labelIdx(1), 1);
        fprintf('True label: %d Predict label: %d\n', data.testY(n, 1), predictYTest(n))        
    end

    % Calcualte  the classification accuracy
    acc = mean(predictYTest == data.testY(:, 1));
    fprintf('Classification acc: %.2f\n', acc);
    
   
% End  of the function
end



 %% 

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

end

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
  
  
  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
   
end

