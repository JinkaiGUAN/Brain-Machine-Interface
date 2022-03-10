function [data] = Knn(trainingData, timeRange, K, splitRatio)
    % trainingData (struct): This should be specified as the name. Note this
    %   should be `trial` name in the workspace.
    % timeRange (List) : [startTime, endTime], i.e., a range of starting time
    %   and end time in the sliding window.
    % K (int): How many neighbours should be counted or sampled in KNN
    %   algorithm.

    % Example:
    % >>> data = Knn(trial, [1, 320], 10);

    % Check the input variblaes. If you do not specifcy the K and splitRatio,
    % they will be default to 10 and 0.76.
    switch nargin
        case 2
            K = 10;
            splitRatio = 0.76;
        case 3
            splitRatio = 0.76;
        case 4
            disp("Input correctly in Knn function.");
        otherwise
           exception = MException(['Please check the input variable numbers,' ...
                ' you need to input training data and time range']);
           throw(exception);
    end

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
    
   
    % Get the size of all dataset
    [M.all, N] = size(data.X);
    % Shuffle the dataset 
    shuffleIdx = randperm(size(data.X, 1));
    data.X = data.X(shuffleIdx, :);             % All data for features
    data.y = data.y(shuffleIdx, 1);             % All data for labels
    % Split the dataset into real training and testing set accoridng to ratio of
    % 0.9, i.e., 10% of all data will be used as testing data.
    splitIdx = floor(M.all * splitRatio);
    data.trainingX = data.X(1 : splitIdx, :);   % traing dataset of features
    data.trainingY = data.y(1 : splitIdx, :);   % training datset of labels
    data.testX = data.X(splitIdx + 1 : end, :); % test dataset of features
    data.testY = data.y(splitIdx + 1 : end, :); % test dataset of labels
    
    %%% Perform single KNN
    % Get the size of taining and testing dataset
    M.train = size(data.trainingX, 1);  % Training trail number
    M.test = size(data.testX, 1);       % Test trail number

    % Initialize disatance and prediction matrix
    Dis = zeros(M.train, 1);            % Distance matrix
    predictYTest = zeros(M.test, 1);    % Predicted labels for test dataset
   
%     % Calculate the distance between test dataset and training dataset
%     for n = 1 : M.test
%         % n controls the index of each trail in the test dataset.
%         for i = 1 : M.train
%             % i controls the index of each trial in the training dataset.
%             %%% This part is going to check the distance between trail `n` and
%             %%% trail `i` for N neuros. 
%             distance1 = 0;
%             for j = 1 : N
%                 distance1 = distance1 + (data.testX(n, j) - data.trainingX(i, j)).^2;
%             end
%             Dis(i, 1) = distance1.^0.5;
%         end
% 
%         % Get the minimun k distance 
%         [~, index] = sort(Dis);
%         for k = 1 : K
%             temp(k) = data.trainingY(index(k));
%         end
%         % Samping the data
%         table = tabulate(temp);
%         % Get the maxinum number of happening
%         maxCount = max(table(:, 2, :)); 
%         % Get the label of maxCount 
%         labelIdx = find(table(:, 2, :) == maxCount);
%         % Here we only select the first index
%         %%% todo: How to tackle the problem of the multiple label with the same
%         %%% counting number.
%         predictYTest(n) = table(labelIdx(1), 1);
%         fprintf('True label: %d Predict label: %d\n', data.testY(n, 1), predictYTest(n))        
%     end
% 
%     % Calcualte the classification accuracy
%     acc = mean(predictYTest == data.testY(:, 1));
%     fprintf('Classification acc: %.2f\n', acc);
    
    %%% todo: Using cumulative gassuain function to fit the x and y data.
    
   
% End  of the function
end