function [modelParameters] = positionEstimatorTraining(training_data)
%Teamname: Monkey Tricky
%Author: Kexin Huang; Zhongjie Zhang;  Peter Guan; Haonan Zhou.
% firing rate window size = 299, from 320ms, sliding step = 20ms. 
%knn
modelParameters.Knn = KnnTrainer(training_data, [1, 340]);
%linear regression estimator
bins = 300;
modelParameters.ployx = cell(8);
modelParameters.ployy = cell(8);
modelParameters.endTime = zeros(8);
modelParameters.meanend = zeros(2,8);
modelParameters.stdend = zeros(2,8);
for j = 1:size(training_data,2)
    onePxy = [];
    onetime = [];
    timeEnd = [];
    twotime = [];
    twoPxy = [];
    threetime = [];
    threePxy = [];
    for i = 1:size(training_data,1)
        t = size(training_data(i,j).spikes,2);
        timebin = 300:1:t;
        onetime = [onetime,timebin(62:length(timebin)-100)];
        onexy = training_data(i,j).handPos(1:2,361:t-100)-...
                    training_data(i,j).handPos(1:2,1);
        onePxy = [onePxy,onexy];

        timeEnd = [timeEnd,size(training_data(i,j).spikes,2)-120];
        twotime = [twotime,timebin(length(timebin)-100:length(timebin))];
        twoxy = training_data(i,j).handPos(1:2,t-100:t)-...
                    training_data(i,j).handPos(1:2,1);
        twoPxy = [twoPxy,twoxy];
        
        threetime = [threetime,timebin(1:61)];
        threexy = training_data(i,j).handPos(1:2,300:360)-...
                    training_data(i,j).handPos(1:2,1);
        threePxy = [threePxy,threexy];

         
    end
    %mid phase
    modelParameters.ployx{j,1} = ployjason(onetime,onePxy(1,:),2);
    modelParameters.ployy{j,1} = ployjason(onetime,onePxy(2,:),2);
    %stop phase
    modelParameters.ployx{j,2} = ployjason(twotime,twoPxy(1,:),2);
    modelParameters.ployy{j,2} = ployjason(twotime,twoPxy(2,:),2);
    modelParameters.endTime(j) = mean(timeEnd);
    modelParameters.meanend(:,j) = mean(twoPxy,2);
    modelParameters.stdend(:,j) = std(twoPxy,0,2);
    %start phase
    modelParameters.ployx{j,3} = ployjason(threetime,threePxy(1,:),2);
    modelParameters.ployy{j,3} = ployjason(threetime,threePxy(2,:),2);
end


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

function [beta] = ployjason(x,y,q)
    n = length(y);
    b = zeros(n,1);
    A = zeros(n,q+1);
    for i=1:n
        b(i) = y(i);
        for j=1:q+1
            A(i,j) = x(i)^(q+1-j);
        end
    end
    C = A'*A;
    d = A'*b;
    beta = C\d;
end