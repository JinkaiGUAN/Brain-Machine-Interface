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


%%% Generate KNN Parameters
modelParameters.Knn = KnnTrainer(training_data, [1, 320]);

%%% Generate linear regression parameters
% modelParameters.regression = LinearRegression(training_data);
modelParameters.regression = PxylinearRegression(training_data);

end

%% KNN

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

%% Linear Regression

function [modelParameters] = LinearRegression(training_data)
%Teamname: Monkey Tricky
%Author: Kexin Huang; Zhongjie Zhang;  Peter Guan; Haonan Zhou.
% firing rate window size = 300, from 320ms, sliding step = 20ms. 
%linear regression estimator
firingRate = cell(size(training_data,1),8);
bins = 300;
totalFR = []; % total firing rate: 98 x total number data point
Pxy = [];
xyVxyAxy = cell(size(training_data,1),8);
for i = 1:size(training_data,1)
    for j = 1:size(training_data,2)
        position = zeros(2,size(training_data(i,j).handPos,2));
        positionDiff = zeros(2,size(training_data(i,j).handPos,2));
        positionDD = zeros(2,size(training_data(i,j).handPos,2));
        for k = 1: 2
            position(k,:) = training_data(i,j).handPos(k,:);
            positionDiff(k,2:size(training_data(i,j).handPos,2)) = diff(training_data(i,j).handPos(k,:));
            positionDD(k,2:size(training_data(i,j).handPos,2)) = diff(positionDiff(k,:));
        end
        xyVxyAxy{i,j} = [position;positionDiff;positionDD];
    end
end  

for i = 1:size(training_data,1)
    for j = 1:size(training_data,2)
        timebin = 320:20:(size(training_data(i,j).spikes,2)-80);
        onefiringRate = zeros(size(training_data(i,j).spikes,1),length(timebin));
        onePxy = zeros(2,length(timebin));
        % firing rate: 98cell x bins number
            for t = 1:length(timebin)
                onefiringRate(:,t) = sum(training_data(i,j).spikes(:,timebin(t)-bins:timebin(t)),2);
                onePxy(:,t) = xyVxyAxy{i,j}(1:2,timebin(t))-xyVxyAxy{i,j}(1:2,timebin(1)); % minus start position
            end
%          plot(oneVxy(1,:),'b')
        firingRate{i,j} = onefiringRate;
        totalFR = [totalFR,onefiringRate];
        Pxy = [Pxy,onePxy];
    end
end
% Falman filtering encoding
n = size(totalFR,2);
%Linear regression 
l = ones(1,n);
X = [totalFR;l];
%r = corr(X.',Vxy.')
% bX = regress(Vxy(1,:).',X.');
% bY = regress(Vxy(2,:).',X.');
% modelParameters = [bX,bY];
modelParameters = pinv(X*X.')*X*Pxy.';
end


function [modelParameters] = PxylinearRegression(training_data)
%Teamname: Monkey Tricky
%Author: Kexin Huang; Zhongjie Zhang;  Peter Guan; Haonan Zhou.
% firing rate window size = 300, from 320ms, sliding step = 20ms. 
%linear regression estimator
bins = 300;
FR = []; % total firing rate: 98 x total number data point
Pxy = [];
modelParameters.linearRegression = cell(8);
for j = 1:size(training_data,2)
    FR = [];
    Pxy = [];
    for i = 1:size(training_data,1)
        timebin = 320:20:(size(training_data(i,j).spikes,2)-80);
        % firing rate: 98cell x bins number
            for t = 1:length(timebin)
                FR = [FR,sum(training_data(i,j).spikes(:,timebin(t)-bins:timebin(t)),2)];
                xy = training_data(i,j).handPos(1:2,timebin(t)) - ...
                    training_data(i,j).handPos(1:2,timebin(1));
                Pxy = [Pxy,xy]; % minus start position
            end
    end
    n = size(FR,2);
    l = ones(1,n);
    X = [FR;l];
    modelParameters.linearRegression{j} = linearRegression(Pxy,X);
end
end

function [Parameter] = linearRegression(y, X)
% Input: y: matrix (nlabel,nSamlpe)
%        X: matrix (nfeature,nSamlpe)
%Output: Parameter: matrix(nfeature,nlabel)
%To calculate coefficient estimates for a model with constant terms (intercepts), 
% include a column consisting of 1 in the matrix X.
    Parameter = transpose(pinv(X*X.')*X*y.');
%To calculate y: y = Parameter*X;
end