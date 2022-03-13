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
timeLength = size(test_data.spikes, 2);
if timeLength <= 340
    firingRate = sum(test_data.spikes(:, 1:timeLength), 2)';
else 
    firingRate = sum(test_data.spikes(:, 1:340), 2)';
end
%%% Knn
predictLabel = KnnEstimator(firingRate, modelParameters.Knn, 12);
%   predictLabel = bayesPredictor(firingRate, ...
%       modelParameters.bayes.statisticsSummaries, modelParameters.bayes.Num);
%
%   x = predictLabel;
%   y = 1;

%%% Linear regression
% [x,y] = linearRegressionEstimatorV2(test_data, modelParameters.regression, predictLabel);

[x,y] = positionEstimatorKalman(test_data, modelParameters.regression, predictLabel);
% x = predictLabel;
% y = 1;
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

% retrieve statistic information
stat = modelParameters.statistics{label};

spikes = test_data.spikes;
t = size(spikes,2);
n =  (t-320)/20;
bins = 300;
FR = [sum(spikes(:,t-bins:t),2)];
pX = [FR;1];

% Input scaling
pX = (pX(1:end-1) - stat.meanFeatures(1:end-1)); %./ stat.stdFeatures;

xy = modelParameters.linearRegression{label}*[pX; 1];

% reverse-mapping
xy = xy + stat.meanPosition;

xyP = xy + test_data.startHandPos.';


x = xyP(1);
y = xyP(2);

end

function [x,y] = positionEstimatorKalman(test_data, modelParameters, label)
%Teamname: Monkey Tricky
%Author: Kexin Huang; Zhongjie Zhang;  Peter Guan; Haonan Zhou.
%linear regression estimator
    spikes = test_data.spikes;
    t = size(spikes,2);
    bins = 299;
    timepoint = 300:20:t;
    n =  length(timepoint);
    obsZ = zeros(6,n);
    for i = 1:n
        FR = sum(spikes(:,timepoint(i)-bins:timepoint(i)),2);
%         FR = FR-modelParameters.meanFR{modelParameters.lable};
        pX = [FR;1];
        obsZ(:,i) = modelParameters.linearRegression{label}*pX;
    end
%     States = obsZ(1:2,n);    % for test linear Regresion Result.
    States = KalmanFilterProcess(obsZ, modelParameters.R{label}, n);
    Pxy = States + test_data.startHandPos;
%     load handposition to esimator for testing
%     xyP = test_data.handposition(:,t);   
    x = Pxy(1);
    y = Pxy(2);
%     if x^2+y^2>10000
%         x = x*(x/sqrt(x^2+y^2))
%         y = y*(y/sqrt(x^2+y^2))
end


%% Kalman filter for estimating

function [PositionFinal] = KalmanFilterProcess(ObsZ, R, nDataPoints)
%   nDataPoints: The number of data points.
%   
%   Detailed explanation goes here

    tStep = 20;
    A = [eye(2), tStep*eye(2), 0.5*tStep^2*eye(2);...
        zeros(2), eye(2), tStep*eye(2);...
        zeros(2),zeros(2), eye(2)];
    
    States = zeros([6, nDataPoints]);
    
    SigmaQ = 0.3;
    Q=[ SigmaQ^6/36  0  SigmaQ^5/12  0  SigmaQ^4/6 0;...
        0 SigmaQ^6/36  0  SigmaQ^5/12  0  SigmaQ^4/6;...
        SigmaQ^5/12  0  SigmaQ^4/4  0  SigmaQ^3/2 0;...
        0 SigmaQ^5/12  0  SigmaQ^4/4  0  SigmaQ^3/2;...
        SigmaQ^4/6   0   SigmaQ^3/2  0  SigmaQ^2 0;...
        0 SigmaQ^4/6   0   SigmaQ^3/2  0  SigmaQ^2];

    % Sigmatm1 = eye(6);
    Sigma = eye(6)*1;
    if exist('sigma.mat','file') ~= 0
        load('sigma.mat',"Sigma");
    end
    C = eye(6);
    % statetm1=zeros([6,1]);
    for t = 2:nDataPoints
        StatePrior = A*States(:,t-1);
        SigmaPrior = A*Sigma*A'+Q;
        K = SigmaPrior*C'/(C*SigmaPrior*C'+R);
        States(:,t) = StatePrior + K*(ObsZ(:,t) - C*StatePrior); % 6 (x, y, vel, ...) by N (2) - 
        Sigma = (eye(6)-K*C)*SigmaPrior;
       
    end
    
    %% todo: trim the data larger than 100 time step.

    save('sigma.mat',"Sigma",'-mat');
    PositionFinal = States(1:2, end);
    
end