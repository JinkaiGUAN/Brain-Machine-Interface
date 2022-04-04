% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

% function RMSE = testFunction_for_students_MTb(teamName)

% load monkeydata0.mat
load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

addpath('Monkey_tricky');

% Select training and testing data (you can choose to split your data in a different way if you wish)
%%% Getting the training and testing datset
trainingData = trial(ix(1:60),:);
testData = trial(ix(61:end),:);

%%% 
fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid

tic
% Train Model
modelParameters = positionEstimatorTraining(trainingData);
predictLabels = [];
trueLabels = [];
for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];

        times=340:20:size(testData(tr,direc).spikes,2); 

        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
         
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            predictLabels = [predictLabels, decodedPosX];
            trueLabels = [trueLabels, direc];

            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end

        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end

end

% Output classification accuracy
acc = sum(predictLabels == trueLabels) / length(trueLabels);
disp(['Classification accuracy: ', num2str(acc * 100), '%']);

legend('Decoded Position', 'Actual Position');

RMSE = sqrt(meanSqError/n_predictions);

% rmpath(genpath(teamName))

% end
toc

