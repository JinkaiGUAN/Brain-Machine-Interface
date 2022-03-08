function RMSE = RMSEselftest(teamName)
% for self-test.
load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:70),:);
testData = trial(ix(71:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData);
for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
%     for direc=2
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            pastCurrentTrial.trialId = testData(tr,direc).trialId;
            pastCurrentTrial.spikes = testData(tr,direc).spikes(:,1:t); 
            pastCurrentTrial.decodedHandPos = decodedHandPos;
%               load handposition to esimator for testing

%             pastCurrentTrial.handposition = testData(tr,direc).handPos(1:2,1:t); 

            pastCurrentTrial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            [decodedPosX, decodedPosY] = positionEstimator(pastCurrentTrial, modelParameters);
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

legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions) 

rmpath(genpath(teamName))

end