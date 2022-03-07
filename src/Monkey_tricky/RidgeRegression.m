function [Beta,Loss] = RidgeRegression(XTrain, YTrain, batchSize, alpha, epochs, phi)
%   The function for ridge regression.
%   Author: Haonan Zhou
%
%   Output:
%   ------Beta, the model parameters after training, with
%   size[nFeatures,nLabels]
%   ------Loss, the last value for the loss function, just for reference.
% 
%   Input: 
%   ------XTrain: The trainning data from the number of spikes activiated
%   times, with size [nSamples, nFeatrues].
%   ------YTrain: The trainning label data for deltaXPos, deltaXVel,
%   deltaAcc, with size [nSamples, nLabels].
%   
%   Parameters:
%   ------batchSize, the volume for every mini-batch, default:50.
%   ------alpha, the learning rate, default:0.001.
%   ------epochs, the times of iterations, default: 10000
%   ------phi, a parameter for ridge model, default:0.5.

    global nFeatures nLabels; %#ok<GVMIS> 
    
    nFeatures = 98; % = size(XTrain,2)
    nLabels = 2;    % = size(YTrain,2)

    betaK = ones([nFeatures, nLabels]);
    nSamples = size(XTrain, 1);

    index = randsample(nSamples, batchSize);
    for i = 1:epochs
        betaKPlus = betaK - alpha*lossGradient(XTrain(index, :), YTrain(index, :), batchSize,phi, betaK);
        betaK = betaKPlus;
    end
    Loss = lossFunction(XTrain, YTrain, batchSize,phi, beta);
    Beta = betaK;

end


function [Loss] = lossFunction(XTrain, YTrain, batchSize,phi, beta)

    Loss = (1/batchSize) * (YTrain - XTrain*beta)' * (YTrain - XTrain*beta) + phi * beta' * beta; %#ok<MHERM> 

end


function [gradient] = lossGradient(XTrain, YTrain, batchSize,phi, beta)

    gradientRecord = zeros([nFeaters, nLabels, batchSize]);
    
    for i = 1:batchSize
        gradientRecord(:,:, i) = -2*XTrain*(YTrain-XTrain'*beta) + 2*phi*beta;
    end
    
    gradient = mean(gradientRecord, 3);

end