
function [predictedLabel] = bayesPredictor(firingRate, bayesParams, Num)
%%% Input:
% firingRate (List): The firing rate for trained neuros along the time window
%   set by the corresponding test function.
% bayesParams (Cell): The statistics summaries calculated by the training
%   dataset using Naive Bayes Algorithm. `statisticsSummaries`
% Num (strcut): A structure contains the size of raw data. 
%   reachingleAngle (int): The number of labels, which is 8 in this project.
%   neuro (int): The number of neuros, it is 98 here.

%%% Output
% predictedLabel (int): The predicted angle of the input firing rate. 
    
    row = firingRate;
    probabilities = zeros(1, Num.reachingAngle);
    totalNum = size(Num.neuro * Num.trial, 1);
    
    for reachingIdx = 1 : Num.reachingAngle
        classSummaries = bayesParams{reachingIdx};
    
        probabilities(1, reachingIdx) = classSummaries(3, 1) / totalNum;
        
        for neuroIdx = 1 : Num.neuro
            meanVal = classSummaries(1, neuroIdx);
            stdVal = classSummaries(2, neuroIdx);
    
            probabilities(1, reachingIdx) = probabilities(1, reachingIdx) * calGaussianProb(row(1, neuroIdx), meanVal, stdVal);
        end
    end
    
%     predictedLabel = Num.reachingAngles(find(probabilities == max(probabilities)));
    predictedLabel = find(probabilities == max(probabilities));

end



