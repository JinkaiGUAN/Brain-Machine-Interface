function [modelParameters] = polyfitRegression(training_data)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% modelParameters.betaXP1 = zeros([3,8]);
modelParameters.betaXP2 = zeros([4,8]);
modelParameters.betaXP3 = zeros([3,8]);
% modelParameters.betaYP1 = zeros([3,8]);
modelParameters.betaYP2 = zeros([4,8]);
modelParameters.betaYP3 = zeros([3,8]);

    for label = 1:8
        posXP2 = [];
        posYP2 = [];
        timeRecordP2 = [];
        posXP3 = [];
        posYP3 = [];
        timeRecordP3 = [];
        for trial = 1:length(training_data,1)
            for t = 320:20:500
                % Part2
                posXP2 = [posXP2; training_data(trial, label).handPos(1, t)];
                posYP2 = [posYP2; training_data(trial, label).handPos(2, t)];
                timeRecordP2 = [timeRecordP2; t];
            end
            for t = 520:20:length(training_data(trial, label).handPos(2, :))
                % Part3
                posXP3 = [posXP3; training_data(trial, label).handPos(1, t)];
                posYP3 = [posYP3; training_data(trial, label).handPos(2, t)];
                timeRecordP3 = [timeRecordP3; t];
            end
        end
        modelParameters.betaXP2(:, label) = ployjason(timeRecordP2, posXP2, 3);
        modelParameters.betaYP2(:, label) = ployjason(timeRecordP2, posYP2, 3);
        modelParameters.betaXP3(:, label) = ployjason(timeRecordP3, posXP3, 2);
        modelParameters.betaYP3(:, label) = ployjason(timeRecordP3, posYP3, 2);
    end

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