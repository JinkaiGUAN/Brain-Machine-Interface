function [modelParameters] = linearRegressionKalmanTraining(training_data)
%Teamname: Monkey Tricky
%Author: Kexin Huang; Zhongjie Zhang;  Peter Guan; Haonan Zhou.
% firing rate window size = 300, from 320ms, sliding step = 20ms. 
%linear regression estimator
bins = 299;
modelParameters.linearRegression = cell(8);
modelParameters.R = cell(8);
modelParameters.meanFR = cell(8);
for j = 1:size(training_data,2)
    FR = [];
    PxyVxyAxy = [];
    for i = 1:size(training_data,1)
        timebin = 320:20:(size(training_data(i,j).spikes,2)-20);
        Vxy = diff(training_data(i,j).handPos(1:2,:),1,2);
        Axy = diff(training_data(i,j).handPos(1:2,:),2,2);
        % firing rate: 98cell x bins number
            for t = 1:length(timebin)
                FR = [FR,sum(training_data(i,j).spikes(:,timebin(t)-bins:timebin(t)),2)];
                PVA = [training_data(i,j).handPos(1:2,timebin(t))-...
                    training_data(i,j).handPos(1:2,timebin(1));...
                    Vxy(1:2,timebin(t)+1);Axy(1:2,timebin(t)-2)];
                PxyVxyAxy = [PxyVxyAxy,PVA]; % minus start position
            end
    end
%     modelParameters.meanFR{j} = mean(FR,2);
%     FR = FR - modelParameters.meanFR{j};
    n = size(FR,2);
    l = ones(1,n);
    X = [FR;l];
%     modelParameters.linearRegression{j} = zeros(2,99);
%     modelParameters.linearRegression{j}(1,:) = regress(Pxy(1,:).',X.');
%     modelParameters.linearRegression{j}(2,:) = regress(Pxy(2,:).',X.');
    modelParameters.linearRegression{j} = linearRegression(PxyVxyAxy,X);
    r = modelParameters.linearRegression{j}*X-PxyVxyAxy;
    modelParameters.R{j} = cov(r.');
end
modelParameters.lable = 5; % for test, you can change to int(1-8).
end