function [modelParameters] = positionEstimatorTraining(training_data)
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
                onePxy(:,t) = xyVxyAxy{i,j}(1:2,timebin(t));
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