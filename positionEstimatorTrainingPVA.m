function [modelParameters] = positionEstimatorTrainingPVA(training_data)
%preprocessing firing rate bins = 300, from 320ms, sliding step = 20ms
firingRate = cell(size(training_data,1),8); % cell size: 100trial x 8 angular
bins = 300;
xyVxyAxy = cell(size(training_data,1),8); % cell size: 100trial x 8 angular
for i = 1:size(training_data,1)
    for j = 1:size(training_data,2)
        position = zeros(2,size(training_data(i,j).handPos,2));
        positionDiff = zeros(2,size(training_data(i,j).handPos,2));
        positionDD = zeros(2,size(training_data(i,j).handPos,2));
        for k = 1: 2
            position(k,:) = training_data(i,j).handPos(k,:);
            positionDiff(k,2:size(training_data(i,j).handPos,2)) = diff(training_data(i,j).handPos(k,:));
            positionDD(k,3:size(training_data(i,j).handPos,2)) = diff(training_data(i,j).handPos(k,:),2);
        end
        xyVxyAxy{i,j} = [position;positionDiff;positionDD]; 
        %6(PositionX,Y VelocityX,Y AccX,Y) x nsample in one trial (sliding step = 20ms)
    end
end  

totalFR = []; % total firing rate: 98(neuron number) x nsample in all trial
VAC = []; %size: 6(PositionX,Y VelocityX,Y AccX,Y) x nsample in all trial (sliding step = 20ms)
for i = 1:size(training_data,1)
    for j = 1:size(training_data,2)
        timebin = 320:20:(size(training_data(i,j).spikes,2)-80);
        onefiringRate = zeros(size(training_data(i,j).spikes,1),length(timebin));
        oneVAC = zeros(6,length(timebin));
        % firing rate: 98cell x bins number
            for t = 1:length(timebin)
                onefiringRate(:,t) = sum(training_data(i,j).spikes(:,timebin(t)-bins:timebin(t)),2);
                oneVAC(:,t) = [xyVxyAxy{i,j}(1:2,timebin(t))-xyVxyAxy{i,j}(1:2,1);...
                
                xyVxyAxy{i,j}(3:4,timebin(t));...
                    xyVxyAxy{i,j}(5:6,timebin(t))];
            end
%          plot(oneVxy(1,:),'b')
        totalFR = [totalFR,onefiringRate];
        VAC = [VAC,oneVAC];
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
%modelParameters = linearRegression(VAC,X);
modelParameters = pinv(X*X.')*X*VAC.';
end

%linear regression-- least square estimation
