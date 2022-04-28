function [x,y] = positionEstimatorKalman(test_data, modelParameters)
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
        obsZ(:,i) = modelParameters.linearRegression{modelParameters.lable}*pX;
    end
%     States = obsZ(1:2,n)    % for test linear Regresion Result.
    States = KalmanFilterProcess(obsZ, modelParameters.R{modelParameters.lable}, n);
    Pxy = States + test_data.startHandPos;
%     load handposition to esimator for testing
%     xyP = test_data.handposition(:,t);   
    x = Pxy(1);
    y = Pxy(2);
%     if x^2+y^2>10000
%         x = x*(x/sqrt(x^2+y^2))
%         y = y*(y/sqrt(x^2+y^2))
end