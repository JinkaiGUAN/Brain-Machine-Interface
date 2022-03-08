function [x,y] = positionEstimator(test_data, modelParameters)
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