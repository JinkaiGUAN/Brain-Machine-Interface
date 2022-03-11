function [PositionFinal] = KalmanFilterProcess(ObsZ, R, nDataPoints)
%   nDataPoints: The number of data points.
%   
%   Detailed explanation goes here

    tStep = 0.02;
    A = [eye(2), tStep*eye(2), 0.5*tStep^2*eye(2);...
        zeros(2), eye(2), tStep*eye(2);...
        zeros(2),zeros(2), eye(2)];
    
    States = zeros([6, nDataPoints]);
    
    SigmaQ = 0.3;
    Q=[ SigmaQ^6/36  0  SigmaQ^5/12  0  SigmaQ^4/6 0;...
        0 SigmaQ^6/36  0  SigmaQ^5/12  0  SigmaQ^4/6;...
        SigmaQ^5/12  0  SigmaQ^4/4  0  SigmaQ^3/2 0;...
        0 SigmaQ^5/12  0  SigmaQ^4/4  0  SigmaQ^3/2;...
        SigmaQ^4/6   0   SigmaQ^3/2  0  SigmaQ^2 0;...
        0 SigmaQ^4/6   0   SigmaQ^3/2  0  SigmaQ^2];

    % Sigmatm1 = eye(6);
    Sigma = eye(6)*1;
    C = eye(6);
    % statetm1=zeros([6,1]);

    for t = 2:nDataPoints
        StatePrior = A*States(:,t-1);
        SigmaPrior = A*Sigma*A'+Q;

        K = SigmaPrior*C'/(C*SigmaPrior*C'+R);
        States(:,t) = StatePrior + K*(ObsZ(:,t) - C*StatePrior);
        Sigma = (eye(6)-K*C)*SigmaPrior;
    end

    PositionFinal = States(1:2, end);
end
