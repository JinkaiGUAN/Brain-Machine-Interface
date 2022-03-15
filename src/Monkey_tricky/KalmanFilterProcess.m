function [PositionFinal] = KalmanFilterProcess(ObsZ, R, nDataPoints)
%   nDataPoints: The number of data points.
%   
%   Detailed explanation goes here

    tStep = 20; % ms
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
%     Sigma = eye(6)*1;
    Sigma = [   13.1953    2.0629    0.1273    0.0629   -0.0054   -0.0005
                2.0629    2.6261   -0.0238    0.0234   -0.0011   -0.0010
                0.1273   -0.0238    0.0107    0.0019   -0.0000    0.0000
                0.0629    0.0234    0.0019    0.0024   -0.0001    0.0000
                -0.0054   -0.0011   -0.0000   -0.0001    0.0000    0.0000
                -0.0005   -0.0010    0.0000    0.0000    0.0000    0.0000];
    C = eye(6);
    % statetm1=zeros([6,1]);

    for t = 2:nDataPoints
        StatePrior = A*States(:,t-1);
        SigmaPrior = A*Sigma*A'+Q;

        K = SigmaPrior*C'/(C*SigmaPrior*C'+R);
        States(:,t) = StatePrior + K*(ObsZ(:,t) - C*StatePrior);
        Sigma = (eye(6)-K*C)*SigmaPrior;
    end
%     disp(Sigma);

    PositionFinal = States(1:2, end);
end
