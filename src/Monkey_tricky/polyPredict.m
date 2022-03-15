function [y] = polyPredict(x,beta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    n = length(beta);
    y = 0;
    for i = 1:n
        y = y + beta(i)*x(i)^(i-1);
    end
end