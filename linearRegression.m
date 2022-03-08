function [Parameter] = linearRegression(y, X)
% Input: y: matrix (nlabel,nSamlpe)
%        X: matrix (nfeature,nSamlpe)
%Output: Parameter: matrix(nfeature,nlabel)
%To calculate coefficient estimates for a model with constant terms (intercepts), 
% include a column consisting of 1 in the matrix X.
    Parameter = pinv(X*X.')*X*y.';
%To calculate y: y = X.'*Parameter;
end