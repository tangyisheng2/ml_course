function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hThetaX = X * theta;
%cost function
J = 0.5 / m *  sum(power(hThetaX - y,2));

%regularization
L = 0.5 * lambda / m * sum( power(theta(2:end),2));

%combime costfunction and regression
J = J + L;

%Gradient
% if size(theta) == 1
%     grad = 1/m * (hThetaX - y)'*X;
% else
%     grad = 1/m * (hThetaX - y)'*X + (lambda / m) .* theta(2:end);  %no regression for theta0()
% end 
% 

% regression
% 要确定后面的regression出来是一个2*1矩阵
% match grad，上面的写法出来是一个数，不能match grad
L = (lambda/m) .* theta;
L(1) = 0; % theta0不进行正则化

grad = ((1/m) .* X' * (X*theta - y)) + L;




% =========================================================================

grad = grad(:);

end
