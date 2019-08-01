function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%cost function
htheta = sigmoid(X * theta);
J = 1 / m * sum(-y .* log(htheta) - (1 - y) .* log(1 - htheta));    %不带惩罚项的cost funtion
L = 0;  %惩罚项
for j=2:size(theta) %从2起计
    L = L + 0.5*lambda/m*theta(j).^2;
end
J = J + L;


%grad
grad(1) = 1./m*(X(:,1))'*(sigmoid(X*theta)- y  );
for i = 2:size(theta, 1)
    grad(i) = 1./m*(X(:,i))'*(sigmoid(X*theta)- y  )+ lambda/m*theta(i);
end



% =============================================================

end
