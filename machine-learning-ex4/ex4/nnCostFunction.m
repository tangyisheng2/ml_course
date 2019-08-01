function [J,grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%先设定好神经网络的架构
a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = [ones(size(z2,1), 1) sigmoid(z2)]; %  hidden layer add bias

z3 = a2 * Theta2';
% z_3 = Theta2 * a_2';  为什么不能反过来乘？A:反过来边正向传播
a3 = sigmoid(z3);
hThetaX = a3;


%K有10个数字
K=10;

%计算代价函数


yVec = zeros(m,num_labels);

for i = 1:m
    yVec(i,y(i)) = 1;   %创建输出的矩阵 Eg:1 > [1;0;0;.....0]
end

J = -1/m * sum(sum(yVec.*log(hThetaX)+(1-yVec).*log(1-hThetaX)));

%regression terms
L = 0;
L = L + 0.5*lambda/m * (sum(sum(power(Theta1(:,2:end),2),1:400),1:25)+...
    sum(sum(power(Theta2(:,2:end),2),1:25),1:10));    
    %No regularization for bias
J = J + L;

%grad 反向传播
for i=1:m
    
    %输入层
    a1 = [1;X(i,:)'];   %add bias (ref:figure2)
    
    %第一层隐藏层
    z2 = Theta1 * a1;
    a2 = [1;sigmoid(z2)];
    
    %第三层输出层
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
%   hthetaX = a3;   %No Use
    
    yy = ([1:num_labels]==y(i))'; %求出y(i)对应的输出矩阵 Eg:1 > [1;0;0;.....0]
    
    
    delta3 = a3 - yy;   %对比预测值和实际值误差
    % Eg if yy = [1;0;0;.....0]
    %       y  = [0;1;0;.....0]
    % something like that
    
    delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
    delta2 = delta2(2:end); %取出bias
    
    % delta_1 is not calculated because we do not associate error with the input
    
    Theta1_grad = Theta1_grad + delta2  * a1';
    Theta2_grad = Theta2_grad + delta3 * a2';
    
end

% Theta1_grad = Theta1_grad + 0.5*lambda/m *sum(sum(sum(power(The,1:2),1:1),1:2)

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];   %add bias
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
