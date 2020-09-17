function [J grad] = nnCostFunction(nn_params, ...
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

% Expand y output values into a matrix of single values
y_matrix = eye(num_labels)(y, :);

% Perform forward propagation
X = [ones(size(X, 1), 1) X];
a2 = sigmoid(Theta1 * X');
a2 = [ones(size(a2, 2), 1) a2'];
a3 = sigmoid(Theta2 * a2');

% Compute unregularized cost
for i = 1:m
    J = J + (-y_matrix(i, :) * log(a3(:, i)) - (1 - y_matrix(i, :)) * log(1 - a3(:, i)));
end
J_unreg = J / m;

% Compute regularization terms
reg = lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

% Compute regularized cost
J = J_unreg + reg;

% Implement backpropagation algorithm
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for i = 1:m

    % Feedforward pass
    a1 = X(i, :);
    a2 = sigmoid(Theta1 * a1');
    a2 = [ones(size(a2, 2), 1) a2'];
    a3 = sigmoid(Theta2 * a2');

    % Error in output layer
    d3 = a3 - y_matrix(i, :)';
      
    % Error in hidden layer
    d2 = Theta2(:, 2:end)' * d3 .* sigmoidGradient(Theta1 * a1');

    % Accumulate gradient
    D1 = D1 + d2 * a1;
    D2 = D2 + d3 * a2;
end

% Compute unregularized gradient
Theta1_grad_unreg = D1 / m;
Theta2_grad_unreg = D2 / m;

% Compute regularized gradient
reg1 = lambda / m .* Theta1;
reg2 = lambda / m .* Theta2;
reg1(:, 1) = 0; reg2(:, 1) = 0;
Theta1_grad = Theta1_grad_unreg + reg1;
Theta2_grad = Theta2_grad_unreg + reg2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
