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


% change y into a matrix based on numLabels
y1 = y * ones(1,num_labels);
ym = (y1 == (1:num_labels))';

[a2,h] = h_theta(Theta1, Theta2, X);
% cost function but with h_theta = forward propogation algo
J = sum(sum(-   ym  .*log(h) 
            - (1-ym).*log(1 - h)
        ) / m) + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + 
                                 sum(sum(Theta2(:,2:end).^2)));






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

X = [ones(m,1),X];
a2 = [ones(1,m);a2];
delta3 = h-ym ;         % (4,16)
size(Theta2');          % (5,4)
size(Theta2' * delta3); % (5,16)
size((a2.*(1-a2)));     % (4,16)
delta2 = (Theta2' * delta3) .* (a2.*(1-a2)); % (5,16)

Delta2 = zeros(size(delta3,1),size(a2,1));
sizeDelta3 = size(delta3);
Delta2 = delta3*a2'; 

delta2 = delta2(2:end,:);
Delta1 = delta2*X;

sTheta1 = size(Theta1); % (4,3)
sTheta2 = size(Theta2); % (4,5)

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;
sTheta1_grad = size(Theta1_grad);
sTheta2_grad = size(Theta2_grad);

    
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% zeros(size(Theta1,1),1)

reg1 = lambda / m * [ zeros(hidden_layer_size,1) , Theta1(:,2:end)];
reg2 = lambda / m * [ zeros(num_labels,1), Theta2(:,2:end)];
Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end