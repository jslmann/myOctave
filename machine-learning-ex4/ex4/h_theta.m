function [a1,p] = h_theta(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = h_theta(Theta1, Theta2, X) outputs the activation levels for 
%  final level of the network

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Add ones to the X data matrix
X = [ones(m, 1) X];
[m,n] = size(X);

a1 = sigmoid(Theta1 * X'); % 

a1p = [ones(m,1)'; a1];

p = sigmoid(Theta2 * a1p);

% =========================================================================


end