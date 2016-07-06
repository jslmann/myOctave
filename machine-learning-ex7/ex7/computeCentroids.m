function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
%size(kBool) = (m,K)
size_idx = size(idx)

kBool = squeeze(idx(:) == 1:K); 
size_kBool = size(kBool)
% dim 1 is the point
% dim 2 is the x,y
% dim 3 are the copies
% a different shape will mean that reshape messes things up

%size(fatX) = 300,2,K

% make K copies of X along dim 2 : eg. [X, X, X (k-times)]
Xcopies = repmat(X,1,K);

% push X copies into right shape
fatX = reshape(Xcopies,m,n,K);

% multi-dimensional transpose so that we can multiply with kBool
% this moves dim 3 to dim1, dim 1 ->2 and 2 -> 3
% fatXp -> (K,m,n)
fatXp = permute(fatX,[3,1,2]);

% mask FatXp with kBool 
% size(kBool) = (K,...)  size(fatXp) = (K, f...)
% kSeparated is

% size(kBool) -> (m,K)
size(kBool)
kSeparated = kBool' .* fatXp;


% summing over dimension 2 collapses so new size should be 3,2 - its actually 3,1,2
% we can remove the dimension with 1 with 'squeeze'
kSums = squeeze(sum(kSeparated,2));

centroids = kSums ./ sum(kBool,1)';

% =============================================================


end