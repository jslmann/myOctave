function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% in english: for each point in X, I want a vector which is the 
% distance to each centroid


%for i = 1:size(X,1)
%    smallest_d = sum( (X(i,:)- centroids(1,:)).^2);
%    idx(i) = 1;
%    for k = 2:K;
%        d = sum( (X(i,:)- centroids(k,:)).^2);
%        if d < smallest_d;
%            smallest_d = d;
%            idx(i) = k;
%        end
%    end
%end


% in english: for each point in X, I want a vector which is the 
% distance to each centroid
[m,n] = size(X)
Xcopies = repmat(X,1,K); 
fatX = reshape(Xcopies,m,n,K);
fatXp = permute(fatX,[3,2,1]);

% ask this question: which dimensions have to line up
% in the simplest case, its only the first dimension
% but in general, one of the dimensions should be a prefix of the other?
% yes. it turns out.

%size(delta) = K, n, m
delta = centroids - fatXp;

%anonymous function
f =  @(x)  sum(x.^2) 
%gv = vectorize( f )

%reduce on n (eg. dx,dy -> dx^2 + dy^2 )
% the function works on the 1st dimension 
% questions: what would happen if f returned a vector..
% and ... how would I use a function with with 2 or more args?

%size(dist) = 1,3,300
dist = f( permute(delta, [2,1,3]));
size(dist);

%min(v) = (min_value, min_idx)
[v,idx] = min(dist);

% =============================================================

end