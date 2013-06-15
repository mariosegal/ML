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
points=size(X,2);
%cycle thorugh the examples, do all else vectorized;
%for i = 1:size(X)(1)
%	[m idx(i)] = min(sum((bsxfun(@minus,X(i,:),centroids)).^2,2));
%endfor

[m n]=size(X);

%vectorized code;

aux=-(2*X*centroids')+(ones(m,1)*((centroids.^2)*ones(n,1))'); 
[min1 idx] = min(aux,[],2);

%this is the rsult of expanding (a-b)^2, and realizing that for each row x2 will be a;
%constant so why calculate it, plus the aux matrices to make it all work
%X=[m,n] centroids'=[n,k] so first term is [m,k];
%second term is then [m,1]*([k,n]*[n,1]= [[k,1])'= [m,k], cosistent with first;





% =============================================================

end