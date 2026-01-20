function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
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

% Initializing centroids array
centroids = zeros(K, n);

for clusterNum = 1:K
   % Getting elements belonging to perticular cluster
   clusterEles = X(idx == clusterNum,:);
        
   % Getting no. of elements in the cluster
   nCluster = size(clusterEles,1);
   for i = 1:n
   % Calculating mean of cluster elements to find clusterCentoid
   centroids(clusterNum,i) =  (1/nCluster)*sum(clusterEles(:,i));
   endfor
endfor

% =============================================================
end

