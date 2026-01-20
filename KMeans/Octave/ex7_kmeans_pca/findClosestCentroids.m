function idx = findClosestCentroids(X, centroids)
% Finding sample size
m = size(X,1);
% Set K
K = size(centroids, 1);

% Initializing cluster array
idx = zeros(m, 1);


for i = 1:m
  % Initializing belonging cluster as 1
  clusterIdx = 1;
        
  % Initializing the min-distance with the distance between x(i) and first-centroid
  minDistance = sum((X(i,:) - centroids(1,:)).^2);
  
  % Iterating over each centroid to find the min-distance
  for j=1:K
      distance = sum((X(i,:) - centroids(j,:)).^2);
      if(distance < minDistance)
          minDistance = distance;
          clusterIdx = j;
      endif
  endfor
  % Assigning the cluster value to a test-item       
  idx(i) = clusterIdx;
endfor

% =============================================================

end

