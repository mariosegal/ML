function iterate_kmeans(max)
max_iters = 10;
K=3;
load('ex7data2.mat');
best = 999999999999999;

for i = 1:max
    
	initial_centroids = kMeansInitCentroids(X, K);
	[centroids, idx] = runkMeans(X, initial_centroids, max_iters, false);
		
	m=size(X,1);
	cost = 0;
	for j = 1:m
		cost = cost+(1/m)*sum((X(j,:)-centroids(idx(j),:)).^2);
	endfor
	if (cost <= best) 
		best=cost;
		best_centroids = centroids;
	    best_idx = idx;
	endif
	fprintf('Overall iteration %d/%d... Best Cost=%g\n', i, max,best);
	if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
endfor

plotDataPoints(X, best_idx, K)

end;