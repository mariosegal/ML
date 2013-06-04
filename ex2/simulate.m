function simulate(lambda)
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
initial_theta = zeros(size(X, 2), 1);
X = mapFeature(X(:,1), X(:,2));

initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

end