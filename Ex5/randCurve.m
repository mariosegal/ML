



error_train = zeros(12,1);
error_val = zeros(12,1);
%lambda = 1;
for i = 1:12
	for k = 1:50
	      indx = randperm(12,i);
		theta = trainLinearReg(X_poly(indx,:), y(indx), lambda);
   		error_train(i) = error_train(i)+ linearRegCostFunction(X_poly(indx,:), y(indx), theta, 0);
   		error_val(i) = error_val(i)+linearRegCostFunction(X_poly_val, yval, theta, 0);
   	end
end
error_train = error_train ./ 50;
error_val = error_val ./ 50;

x= 1:12;

plot(x,error_train,'b;Training Set;',x,error_val,'r;Validation Set;');
hold on;
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
xlabel('Number of Training examples')
ylabel('Error')
hold off;



   	
   	