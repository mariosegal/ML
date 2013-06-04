error = 0;
error_train = zeros[1:12];
error_val = zeros[1:12];
	for k = 1:50
	    i = floor(rand(1000,1)*12)+1;
		theta = trainLinearReg(X_poly(1:i,:), y(1:i), lambda);
   		error_train(i) = error_train(i)+ linearRegCostFunction(X_poly(1:i,:), y(1:i), theta, 0);
   		error_val(i) = error_val(i)+linearRegCostFunction(X_poly_val, yval, theta, 0);
   	end
   	
   	