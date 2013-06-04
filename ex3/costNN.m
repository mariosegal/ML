function [J,grad] = costNN(X,theta1,theta2,y,lambda)

m=size(X)(1);
[a2 pred]=forward(X,theta1,theta2);

J= y'*log(pred) + (1-y')*log(1-pred);
J=J*(-1/m) ;
reg_term = sum(sum(theta1(:,2:size(theta1)(2)).^2));
reg_term = reg_term + sum(sum(theta2(:,2:size(theta2)(2)).^2));

J=J+reg_term*(lambda/(2*m));

end;
