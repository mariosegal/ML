function [d2 d3] = backward(X, theta1, theta2,y)

[A2 A3 z2 z3] = forward(X,theta1,theta2);

d3 = (y-A3).*fprime(A3);
%d3 validated
d2 = (theta2'*d3).*fprime(A2);

grad3 = d3*A2';
grad2 = d2*X';



%LETS DO LOOP
%for sake of argument, D is the partial derivative
% capital delta, the triangle thing will be DELTA
% delta would be just delta

L = 2;
m = 1; %for now, eventually it will be wahtever it is;
K = 1; %for now;
S = [2 1]; %this is a vector of the numbe rof units in each layer ;

DELTA2 = zeros(size(theta2));
DELTA1 = zeros(size(theta1));

for i = 1:m
   A1 = X(i,:);
   [A2 A3 z2 z3] = forward(A1,theta1,theta2);
   delta3=A3-y;
   delta2=(theta2'*delta3).*fprime(A2)';
   
   DELTA2=DELTA2 + A2*delta3';
   aux2 = [zeros(size(DELTA2)(1),1) ones(size(DELTA2)(1),size(DELTA2)(2)-1)]; 
   %aux2  is used so I do not have to deal sepate;y with the 0 element of theta;
   D2=(1/m)*(DELTA2 + lambda*aux2.*theta2);
   
   DELTA1=DELTA1 + A1*delta2';
   aux1 = [zeros(size(DELTA1)(1),1) ones(size(DELTA1)(1),size(DELTA1)(2)-1)]; 
   D1=(1/m)*(DELTA1 + lambda*aux1.*theta1);
end
%by god it seems I did it, I need to test this though

