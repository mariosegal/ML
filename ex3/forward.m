function [A2 A3 z2 z3] = forward(X, theta1, theta2)
%for a model 2-2-1 (not counting the bias terms, with them is 3,3,1)
m=size(X)(1);
n=size(X)(2);

z2 = X*theta1';
A2=sigmoid(z2);

A2=[ones(size(A2)(1),1) A2];

z3=A2*theta2';
A3=sigmoid(z3);

%p=zeros(size(A3));
%p(A3>=0.5)=1;
%sp=A3;
end;


