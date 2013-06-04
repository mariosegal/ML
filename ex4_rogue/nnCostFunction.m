function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

   %1 compute the forward feed using my function;

   [A2 A3 Z2 Z3]=forward([ones(m,1) X],Theta1,Theta2);
   
   %to do cost I need to somehow recode the y as a matrix Y
   Y = zeros(m,num_labels);
   for i = 1:num_labels
      Y(y==i,i) = 1;
   end;
   
%two loop implementation;
 %  J=0;
 %  for i = 1:m;
 %     for k = 1:num_labels;
 %        J= J-Y(i,k).*log(A3(i,k)) - (1-Y(i,k)).*log(1-A3(i,k));
 %     end;
 %   end;
 % J=sum(J)*(1/m) ;

%one loop implemenation;
%   J=zeros(1,num_labels);
%   for i = 1:m;
%         J= J-Y(i,:).*log(A3(i,:)) - (1-Y(i,:)).*log(1-A3(i,:));
%   end;
%   J=sum(J)*(1/m) ;
   
%new approach - vectorized - I did not know about dot product for matrices;

J=0;
J= (Y.*log(A3) + (1-Y).*log(1-A3));
J=sum(sum(J))*(-1/m);


%add regularization;
%I need an auxiliary vector with zero for bias term for each matrix, so I can vectorize;
aux1 = [0 ones(1,input_layer_size)];
aux2 = [0 ones(1,hidden_layer_size)];

J = J +(lambda/(2*m))*(sum(aux1*(Theta1.^2)') + sum(aux2*(Theta2.^2)'));

%now lets do the backpropagation;
%D1 = 0;
%D2 = 0;
%for t=1:m
%   [a2 a3 z2 z3]=forward([1 X(t,:)],Theta1,Theta2);
%   delta3 = a3 - Y(t,:); %error for current trainign example;
%   %g = A2(t,:).*(1-A2(t,:));
%   delta2 =(delta3*Theta2(:,2:end)).*sigmoidGradient(z2);
%   D2 = D2 + delta3'*a2;
%   D1 = D1 + delta2'*[ones(m,1) X](t,:);
%endfor


%vectorized version;
D2 = zeros(10,26);
D1 = zeros(25,401);
%you have A3, A2, Z2, Z3 from cost part above.
delta3 = A3 - Y; %[5000 10] - [5000 10] = [5000 10]
delta2=(delta3*Theta2(:,2:end)).*sigmoidGradient(Z2); %[5000 10] *[10 25] .* [5000 25]=[5000 25]
D2 = delta3'*A2;
D1 = delta2'*[ones(m,1) X];

Theta1_grad = D1.*(1/m);
Theta2_grad = D2.*(1/m);

aux1 = [zeros(hidden_layer_size,1) ones(hidden_layer_size,input_layer_size)];
aux2 = [zeros(num_labels,1) ones(num_labels,hidden_layer_size)];
Theta1_grad = Theta1_grad +(lambda/m)*(aux1.*Theta1);
Theta2_grad = Theta2_grad +(lambda/m)*(aux2.*Theta2);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


