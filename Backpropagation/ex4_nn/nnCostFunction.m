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

# Generate one hot encoded vectors
y_encoded = zeros(n,m);  
for i = 1:m
    y_encoded(y(m,1),i)=1;
endfor

# X - m * n
a1 = [ones(m,1); X']; # Add bias - (n+1)*m
z1 = sigmoid(Theta1*a1);  # NN1 and add activate function (s1*(n+1))*((n+1)*m) - (s1*m)
a2 = [ones(hidden_layer_size,1); z1]  #(s1+1*m)
hx = sigmoid(Theta2*a2)  # NN2 and add activate function   #(s2*s1+1) (s1+1*m)  - s2*m=n*m

J = (-y_encoded  * log(hx) - (1 - y_encoded) * log(1 - hx));


for i = 1:m
    %The raw neuros counts the bias unit, which means it as a0 -> a25, not count a1
    a1 = [1 X(i,:)]'; % insert the bias unit %(401*1)
    z2 = Theta1 * a1;%(25*401)(401*1)
    a2 = [1 ; sigmoid(z2)];%(26*1)
    z3 = Theta2 * a2;%(10*1)
    a3 = sigmoid(z3);%(10*1)

    hx = a3;%(10*1)
    y_vec = zeros(num_labels,1);
    y_vec(y(i)) = 1; % this the compare y to be a binary vector

    single_cost = (-y_vec' * log(hx) - (1 - y_vec') * log(1 - hx));
    J = J + single_cost;



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
    
    delta3 = (a3 - y_vec);%(10*1)
    delta2 = (Theta2(:,2:end)' * delta3) .* sigmoidGradient(z2);%(25*1)
    Theta2_grad = Theta2_grad + delta3 * a2';%(10*26)
    Theta1_grad = Theta1_grad + delta2 * a1';%(25*401)

endfor

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Theta 0 does not involve in the regularlization From Cost Fn
theta2_reg = Theta2;
theta2_reg(:,1) = 0;
theta1_reg = Theta1;
theta1_reg(:,1) = 0;

% Regularlization of the Averaged cost function

J = J / m + lambda / (2 * m) * (sum( (theta1_reg.^2)(:) ) + sum( (theta2_reg.^2)(:) ) );

Theta2_grad = Theta2_grad / m + lambda / m * theta2_reg;
Theta1_grad = Theta1_grad / m + lambda / m * theta1_reg;


% =========================================================================


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


% an alternative solution
use_vectorizing = 0;
if use_vectorizing > 0
    % recode y to Y, a good way to learn
    I = eye(num_labels);
    Y = zeros(m, num_labels);
    for i=1:m
      Y(i, :)= I(y(i), :);
    endfor

    % feedforward
    a1 = [ones(m, 1) X];
    z2 = a1*Theta1';
    a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    h = a3;

    % calculte penalty
    p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));

    % calculate J
    J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);

    % calculate deltas
    delta3 = a3.-Y;
    delta2 = (delta3*Theta2(:,2:end)).*sigmoidGradient(z2);

    % accumulate gradients
    Delta1 = (delta2'*a1);
    Delta2 = (delta3'*a2);

    % calculate regularized gradient
    Theta1_reg = Theta1; Theta1_reg(:,1) = 0;
    Theta2_reg = Theta2; Theta2_reg(:,1) = 0;
    p1 = (lambda/m) * Theta1_reg;
    p2 = (lambda/m) * Theta2_reg;
    Theta1_grad = Delta1 / m + p1;
    Theta2_grad = Delta2 / m + p2;

endif

% -------------------------------------------------------------


