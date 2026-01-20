function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
size(Theta2)
X = [ones(m, 1) X];

%Layer 1 o/p
%Layer 2 i/p
z_2 = Theta1*X';


%Layer 2 o/p
a_2 = sigmoid(z_2);

%Adding a0_2
a_2 = [ones(1,size(a_2,2)); a_2];


%Layer 3 i/p
z_3 = Theta2*a_2;

%Layer 3 o/p
h_theta = sigmoid(z_3);

%Prediction
[max_p,idx] = max(h_theta);

p = idx';







% =========================================================================


end
