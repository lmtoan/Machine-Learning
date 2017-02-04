function [J, grad] = nnCostFunction(nn_params, ...
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
                 hidden_layer_size, (input_layer_size + 1)); %input_layer_size + 1 to account for the bias unit, after undo the unrolling

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

%Set up X with added bias units, X is 5000x401
X = [ones(m, 1), X];

%Feedforward. Theta1: 25x401, Theta2: 10x26
z_2 = X * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2, 1), 1), a_2]; %5000x26 matrix

z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3); %5000x10 matrix. This is final out. All rows should be 1,0. Don't need to add another column of bias unit

%a3(2145, :) %test print
h_ff = a_3'; %transpose to 10x5000. Each column is a predicted output.

for i = 1:m %loop through all m/5000 examples
    actual_vector = convert(y(i), num_labels); %10x1
    predicted_vector = h_ff(:, i); %10x1
    k_vector = (-actual_vector) .* log(predicted_vector) - (1 - actual_vector) .* log(1 - predicted_vector); %10*1
    class_sum = sum(k_vector); %sum of all k classes
    J = J + class_sum;
end;

J = (1/m) * J;
% ------------------------Regularization

%Change the matrix theta_1, theta_2: make the first column all zero
Theta1_reg = Theta1;
Theta1_reg(:, 1) = 0; %or f_theta1 = Theta1(:,2:end) i.e so that the whole first column is removed entirely
Theta2_reg = Theta2;
Theta2_reg(:, 1) = 0;

%Add the regularization/penalty part
Theta1_sqr = Theta1_reg .^ 2; %square all elements & add up
sum_t1 = sum(Theta1_sqr(:));
Theta2_sqr = Theta2_reg .^ 2; %square all elements & add up
sum_t2 = sum(Theta2_sqr(:));
J = J + lambda / (2*m) * (sum_t1 + sum_t2);

% ------------------------Backpropagation: Loop through all m examples. Theta1: 25x401, Theta2: 10x26
for t = 1:m,
    a_1 = X(t, :);
    a_1 = a_1'; %vector 401x1
    z_2 = Theta1 * a_1;
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2]; %26x1
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3); %Output 10x1
    
    output = convert(y(t), num_labels);
    
    delta_3 = a_3 - output; %10x1
    g = sigmoidGradient(z_2);
    delta_2 = g .* (Theta2_reg' * delta_3); %Use Theta2_reg so that the column associated with bias unit is eliminated, so delta2 becomes 25x1
    
    Theta2_grad = Theta2_grad + delta_3 * (a_2)'; %10x26. Accumulate all the matrices across m training examples
    Theta1_grad = Theta1_grad + delta_2 * (a_1)'; %25x401. Accumulate all the matrices across m training examples
end;

Theta2_grad = (1/m) .* Theta2_grad;
Theta1_grad = (1/m) .* Theta1_grad;

% ------------------------Add regularization to the gradients D-layer1,2

Theta1_grad_reg = Theta1_grad; %Initialize for all original columns
Theta2_grad_reg = Theta2_grad;

Theta1_grad_reg = Theta1_grad_reg + (lambda/m) .* Theta1_reg; %Theta1_reg has the first column all 0
Theta2_grad_reg = Theta2_grad_reg + (lambda/m) .* Theta2_reg; %Theta2_reg has the first column all 0

Theta1_grad = Theta1_grad_reg;
Theta2_grad = Theta2_grad_reg;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
