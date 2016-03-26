function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

h_theta = X*theta;
y_h_diff = h_theta-y;

J = sum( y_h_diff .* y_h_diff )/m/2;
J = J + (lambda/2/m * sum(theta(2:size(theta,1),1).^2) );

grad = grad(:);
grad(1,1) = sum( X(:, 1) .* y_h_diff )/m;
reg_term = theta * lambda/m;
grad(2:end,1) = (( X'(2:end, : ) * y_h_diff )/m) + reg_term(2:end,1);



end
