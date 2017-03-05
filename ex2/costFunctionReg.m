function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m,
	J += (-1) * y(i, 1) * log(sigmoid(X(i, :) * theta)) + (-1) * (1 - y(i, 1)) * log(1 - sigmoid(X(i, :) * theta));
end;
J = J / m;

sumThetaSquare = 0;
for ii = 2:size(theta, 1),
	sumThetaSquare += theta(ii) ^ 2;
end;

J += lambda * sumThetaSquare / (2 * m);

for j = 1:size(grad, 1),
	for k = 1:m,
		grad(j) += (sigmoid(X(k, :) * theta) - y(k, 1)) * X(k, j);
	end;

	if(j == 1)
		grad(j) = grad(j) / m;
	else
		grad(j) = grad(j) / m + lambda * theta(j) / m;
	end;
end;


% =============================================================

end
