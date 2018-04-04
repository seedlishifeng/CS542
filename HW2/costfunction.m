function J = costfunction(X, y, theta)
m = length(y); % number of training examples

prediction=X*theta;
J=(prediction-y)'*(prediction-y)/(2*m);

end
