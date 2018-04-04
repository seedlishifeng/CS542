load detroit.mat;
M=data;
re=zeros(7,1);

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

feature_number=size(X,2);
temp=zeros(feature_number,1);

for iter = 1:num_iters
    for i=1:feature_number
        temp(i)=theta(i)-(alpha/m)*sum((X*theta-y).*X(:,i));
    end
    for j=1:feature_number
        theta(j)=temp(j);
    end
    J_history(iter) = computeCostMulti(X, y, theta);
end
end

function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
mu=mean(X);
sigma=std(X);
for i=1:size(mu,2)
    X_norm(:,i)=(X(:,i)-mu(i))./sigma(i);
end
end

function J = computeCostMulti(X, y, theta)
m = length(y); % number of training examples

prediction=X*theta;
J=(prediction-y)'*(prediction-y)/(2*m);

end

for i = 2:7
 a = length(M);
 x_0 = ones(a,1);
 x_1 = data(1:a,1);
 x_2 = data(1:a,9);
 x_3 = data(1:a,i);
 y = data(1:a,10);
 matrix =[x_1 x_2 x_3];
[matrix,mu,sigma] = J(matrix);
 X = [ones(a, 1) matrix];
 alpha = 0.01;
num_iters = 5000;
theta = zeros(4, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
fprintf('Cost computed from gradient descent: \n');
fprintf(' %f \n', J_history(num_iters,1));
fprintf('\n');

re(i,1)=J_history(num_iters,1);
end
    
    