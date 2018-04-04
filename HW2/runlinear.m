load('detroit.mat','data');
re=zeros(7,1);
for i =2:7
X = [data(:, 1),data(:, 8),data(:, i)];%change the third column to
y = data(:, 10);
m = length(y);

% Scale features and set them to zero mean
[X,mu,sigma] = Normalizefunction(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Choose some alpha value
alpha = 0.01;
num_iters = 5000;

% Init Theta and Run Gradient Descent 
theta = zeros(4, 1);
[theta, J_history] = gradientDescentfunction(X, y, theta, alpha, num_iters);

% Plot the convergence graph



% Display gradient descent's result
fprintf('Cost computed from gradient descent: \n');
fprintf(' %f \n', J_history(num_iters,1));
fprintf('\n');

re(i,1)=J_history(num_iters,1);
end

figure
x=2:7;
y=re(2:7,1);
bar(x,y)
set(gca,'xticklabel',{'UEMP','MAN','LIC','GR','NMAN','GOV','HE'})
xlabel('variable in dataset');
ylabel('Minmum cost for for each variable in dataset');
min=2;
