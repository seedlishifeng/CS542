load('detroit.mat','data');
re=zeros(7,1);
for i =2:8
X = [data(:, 1),data(:, 8),data(:, i)];
y = data(:, 10);
m = length(y);


X_norm = X;
mu=mean(X);
sigma=std(X);
for j=1:size(mu,2)
    X_norm(:,j)=(X(:,j)-mu(j))./sigma(j);
end
X = X_norm;


X = [ones(m, 1) X];


alpha = 0.05;
num_iters = 2000;


theta = zeros(4, 1);
m = length(y);
J_history = zeros(num_iters, 1);

feature_number=size(X,2);
tem=zeros(feature_number,1);

for iter = 1:num_iters
    for j=1:feature_number
        tem(j)=theta(j)-(alpha/m)*sum((X*theta-y).*X(:,j));
    end
    for j=1:feature_number
        theta(j)=tem(j);
    end
    m = length(y); 

    prediction=X*theta;
    J=(prediction-y)'*(prediction-y)/(2*m);
    J_history(iter) = J;
end





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
for index=3:size(re,1)
    if re(min,1)>re(index,1)
        min=index;
    end
end
fprintf('The best input variable is the No.%d variable in dataset and the minimum cost is %f\n',min,re(min,1));