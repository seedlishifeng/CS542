function [X_norm, mu, sigma] = Normalizefunction(X)

X_norm = X;
mu=mean(X);
sigma=std(X);
for i=1:size(mu,2)
    X_norm(:,i)=(X(:,i)-mu(i))./sigma(i);
end
end
