load detroit.mat;
M=data;
a = length(M);
x_0 = ones(a,1);
x_1 = data(1:a,1);
x_2 = data(1:a,9);
x_3 = data(1:a,6);
y = data(1:a,10);
matrix =[x_0 x_1 x_2 x_3];
[b,bint,r,rint,stats]= regress(y, matrix);
weight = b
a =stats
average = mean(r)
rcoplot(r,rint);
