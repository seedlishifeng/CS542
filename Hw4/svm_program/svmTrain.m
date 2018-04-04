function svm = svmTrain(X,Y,C)
options = optimset;    
options.LargeScale = 'off';
options.Display = 'off';

n = length(Y);
K=kernel(X,X);
H = (Y*Y').*K;

f = -ones(n,1); 
A = [];
b = [];
Aeq = Y'; 
beq = 0;
lb = zeros(n,1); 
ub = C*ones(n,1);
a0 = zeros(n,1); 
[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);

%get boundary                     
sv_label = find(abs(a)>1e-8);  
b=0;
num=length(sv_label);
c=Y';
for n = 1:num
    tmp=0;
    for m = 1:num
        tmp=tmp+a(sv_label(m))*c(sv_label(m))*K(sv_label(n),sv_label(m));
    end
    b=b+c(sv_label(n))-tmp;
end
b=b/num;
svm.b=b;
svm.a = a(sv_label);
svm.Xsv = X(sv_label,:);
svm.Ysv = Y(sv_label);
svm.svnum = length(sv_label);
