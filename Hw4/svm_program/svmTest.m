function result = svmTest(svm, Xt, Yt)
K=kernel(svm.Xsv,Xt);
w = (svm.a'.*svm.Ysv')*K;

result.score = (w + svm.b)';
Y = sign(w+svm.b);
result.Y = Y';
if(size(Yt,1)>0)
    result.accuracy = size(find(Y'==Yt))/size(Yt);
else
    result.accurary = -1;
end