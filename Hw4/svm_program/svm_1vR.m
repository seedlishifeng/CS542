data=load('MNIST_data.mat');
%main function
C = inf;
%train samples
x1 = data.train_samples;
y1 = data.train_samples_labels;
%test samples
x2 = data.test_samples;
y2 = data.test_samples_labels;



%1 vs rest testing
score=zeros(1000,10);
for m=0:9
    tmp=int2str(m);
    mysvm=strcat('svm1vr_',tmp,'.mat');
    svm=load(mysvm);
    temp1 = find(y2~=m);
    temp_y2=-ones(1000,1);
    temp_y2(temp1)=1;
    result = svmTest(svm.svm, x2, temp_y2);
    score(:,m+1)=result.score;
    [M,induce]=min(score,[],2);
    accuracy=size(find((induce-1)==y2))/size(y2);
end
fprintf('accuracy of 1 vs rest is %f\n',accuracy);

