data=load('MNIST_data.mat');
%main function
C = inf;
%train samples
x1 = data.train_samples;
y1 = data.train_samples_labels;
%test samples
x2 = data.test_samples;
y2 = data.test_samples_labels;

svm_cell={};


%DAG Path
svm_set=load('svm_1v1_cell.mat');

final=-ones(1000,1);
for i =1:1000
    final(i,1)=iter(x2(i,:),1,9,svm_set.svm_cell);
end
accuracy=size(find(final==y2))/size(y2);
fprintf('accuracy of DAGSVM is %f\n',accuracy);
