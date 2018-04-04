data=load('MNIST_data.mat');
%main function
C = inf;
%train samples
x1 = data.train_samples;
y1 = data.train_samples_labels;
%test samples
x2 = data.test_samples;
y2 = data.test_samples_labels;      

vote=zeros(1000,10);
for r =0:9
    for s=0:9
        if(s>r)
            temp1 = find(y1==r);
            temp2 = find(y1==s);
            k=size(temp1,1)+size(temp2,1);
            temp_y1=zeros(k,1);
            temp_x1=zeros(k,784);
            n=1;
            for i=1:4000
                if(y1(i,1)==r)
                    temp_y1(n,1)=1;
                    temp_x1(n,:)=x1(i,:);
                    n=n+1;
                elseif(y1(i,1)==s)
                    temp_y1(n,1)=-1;
                    temp_x1(n,:)=x1(i,:);
                    n=n+1;
                end
            end
            svm = svmTrain(temp_x1,temp_y1,C);
            result = svmTest(svm, x2, []);
            index1=find(result.Y==1);
            index2=find(result.Y==-1);
            vote(index1,r+1)=vote(index1,r+1)+1;
            vote(index2,s+1)=vote(index2,s+1)+1;
            
        end
    end
    
end
[M,induce]=max(vote,[],2);
accuracy=size(find((induce-1)==y2))/size(y2);
fprintf('accuracy of 1 vs 1 is %f\n',accuracy);
