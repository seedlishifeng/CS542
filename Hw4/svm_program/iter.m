function result = iter(x,i,j,cell)
    svm=cell{i,j};
    res=svmTest(svm, x, []);
    if(i+1==j)
        if(res.Y==1)
            result=i-1;
        else
            result=j-1;
        end
    else
        if(res.Y==1)
            result=iter(x,i,j-1,cell);
        else
            result=iter(x,i+1,j,cell);
        end
    end
end