clear;clc;close all;
load('data_test.mat')
load('data_train.mat')
load('label_train.mat')
data=[data_train label_train];
% rng(0)
x=data_test;
[trainedClassifier, validationAccuracy] = trainClassifier(data);


%% 训练样本及测试样本
[xxx,yyy]=size(data);
A=randperm(xxx);
aaa=floor(0.8*xxx);
numinput=33;
numoutput=1;
B=A(1:aaa);C=A(aaa+1:end);
tran_data1=data(B,:);
test_data1=data(C,:);
input_train=tran_data1(:,1:numinput);
output_train=tran_data1(:,numinput+1:numinput+numoutput);
input_test=test_data1(:,1:numinput);
output_test=test_data1(:,numinput+1:numinput+numoutput);


%% 预测
Ytrain=trainedClassifier.predictFcn(input_train);%训练样本的预测值
Ytest=trainedClassifier.predictFcn(input_test);%检验样本的预测值
Y=trainedClassifier.predictFcn(x);%全部样本的预测值

%% 计算准确度
A=sum(Ytrain==output_train)/size(Ytrain,1);
B=sum(Ytest==output_test)/size(Ytest,1);

%% 几个指标
TP=sum(Ytest==1&output_test==1);
FP=sum(Ytest==1&output_test==-1);
TN=sum(Ytest==-1&output_test==-1);
FN=sum(Ytest==-1&output_test==1);

TPR=TP/(TP+FN);
FPR=FP/(FP+TN);
accuracy=(TP+TN)/(TP+FP+TN+FN);
Precision=TP/(TP+FP);
Recall=TP/(TP+FN);
F1=A*TP/(2*TP+FP+FN);
PPV=TP/(TP+FP);
sensitivity1=TP/(TP+FN);
sensitivity2=TN/(TN+FP);