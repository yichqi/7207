clear;clc;close all;
load('data_test.mat')
load('data_train.mat')
load('label_train.mat')
data=[data_train label_train];
%rng(0)
%% 数据标准化
numinput=33;
numoutput=1;
input=data(:,1:numinput);
output=data(:,numinput+1:numinput+numoutput);
output=output';
[input,a]=mapminmax(input',0,1);
% [output,b]=mapminmax(output',0,1);
data1=[input;output]';
x=mapminmax('apply',data_test',a);

%% 训练样本及测试样本
[xxx,yyy]=size(data1);
A=randperm(xxx);
aaa=floor(0.8*xxx);

B=A(1:aaa);C=A(aaa+1:end);
tran_data1=data1(B,:);
test_data1=data1(C,:);
input_train=tran_data1(:,1:numinput)';
output_train=tran_data1(:,numinput+1:numinput+numoutput)';
input_test=test_data1(:,1:numinput)';
output_test=test_data1(:,numinput+1:numinput+numoutput)';

%% 神经网络
net_trained = newrb(input_train,output_train,0.12,1.3,200,1)

%% 预测
train_sim=sim(net_trained,input_train);%训练样本的预测值
test_sim=sim(net_trained,input_test);%检验样本的预测值
yuce=sim(net_trained,x);%全部样本的预测值

%% 分类
Ytrain=ones(size(train_sim));
Ytrain(train_sim<0)=-1;
A=sum(Ytrain==output_train)/size(Ytrain,2);

Ytest=ones(size(test_sim));
Ytest(test_sim<0)=-1;
B=sum(Ytest==output_test)/size(Ytest,2);

Y=ones(size(yuce));
Y(yuce<0)=-1;

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



