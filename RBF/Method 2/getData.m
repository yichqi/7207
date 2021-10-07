function [trainX,trainY,valX,valY,testX] = getData()

rand('seed',2)

load('data_train.mat')
%load('.\ѵ���Ͳ�������\data_train.mat')
dataX = data_train;

load('label_train.mat')
%load('.\ѵ���Ͳ�������\label_train.mat')
dataY = label_train;
dataY = dataY>0;

% �����������˳��
dataNum = length(dataY);
randInd = randperm(dataNum);
dataX = dataX(randInd,:);
dataY = dataY(randInd,:);

% ����ѵ��������֤��
trainX = dataX(1:round(0.8*dataNum),:);
trainY = dataY(1:round(0.8*dataNum),:);
valX = dataX(1+round(0.8*dataNum):end,:);
valY = dataY(1+round(0.8*dataNum):end,:);

load('data_test.mat')
%load('.\ѵ���Ͳ�������\data_test.mat')
testX = data_test;