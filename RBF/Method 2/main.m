clc;clear;close all;

%% ��ȡ���ݲ��������ݼ�
[trainX,trainY,valX,valY,testX] = getData();

%% 利用不同的中心数分别训练RBF
% ��������Χ
n_center_Range = 10:1:200;

% sigma
sigma = 4.3;

for i = 1:length(n_center_Range)
    % ������
    n_center = n_center_Range(i);
    
    % ѵ��
    [W,C] = RBF_training(trainX,trainY,n_center,sigma);
    
    % ��ѵ����Ԥ��
    trainYp_out = RBF_predict(trainX,W,sigma,C);
    % ת��Ϊ���
    trainY_out = trainYp_out>0.5;  
    % �������ֵ
    [TPR_train(i),FPR_train(i),accuracy_train(i),precision_train(i),recall_train(i),F1_train(i),PPV_train(i),...
        sensitivity_train(i),specificity_train(i)] = calMetrics(trainY,trainY_out);
    
    % ����֤��Ԥ��
    valYp_out = RBF_predict(valX,W,sigma,C);
    % ת��Ϊ���
    valY_out = valYp_out>0.5;  
    % �������ֵ
    [TPR_val(i),FPR_val(i),accuracy_val(i),precision_val(i),recall_val(i),F1_val(i),PPV_val(i),...
        sensitivity_val(i),specificity_val(i)] = calMetrics(valY,valY_out);
end

%% ��ͼ
plotFigures(TPR_train,FPR_train,accuracy_train,precision_train,recall_train,F1_train,PPV_train,sensitivity_train,specificity_train,n_center_Range,'ѵ����');

plotFigures(TPR_val,FPR_val,accuracy_val,precision_val,recall_val,F1_val,PPV_val,sensitivity_val,specificity_val,n_center_Range,'��֤��');

%% �ҳ�����������
[accuracy_val_max,ind] = max(accuracy_val);
n_center_best = n_center_Range(ind);
fprintf('�ҳ�����������Ϊ��%d\n',n_center_best)
fprintf('��ʱ׼ȷ��Ϊ��%f%%\n',accuracy_val_max*100)

%% ��������������ѵ��RBF
[W,C] = RBF_training(trainX,trainY,n_center_best,sigma);

%% ��ѵ����Ԥ��
trainYp_out = RBF_predict(trainX,W,sigma,C);
% תΪ���
trainY_out = trainYp_out>0.5;

% ����PR���ߺ�ROC����
PR_ROC(trainYp_out,trainY,'ѵ����');

%% ����֤��Ԥ��
valYp_out = RBF_predict(valX,W,sigma,C);
% תΪ���
valY_out = valYp_out>0.5;

% ����PR���ߺ�ROC����
PR_ROC(valYp_out,valY,'validation set');

%% �Բ��Լ�Ԥ��
testYp_out = RBF_predict(testX,W,sigma,C);
% תΪ���
testY_out = testYp_out>0.5;