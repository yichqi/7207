clc;clear;close all;

%% 读取数据并划分数据集
[trainX,trainY,valX,valY,testX] = getData();

%% 鍒╃敤涓嶅悓鐨勪腑蹇冩暟鍒嗗埆璁粌RBF
% 中心数范围
n_center_Range = 10:1:200;

% sigma
sigma = 4.3;

for i = 1:length(n_center_Range)
    % 中心数
    n_center = n_center_Range(i);
    
    % 训练
    [W,C] = RBF_training(trainX,trainY,n_center,sigma);
    
    % 对训练集预测
    trainYp_out = RBF_predict(trainX,W,sigma,C);
    % 转换为类别
    trainY_out = trainYp_out>0.5;  
    % 计算度量值
    [TPR_train(i),FPR_train(i),accuracy_train(i),precision_train(i),recall_train(i),F1_train(i),PPV_train(i),...
        sensitivity_train(i),specificity_train(i)] = calMetrics(trainY,trainY_out);
    
    % 对验证集预测
    valYp_out = RBF_predict(valX,W,sigma,C);
    % 转换为类别
    valY_out = valYp_out>0.5;  
    % 计算度量值
    [TPR_val(i),FPR_val(i),accuracy_val(i),precision_val(i),recall_val(i),F1_val(i),PPV_val(i),...
        sensitivity_val(i),specificity_val(i)] = calMetrics(valY,valY_out);
end

%% 画图
plotFigures(TPR_train,FPR_train,accuracy_train,precision_train,recall_train,F1_train,PPV_train,sensitivity_train,specificity_train,n_center_Range,'训练集');

plotFigures(TPR_val,FPR_val,accuracy_val,precision_val,recall_val,F1_val,PPV_val,sensitivity_val,specificity_val,n_center_Range,'验证集');

%% 找出最优中心数
[accuracy_val_max,ind] = max(accuracy_val);
n_center_best = n_center_Range(ind);
fprintf('找出最优中心数为：%d\n',n_center_best)
fprintf('此时准确率为：%f%%\n',accuracy_val_max*100)

%% 利用最优中心数训练RBF
[W,C] = RBF_training(trainX,trainY,n_center_best,sigma);

%% 对训练集预测
trainYp_out = RBF_predict(trainX,W,sigma,C);
% 转为类别
trainY_out = trainYp_out>0.5;

% 画出PR曲线和ROC曲线
PR_ROC(trainYp_out,trainY,'训练集');

%% 对验证集预测
valYp_out = RBF_predict(valX,W,sigma,C);
% 转为类别
valY_out = valYp_out>0.5;

% 画出PR曲线和ROC曲线
PR_ROC(valYp_out,valY,'validation set');

%% 对测试集预测
testYp_out = RBF_predict(testX,W,sigma,C);
% 转为类别
testY_out = testYp_out>0.5;