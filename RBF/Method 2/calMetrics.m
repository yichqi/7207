function [TPR,FPR,accuracy,precision,recall,F1,PPV,sensitivity,specificity] = calMetrics(real,predict)

TP = sum(predict==1 & real==1);
FP = sum(predict==1 & real==0);
TN = sum(predict==0 & real==0);
FN = sum(predict==0 & real==1);

% 真阳性率
TPR = TP/(TP+FN);

% 假阳性率
FPR = FP/(FP+TN);

% 准确率
accuracy = (TP+TN)/(TP+TN+FP+FN);

% 精确率
precision = TP/(TP+FP);

% 召回率
recall = TP/(TP+FN);

% F1
F1 = 2*TP/(2*TP+FP+FN);

% positive predictive value
PPV = TP/(TP+FP);

% 敏感性
sensitivity = TP/(TP+FN);

% 特异性
specificity = TN/(TN+FP);
