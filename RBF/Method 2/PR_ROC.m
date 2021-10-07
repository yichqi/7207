function PR_ROC(predict_p,real,name)

thrRange = 0:0.02:1;
for i = 1:length(thrRange)
    thr = thrRange(i);
    predict = predict_p>thr;
    
    TP = sum(predict==1 & real==1);
    FP = sum(predict==1 & real==0);
    TN = sum(predict==0 & real==0);
    FN = sum(predict==0 & real==1);

    % 真阳性率
    TPR(i) = TP/(TP+FN);

    % 假阳性率
    FPR(i) = FP/(FP+TN);

    % 精确率
    precision(i) = TP/(TP+FP);

    % 召回率
    recall(i) = TP/(TP+FN);
end

% P-R曲线
figure
plot(precision,recall,'linewidth',1.5)
xlabel('precision','fontsize',12)
ylabel('recall','fontsize',12)
grid on
title([name ' P-R curve'],'fontsize',12)

% ROC曲线
figure
plot(FPR,TPR,'linewidth',1.5)
xlabel('FPR','fontsize',12)
ylabel('TPR','fontsize',12)
grid on
title([name ' ROC curve'],'fontsize',12)