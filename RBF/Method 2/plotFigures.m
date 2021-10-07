function plotFigures(TPR,FPR,accuracy,precision,recall,F1,PPV,sensitivity,specificity,n_center_Range,name)

% TPR
figure
plot(n_center_Range,TPR,'linewidth',1.5)
xlabel('中心数','fontsize',12)
ylabel('TPR','fontsize',12)
grid on
title([name ' TPR'],'fontsize',12)

% FPR
figure
plot(n_center_Range,FPR,'linewidth',1.5)
xlabel('中心数','fontsize',12)
ylabel('FPR','fontsize',12)
grid on
title([name ' FPR'],'fontsize',12)

% accuracy
figure
plot(n_center_Range,accuracy,'linewidth',1.5)
xlabel('中心数','fontsize',12)
ylabel('accuracy','fontsize',12)
grid on
title([name ' accuracy'],'fontsize',12)

% precision
figure
plot(n_center_Range,precision,'linewidth',1.5)
xlabel('中心数','fontsize',12)
ylabel('precision','fontsize',12)
grid on
title([name ' precision'],'fontsize',12)

% recall
figure
plot(n_center_Range,recall,'linewidth',1.5)
xlabel('中心数','fontsize',12)
ylabel('recall','fontsize',12)
grid on
title([name ' recall'],'fontsize',12)

% F1
figure
plot(n_center_Range,F1,'linewidth',1.5)
xlabel('中心数','fontsize',12)
ylabel('F1','fontsize',12)
grid on
title([name ' F1'],'fontsize',12)

% PPV
figure
plot(n_center_Range,PPV,'linewidth',1.5)
xlabel('中心数','fontsize',12)
ylabel('PPV','fontsize',12)
grid on
title([name ' PPV'],'fontsize',12)

% sensitivity
figure
plot(n_center_Range,sensitivity,'linewidth',1.5)
xlabel('中心数','fontsize',12)
ylabel('sensitivity','fontsize',12)
grid on
title([name ' sensitivity'],'fontsize',12)

% specificity
figure
plot(n_center_Range,specificity,'linewidth',1.5)
xlabel('中心数','fontsize',12)
ylabel('specificity','fontsize',12)
grid on
title([name ' specificity'],'fontsize',12)
