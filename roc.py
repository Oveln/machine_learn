import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve,roc_curve,auc #计算PR和RPC

confidence_scores = np.array([0.9, 0.46, 0.78, 0.37, 0.6, 0.4, 0.2, 0.16])
confidence_scores = sorted(confidence_scores, reverse=True)  # 按大到小排序
print(f'置信度为：{confidence_scores}')

data_labels = np.array([1, 1, 0, 1, 0, 0, 1, 0])
# TPR,FPR
fpr, tpr, threshold = roc_curve(data_labels, confidence_scores)
# print(fpr,fpr)
plt.figure()
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='Class 0 (AUC = %0.2f)' % auc)
plt.legend('lower left')
plt.show()


# 精确率、召回率、阈值
precision, recall, threshold = precision_recall_curve(data_labels, confidence_scores)
print('准确率：', precision)
print('召回率：', recall)
print('阈值', threshold)
plt.figure()
plt.title('PR')
plt.xlabel('recall')
plt.ylabel('precision')
plt.grid()  # 添加网格线
plt.plot(recall, precision)  # 绘制pr曲线
plt.show()


