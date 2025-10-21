from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import classification_report
#引入鸢尾花数据集
iris_dataset= load_iris()
# DESCR(descr)键对应的值是数据集的简要说明
print(iris_dataset['DESCR'])
# 打印数据集
#print(iris_dataset)
#划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=\
    train_test_split(iris_dataset['data'],iris_dataset['target'],
                     test_size=0.25,random_state=0)
from sklearn import tree
cls=tree.DecisionTreeClassifier(criterion="gini",random_state=10)
cls.fit(X_train,y_train)
#评估模型
y_pred= cls.predict(X_test)
def evaluation(y_test,y_predict):
    accuracy= classification_report(y_test,y_predict,output_dict=True)['accuracy']
    s= classification_report(y_test, y_predict, output_dict=True)['weighted avg']
    precision=s['precision']
    recall = s['recall']
    f1_score=s['f1-score']
    return accuracy, precision, recall,f1_score
list_evalucation=evaluation(y_test,y_pred)
print("accuracy:{:.3f}".format(list_evalucation[0]))
print("weighted precision:{:.3f},".format(list_evalucation[1]))
print("weighted recall:{:.3f}".format(list_evalucation[2]))
print("F1_score:{:.3f}".format(list_evalucation[3]))


#预测，判断一个新的样本属于哪类
X_new=np.array([[5,2.9,1,0.2]])
prediction=cls.predict(X_new)
print("Prediction :{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))