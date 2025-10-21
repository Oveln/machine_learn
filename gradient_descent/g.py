import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载数据
df = pd.read_csv('gradient_descent/diabetes1.csv')

# 2. 特征与标签
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# 3. 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 标准化特征（梯度下降需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 使用梯度下降回归器
model = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='optimal', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. 预测与评估
y_pred = model.predict(X_test_scaled)
print("均方误差:", mean_squared_error(y_test, y_pred))
print("R²得分:", r2_score(y_test, y_pred))