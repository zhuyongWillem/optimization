# Description：
# Author：朱勇
# Email: yong_zzhu@163.com
# Time：2021/3/8 19:57

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

data_train = pd.read_csv('task1_train_data.csv')
data_test = pd.read_csv('task1_test_data.csv')
x_train = data_train.loc[:,'x_d']
y_train = data_train.loc[:,'y_d']
x_test = data_test.loc[:,'x_d']
y_test = data_test.loc[:,'y_d']
fig1 = plt.figure()
plt.scatter(x_train,y_train,label='training data')
plt.scatter(x_test,y_test,label='test data')
plt.title('training-test data')
plt.xlabel('x_d')
plt.ylabel('y_d')
plt.legend()
plt.show()
model1 = LinearRegression()
x_train = np.array(x_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)
model1.fit(x_train,y_train)
y_train_predict = model1.predict(x_train)
y_test_predict = model1.predict(x_test)
r2_train = r2_score(y_train,y_train_predict)
r2_test = r2_score(y_test,y_test_predict)
print(r2_test,r2_train)
#结果可视化,生成新的数据点
x_range = np.linspace(40,400,300).reshape(-1,1)
y_range_predict = model1.predict(x_range)
fig2 = plt.figure()
plt.scatter(x_train,y_train,label='training data')
plt.scatter(x_test,y_test,label='test data')
plt.plot(x_range,y_range_predict,'r',label='predicted curve')
plt.title('training-test data')
plt.xlabel('x_d')
plt.ylabel('y_d')
plt.legend()
plt.show()
#生成二阶属性数据
poly2 = PolynomialFeatures(degree=2)
x_2_train = poly2.fit_transform(x_train)
x_2_test = poly2.transform(x_test)
#建立新的高阶回归模型
model2 = LinearRegression()
model2.fit(x_2_train,y_train)
y_2_train_predict = model2.predict(x_2_train)
y_2_test_predict = model2.predict(x_2_test)
r2_2_train = r2_score(y_train,y_2_train_predict)
r2_2_test = r2_score(y_test,y_2_test_predict)
print(r2_2_train,r2_2_test)
#二阶模型可视化
x_2_range = poly2.transform(x_range)
y_2_range_predict = model2.predict(x_2_range)
fig3 = plt.figure()
plt.scatter(x_train,y_train,label='training data')
plt.scatter(x_test,y_test,label='test data')
plt.plot(x_range,y_2_range_predict,'r',label='predicted curve')
plt.title('training-test data(poly2)')
plt.xlabel('x_d')
plt.ylabel('y_d')
plt.legend()
plt.show()
