# 导入库
import numpy as np  # numpy库
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
# 数据准备
# raw_data = np.loadtxt('regression.txt')  # 读取数据文件
# X = raw_data[:, :-1]  # 分割自变量
# y = raw_data[:, -1]  # 分割因变量

dfx_order=dfx[sheetname[3]]
X=dfx_order.iloc[:,:3]
Y=dfx_order.iloc[:,3]
print(X.head())
print(Y.head())
#数据划分
xtrain=X.iloc[:int(0.7*len(dfx_order)),:]
xtest=X.iloc[int(0.7*len(dfx_order)):,:]
ytrain=Y[:int(0.7*len(dfx_order))]
ytest=Y[int(0.7*len(dfx_order)):]



# 训练回归模型
n_folds = 6  # 设置交叉检验的次数
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr = LinearRegression()  # 建立普通线性回归模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR']  # 不同模型的名称列表
model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr]  # 不同回归模型对象的集合
cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表

pre_y_list2 = []  # 各个回归模型预测的y值列表

for model in model_dic:  # 读出每个回归模型对象
    scores = cross_val_score(model, xtrain, ytrain, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    pre_y_list.append(model.fit(xtrain, ytrain).predict(xtrain))  # 将回归训练中得到的预测y存入列表
    pre_y_list2.append(model.fit(xtrain, ytrain).predict(xtest))
# 模型效果指标评估
n_samples, n_features = xtrain.shape  # 总样本量,总特征数
#explained_variance_score:解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
    #的方差变化，值越小则说明效果越差。
#mean_absolute_error:平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
    #，其其值越小说明拟合效果越好。
#mean_squared_error:均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的
    #平方和的均值，其值越小说明拟合效果越好。
#r2_score:判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
    #变量的方差变化，值越小则说明效果越差。
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(ytrain, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print ('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print (70 * '-')  # 打印分隔线
print ('cross validation result:')  # 打印输出标题
print (df1)  # 打印输出交叉检验的数据框
print (70 * '-')  # 打印分隔线
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框
print (70 * '-')  # 打印分隔线
print ('short name \t full name')  # 打印输出缩写和全名标题
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-')  # 打印分隔线
# 模型效果可视化
plt.figure(figsize=(10,5))  # 创建画布
plt.plot(np.arange(xtrain.shape[0]), ytrain, color='k', label='true y',lw=7)  # 画出原始值的曲线
plt.plot(np.arange(xtrain.shape[0],xtrain.shape[0]+xtest.shape[0]), ytest, color='r', label='true y',lw=7)  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(xtrain.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
    
    plt.plot(np.arange(xtrain.shape[0],xtrain.shape[0]+xtest.shape[0]), pre_y_list2[i])
plt.title('regression result comparison')  # 标题
plt.legend(loc='lower right')  # 图例位置
plt.ylim(-10,20)
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
