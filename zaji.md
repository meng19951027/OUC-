```
def Curve_Fitting(x,y,deg):
    parameter = np.polyfit(x, y, deg)    #拟合deg次多项式
    p = np.poly1d(parameter)             #拟合deg次多项式
    aa=''                               #方程拼接  ——————————————————
    for i in range(deg+1): 
        bb=round(parameter[i],2)
        if bb>0:
            if i==0:
                bb=str(bb)
            else:
                bb='+'+str(bb)
        else:
            bb=str(bb)
        if deg==i:
            aa=aa+bb
        else:
            aa=aa+bb+'x^'+str(deg-i)    #方程拼接  ——————————————————
    plt.scatter(x, y)     #原始数据散点图
    plt.plot(x, p(x), color='g')  # 画拟合曲线
   # plt.text(-1,0,aa,fontdict={'size':'10','color':'b'})
    plt.legend([aa,np.corrcoef(y, p(x))[0,1]**2])   #拼接好的方程和R方放到图例
    plt.show()
#    print('曲线方程为：',aa)
#    print('     r^2为：',round(np.corrcoef(y, p(x))[0,1]**2,2))
```




# 绘制流向图一维
```
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# 模拟时间点的水流流向数据（假设有5个时间点的数据）
time_points = [1, 2, 3, 4, 5]
flow_directions = [45, 180, 90, 135, 270]  # 流向用0~360之间的度数表示，例如：0表示向右，90表示向上，180表示向左，270表示向下

# 转换度数为弧度
flow_angles_rad = np.deg2rad(flow_directions)

# 设置图形大小
plt.figure(figsize=(10, 2))

# 绘制箭头
for i in range(len(time_points)):
    arrow_length = 0.8
    dx = arrow_length * np.cos(flow_angles_rad[i])
    dy = arrow_length * np.sin(flow_angles_rad[i])
    arrow = FancyArrowPatch((i, 0), (i + dx, dy), arrowstyle='->', mutation_scale=20, linewidth=2, color='blue')
    plt.gca().add_patch(arrow)

# 设置坐标轴
plt.xlim(0, len(time_points))
plt.ylim(-1.5, 1.5)

# 设置坐标轴标签
plt.xlabel('时间点')
plt.ylabel('流向')

# 设置标题
plt.title('随时间变化的水流流向图')

# 显示图形
plt.show()
```
