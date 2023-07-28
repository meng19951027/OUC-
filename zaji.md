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
