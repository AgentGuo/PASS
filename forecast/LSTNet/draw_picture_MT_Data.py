import numpy as np
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from statsmodels.tsa.filters.hp_filter import hpfilter


mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif']=['Times New Roman']




def DRAW_PIC(i):
    #original
    original_value_0 = np.genfromtxt('./raw/hybrid_raw/hybrid_101.csv', delimiter=",")
    MTdata = original_value_0[:, i]

    csv_columns = pd.read_csv('./raw/hybrid_raw/hybrid_101.csv')
    columns_name = list(csv_columns.columns.values)

    Original = MTdata




    #。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。

    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


    #原始数据
    plt.figure(figsize=(100, 6))

    plt.plot(Original,  label = "Operation", color='blue')



    # # Set the ticks on x-axis
    indx = np.linspace(0, 17568, 62)
    plt.xticks(indx)
    # plt.ylim(0, 5)  #设置Y轴上下限

    plt.tick_params(axis='x',which='major',labelsize=25, direction='in')
    plt.tick_params(axis='y',which='major',labelsize=25, direction='in')
    plt.grid(color='grey', linestyle='--',linewidth=1, alpha=0.3, axis="y")
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)


    plt.legend(loc= 2, prop = {'size':20})

    plt.ylabel('qps',fontsize=30,fontweight='bold')
    plt.xlabel('One unit represents five minutes',fontsize=30,fontweight='bold')
    plt.title(columns_name[i], fontsize=30,fontweight='bold')  #标题

    plt.savefig('/Users/gejiake/Desktop/'+columns_name[i] +'.svg', bbox_inches='tight')
    # plt.show()



csv_columns = pd.read_csv('./raw/hybrid_raw/hybrid_101.csv')
csv_shape = np.array(csv_columns)
m, n = csv_shape.shape
for i in range(0, n):
    #第几个业务
    print('number:', i)
    print('apk:', csv_columns.columns.values[i])
    DRAW_PIC(i)