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
    original_value_0 = np.genfromtxt('./raw/hybrid_110.csv', delimiter=",")

    # MTdata = original_value_0[:, i]
    MTdata = original_value_0[-1700:, i]




    #predict
    predict_value = np.genfromtxt('./save/hybrid_110.csv', delimiter=",")
    # predict_value_360_288 = np.genfromtxt('./save/1hybrid_360_288.csv', delimiter=",")
    # predict_value_0 = np.genfromtxt('./save/1hybrid.csv', delimiter=",")
    # predict_value_1 = np.genfromtxt('./save/hybrid_test_1.csv', delimiter=",")
    # predict_value_2 = np.genfromtxt('./save/hybrid_test_2.csv', delimiter=",")
    # predict_value_3 = np.genfromtxt('./save/hybrid_test_3.csv', delimiter=",")
    # predict_value_4 = np.genfromtxt('./save/hybrid_test_4.csv', delimiter=",")
    # predict_value_5 = np.genfromtxt('./save/hybrid_test_5.csv', delimiter=",")
    # predict_value_6 = np.genfromtxt('./save/hybrid_test_6.csv', delimiter=",")
    #
    #
    predict_value_110 = predict_value[-1700:, i]
    # test_360_288 = predict_value_360_288[-2931:]
    # test_0 = predict_value_0[-2931:]
    #test_0 = predict_value_0[-1889:, i]
    # test_1 = predict_value_1[-3774:, i]
    # test_2 = predict_value_2[-3774:, i]
    # test_3 = predict_value_3[-3774:, i]
    # test_4 = predict_value_4[-3774:, i]
    # test_5 = predict_value_5[-3774:, i]
    # test_6 = predict_value_6[-3774:, i]




    # #反归一化2
    max_data = np.genfromtxt('./raw/max_value/hybrid_110_max_value.csv', delimiter=",")
    max_data_apk_i = max_data[1, i]

    Original = MTdata * max_data_apk_i
    #
    PREdata110 = predict_value_110 * max_data_apk_i
    # PREdata360_144 = test_360_144 * max_data_apk_i
    # PREdata0 = test_0 * max_data_apk_i
    # PREdata1 = test_1 * max_data_apk_i
    # PREdata2 = test_2 * max_data_apk_i
    # PREdata3 = test_3 * max_data_apk_i
    # PREdata4 = test_4 * max_data_apk_i
    # PREdata5 = test_5 * max_data_apk_i
    # PREdata6 = test_6 * max_data_apk_i



    # # mape
    y_delete_zero = Original

    mape_molecule_test_110 = np.abs((Original - PREdata110))
    # mape_molecule_test_360_144 = np.abs((MTdata1 - test_360_144))
    # mape_molecule_test_0 = np.abs((MTdata1 - test_0))
    # mape_molecule_test_1 = np.abs((MTdata1 - test_1))
    # mape_molecule_test_2 = np.abs((MTdata1 - test_2))
    # mape_molecule_test_3 = np.abs((MTdata1 - test_3))
    # mape_molecule_test_4 = np.abs((MTdata1 - test_4))
    # mape_molecule_test_5 = np.abs((MTdata1 - test_5))
    # mape_molecule_test_6 = np.abs((MTdata1 - test_6))

    y_delete_zero[y_delete_zero == 0] = 1
    mape_test_110 = np.average(mape_molecule_test_110 / y_delete_zero)
    # mape_test_360_144 = np.average(mape_molecule_test_360_144 / y_delete_zero)
    # mape_test_0 = np.average(mape_molecule_test_0 / y_delete_zero)
    # mape_test_1 = np.average(mape_molecule_test_1 / y_delete_zero)
    # mape_test_2 = np.average(mape_molecule_test_2 / y_delete_zero)
    # mape_test_3 = np.average(mape_molecule_test_3 / y_delete_zero)
    # mape_test_4 = np.average(mape_molecule_test_4 / y_delete_zero)
    # mape_test_5 = np.average(mape_molecule_test_5 / y_delete_zero)
    # mape_test_6 = np.average(mape_molecule_test_6 / y_delete_zero)
    #
    print('MAPE_110', mape_test_110)
    # print('MAPE_360_144',mape_test_360_144)
    # print('MAPE_360_288', mape_test_360_288)
    # print('Fill_random', mape_test_1)
    # print('Fill_0', mape_test_2)
    # print('Fill_1', mape_test_3)
    # print('Fill_0.5', mape_test_4)
    # print('Fill_0_and_miss_4_columns', mape_test_5)
    # print('Fill_0_and_miss_10_columns', mape_test_6)


    #。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。

    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


    #原始数据
    plt.figure(figsize=(100, 6))

    # plt.plot(MTdata * max_data_apk_i,  label = "MTdata", color='black')
    plt.plot(Original,  label = "Operation", color='black')
    plt.plot(PREdata110,  label = "MAPE_predict = " + str(mape_test_110), color='red')
    # plt.plot(PREdata360_144,  label = "MAPE_360_144 = " + str(mape_test_360_144), color='skyblue')
    # plt.plot(PREdata360_288, label="MAPE_360_288 = " + str(mape_test_360_288), color='yellow')
    #plt.plot(PREdata1,  label = "Fill_random", color='skyblue')
    #plt.plot(PREdata2,  label = "Fill_0", color='skyblue')
    #plt.plot(PREdata3,  label = "Fill_1", color='skyblue')
    #plt.plot(PREdata4,  label = "Fill_0.5", color='skyblue')
    #plt.plot(PREdata5,  label = "Fill_0_and_miss_4_columns", color='skyblue')
    # plt.plot(PREdata6,  label = "Fill_0_and_miss_10_columns", color='skyblue')


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
    plt.title("apk_"+str(i), fontsize=30,fontweight='bold')  #标题

    plt.savefig('/Users/gejiake/Desktop/apk_'+ str(i) +'.svg', bbox_inches='tight')
    # plt.show()


csv_columns = pd.read_csv('./raw/hybrid_raw/hybrid_101.csv')
csv_shape = np.array(csv_columns)
m, n = csv_shape.shape
for i in range(0, n):
    #第几个业务
    print('number:', i)
    DRAW_PIC(i)