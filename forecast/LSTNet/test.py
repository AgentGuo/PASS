import numpy as np
import pandas as pd
import pickle as pk
import csv

import os
from data_preprocessing import *
from tojson import *
from rawdata_to_npz import *

#真实环境需要开启，测试环境注释
#data_proprecessing

# raw_data_name = 'hybrid_raw/hybrid.csv'
# prepro_data_name = 'hybrid_raw/hybrid_101.csv'
# SQL_CSV_toCSV(raw_data_name, prepro_data_name)


#raw_to_npz
max_value_name = 'max_value/hybrid_101_max_value.csv'
normalization_csv_name = 'hybrid_101.csv'
raw_to_npz(normalization_csv_name, max_value_name)

# Configuration file tojson
TO_JSON(max_value_name)




#
# #根据周期因子的mape值，挑选apk
# csv_columns = pd.read_csv('./raw/hybrid_raw/mape2.csv')
#
# columns_name = list(csv_columns.columns.values)
# print(csv_columns.loc[0])
#
# #选择mape符合标准的
# csv = np.array(csv_columns)
# m, n = csv.shape
# for i in range(0, m):
#     mape = csv[i, 1]
#     if mape > 0.5:
#         apk_cell_name = csv[i, 0]
#         apk_name = apk_cell_name.split("^")[0]
#
#
# # 挑选apk值
# csv = np.array(csv_columns)
# m, n = csv.shape
# lst = []
# for i in range(0, m):
#     apk_cell_name = csv[i, 0]
#     apk_name = apk_cell_name.split("^")[0]
#     if apk_name not in lst:
#         lst.append(apk_name)
#
# print(lst)
# print(len(lst))


#
#
# #-------------2--------------
# csv_columns1 = pd.read_csv('./raw/hybrid_raw/mape.csv')
#
# columns_name1 = list(csv_columns1.columns.values)
# print(csv_columns1.loc[0])
#
# #选择mape符合标准的
# csv1 = np.array(csv_columns1)
# m, n = csv1.shape
# for i in range(0, m):
#     mape = csv1[i, 1]
#     if mape > 0.5:
#         apk_cell_name1 = csv1[i, 0]
#         apk_name1 = apk_cell_name1.split("^")[0]
#
#
# # 挑选apk值
# csv1 = np.array(csv_columns1)
# m, n = csv1.shape
# lst1 = []
# for i in range(0, m):
#     apk_cell_name1 = csv1[i, 0]
#     apk_name1 = apk_cell_name1.split("^")[0]
#     if apk_name1 not in lst1:
#         lst1.append(apk_name1)
#
# print(lst1)
# print(len(lst1))
#
#
# lst10 = [x for x in lst if x not in lst1]
# print('不重复的apk：')
# print(lst10)
#
#

