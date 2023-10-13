import numpy as np
import pandas as pd


#SQL_CSV to CSV
def SQL_CSV_toCSV(raw_data_name, prepro_data_name):

    df = pd.read_csv('./raw/' + raw_data_name, header=0, dtype={'ts':int, 'metric_value':float})
    pivoted = df.pivot(index="ts", columns="group_lable",values="metric_value").fillna(1)
    pivoted.sort_values("ts",inplace=True)

    pivoted.to_csv('./raw/' + prepro_data_name, sep=',', index=True, header=True)
