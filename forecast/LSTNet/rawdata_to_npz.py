import numpy as np
import pandas as pd
import argparse


def raw_to_npz(data_name, raw_data_path, npz_file_path, max_value_path, norm_value_path):
    # csv_columns = pd.read_csv('./raw/hybrid_raw/'+ raw_data)
    raw_data = pd.read_csv(raw_data_path)
    columns_name = list(raw_data.columns.values)

    csv_shape = np.array(raw_data)
    m, n = csv_shape.shape

    # save max_value
    dataframe_max = pd.DataFrame(index=list('q'))
    list_index = [1] * m
    dataframe_normalization = pd.DataFrame(index=list_index)

    for i in range(1, n):
        APK_name = columns_name[i]

        max_value = np.max(abs(csv_shape[0:, i]))
        dataframe_max[APK_name] = max_value

        # normalization
        data = csv_shape[0:, i]
        normalization_data = data / max_value
        dataframe_normalization[APK_name] = normalization_data

    # max_value to csv
    dataframe_max.to_csv('%s/%s.csv' % (max_value_path, data_name), sep=',', index=False, header=True)

    # normalization_value to csv
    dataframe_normalization.to_csv('%s/%s.csv' % (norm_value_path, data_name), index=False, sep=',')

    # more one flow -> one model
    A = dataframe_normalization.astype(np.float32)
    np.savez_compressed('%s/%s' % (npz_file_path, data_name), a=A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="")
    parser.add_argument('--raw_data', type=str, default="")
    parser.add_argument('--npz_file_path', type=str, default="")
    parser.add_argument('--max_value_path', type=str, default="")
    parser.add_argument('--norm_value_path', type=str, default="")
    args = parser.parse_args()
    # raw_to_npz
    # max_value_name = 'max_value/hybrid_7_online_max_value.csv'
    # normalization_csv_name = 'hybrid_7_online.csv'
    # raw_data = 'hybrid_7_online.csv'
    # print(args.data_name, args.raw_data, args.npz_file_path, args.max_value_path, args.norm_value_path)
    raw_to_npz(args.data_name, args.raw_data, args.npz_file_path, args.max_value_path, args.norm_value_path)
