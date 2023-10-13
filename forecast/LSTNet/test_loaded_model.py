import argparse
import time
import datetime
import pandas as pd
from test_utils import *
from LSTNet import LSTNet, LSTNet_multi_inputs
import numpy as np
# from keras.models import model_from_yaml
import pickle as pk
import json
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# limit gpu memory
# def get_session(gpu_fraction=0.1):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# K.set_session(get_session())

def print_shape(data):
    for i in range(len(data.train)):
        print(data.train[i].shape, end=' ')
        # print(type(data), end=' ')
    print("")
    for i in range(len(data.valid)):
        print(data.valid[i].shape, end=' ')
    print("")
    for i in range(len(data.test)):
        print(data.test[i].shape, end=' ')
    print("")


def evaluate(y, yp):
    # rrse
    rrse = np.sqrt(np.sum(np.square(y - yp)) / np.sum(np.square(np.mean(y) - y)))

    # corr
    m, mp, sig, sigp = y.mean(axis=0), yp.mean(axis=0), y.std(axis=0), yp.std(axis=0)
    corr = ((((y - m) * (yp - mp)).mean(axis=0) / (sig * sigp))[sig != 0]).mean()

    # mape
    #mape = np.average((np.abs((y - yp)) / y)[y != 0])
    # y_delete_zero = y
    # mape_molecule = np.abs((y-yp))
    # y_delete_zero[y_delete_zero == 0] = 1
    # mape = np.average( mape_molecule / y_delete_zero)

    mape = np.average((np.abs((y - yp)) / y)[y != 0])

    # R2
    R2 = r2_score(y, yp)

    # MAE
    mae = mean_absolute_error(y, yp)

    # MSE
    mse = mean_squared_error(y, yp)

    # RMAE
    rmse = np.sqrt(mean_squared_error(y, yp))

    return rmse, mape, corr



def save_setting_file(path, item):
    item = json.dumps(item)
    try:
        with open(path, "w", encoding='utf-8') as f:
            f.write(item + ",\n")
            print("保存"+path+"文件到本地完成")
    except Exception as e:
        print("json文件写入失败,请检查路径", e)


# def main(args):
#     K.clear_session()
#     data = Data(args)
#     print_shape(data)
#
#
#     new_model = tf.keras.models.load_model(args.save_pb)
#     predict = new_model.predict(data.test[:-1])
#
#     csv_columns = pd.read_csv('./raw/hybrid_norm.csv')
#     columns_name = list(csv_columns.columns.values)
#     csv_shape = np.array(csv_columns)
#     m, n = csv_shape.shape
#     dataframe_text_data_tocsv = pd.DataFrame()
#     print('predict!!!!!!!', predict.shape)
#
#     for i in range(0, n):
#
#         APK_name = columns_name[i]
#
#         dataframe_text_data_tocsv[APK_name] = predict[0:, i]
#
#     dataframe_text_data_tocsv.to_csv(args.save_csv, index=None)
#     # df = pd.DataFrame(predict)
#     # df.to_csv(args.save_csv, index=None)
#     rmse, mape, corr = evaluate(data.test[-1], predict)
#
#
#     print("\tLoaded Test | rmse: %.4f | mape: %.4f | corr: %.4f " % (rmse, mape, corr))
#
#
#
#
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Keras Time series forecasting')
#     parser.add_argument('--data', type=str, required=True, help='location of the data file')
#     parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
#     parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
#     parser.add_argument('--hidSkip', type=int, default=10)
#     parser.add_argument('--window', type=int, default=24 * 7, help='window size')
#     parser.add_argument('--horizon', type=int, default=3)  # 预测窗口。e.g., 预测t_i,则用(t_i-horizon-window, t_i-horizon)的数据预测。
#     parser.add_argument('--skip', type=int, default=24, help='period')
#     parser.add_argument('--ps', type=int, default=3, help='number of skip (periods)')
#     parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
#     parser.add_argument('--highway_window', type=int, default=3, help='The window size of the highway component')
#     parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
#     parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
#     parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
#     parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
#     parser.add_argument('--seed', type=int, default=54321, help='random seed')
#     # parser.add_argument('--gpu', type=int, default=None)
#     parser.add_argument('--multi', type=int, default=0, help='original(0) or multi-input(1) LSTNet')
#     parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
#     parser.add_argument('--save', type=str, default='save/model.pt', help='path to save the final model')
#     parser.add_argument('--save_pb', type=str, default='save_pb/test', help='path to save the final model')
#     parser.add_argument('--log', type=str, default='logs/model.pt', help='path to save the testing logs')
#     parser.add_argument('--save_csv', type=str, default='save/model.csv', help='path to save the final model of csv')
#     # parser.add_argument('--cuda', type=str, default=True)
#     parser.add_argument('--optim', type=str, default='adam')
#     parser.add_argument('--lr', type=float, default=0.0005)
#     parser.add_argument('--loss', type=str, default='mae')
#     parser.add_argument('--normalize', type=int, default=2)
#     parser.add_argument('--output_fun', type=str, default='sigmoid')
#     parser.add_argument('--exps', type=int, default=1, help='number of experiments')
#     parser.add_argument('--patience', type=int, default=10, help='patience of early stopping')
#     args = parser.parse_args()
#
#     main(args)



def main(args):
    K.clear_session()
    data = Data(args)
    print_shape(data)
    new_model = tf.keras.models.load_model(args.save_pb)
    predict = new_model.predict(data.test[:-1])
    df = pd.DataFrame(predict)
    # df.to_csv(args.save_csv, index=None)
    max_df = pd.read_csv(args.max_value_csv)
    result = df.values * max_df.values
    columns = max_df.columns.to_list()
    result_df = pd.DataFrame(result, columns=columns)
    result_df.to_csv(args.save_csv, index=False)
    rmse, mape, corr = evaluate(data.test[-1], predict)
    print("\tLoaded Test | rmse: %.4f | mape: %.4f | corr: %.4f " %(rmse, mape, corr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras Time series forecasting')
    parser.add_argument('--data', type=str, required=True, help='location of the data file')
    parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
    parser.add_argument('--hidSkip', type=int, default=10)
    parser.add_argument('--window', type=int, default=24 * 7, help='window size')
    parser.add_argument('--horizon', type=int, default=3)  # 预测窗口。e.g., 预测t_i,则用(t_i-horizon-window, t_i-horizon)的数据预测。
    parser.add_argument('--skip', type=int, default=24, help='period')
    parser.add_argument('--ps', type=int, default=3, help='number of skip (periods)')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=3, help='The window size of the highway component')
    parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--multi', type=int, default=0, help='original(0) or multi-input(1) LSTNet')
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--save', type=str, default='save/model.pt', help='path to save the final model')
    parser.add_argument('--save_pb', type=str, default='save_pb/test', help='path to save the final model')
    parser.add_argument('--log', type=str, default='logs/model.pt', help='path to save the testing logs')
    parser.add_argument('--save_csv', type=str, default='save/model.csv', help='path to save the final model of csv')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--loss', type=str, default='mae')
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    parser.add_argument('--exps', type=int, default=1, help='number of experiments')
    parser.add_argument('--patience', type=int, default=10, help='patience of early stopping')
    parser.add_argument('--max_value_csv', type=str, default='max_value/hybrid_max_value.csv',
                        help='path to save the csv of max_value')
    args = parser.parse_args()


    main(args)



