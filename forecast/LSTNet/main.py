import argparse
# import tensorflow
import time
import datetime
from utils import *
from LSTNet import LSTNet, LSTNet_multi_inputs
import numpy as np
# from keras.models import model_from_yaml
import pickle as pk
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from rawdata_to_npz import *


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
    rrse = np.sqrt(np.sum(np.square(y-yp)) / np.sum(np.square(np.mean(y)-y)))
    # corr
    #m, mp = np.mean(y, axis=0), np.mean(yp, axis=0)
    #corr = np.mean(np.sum((y-m)*(yp-mp), axis=0) / np.sqrt(np.sum(np.square(y-m), axis=0)*np.sum(np.square(yp-mp), axis=0)))
    m, mp, sig, sigp = y.mean(axis=0), yp.mean(axis=0), y.std(axis=0), yp.std(axis=0)
    corr = ((((y-m)*(yp-mp)).mean(axis=0) / (sig*sigp))[sig!=0]).mean()
    #corr = ((((y-m)*(yp-mp)).mean(axis=0) / (sig*sigp))).mean()
    # mape = mean_absolute_percentage_error(y, yp) #分母为0，导致mape值变得特别大
    # mape = np.average(np.abs((y-yp) / (y + 0.1))) #mape解决方式1

    # # mape解决方式2
    # y_delete_zero = y
    # mape_molecule = np.abs((y-yp))
    # y_delete_zero[y_delete_zero == 0] = 1
    # mape = np.average(mape_molecule / y_delete_zero)

    # mape解决方式3
    mape = np.average((np.abs((y-yp)) / y)[y!=0])

    R2 = r2_score(y, yp)
    mae = mean_absolute_error(y, yp)
    mse = mean_squared_error(y,yp)
    rmse = np.sqrt(mean_squared_error(y, yp))
    # print(type(y))


    return rmse, corr, mape

def main(args, exp):
    K.clear_session()
    flog = open(args.log, "a")
    s = "\nExp {}".format(exp)
    print(s)
    flog.write(s+"\n")
    now = str(datetime.datetime.now())
    print(now)
    flog.write(now+"\n")
    flog.flush()

    data = Data(args)
    print_shape(data)

    if args.multi==1:
        if args.add_model == 0:
            model = LSTNet_multi_inputs(args, data.m).make_model()
        else:
            model = LSTNet_multi_inputs(args, data.m).make_model()
            model.load_weights(args.save_weights)
            print("加载模型成功")
    else:
        model = LSTNet(args, data.m).make_model()


    ### Train ###
    test_result = [1e6, -1e6]
    best_valid = [1e6, -1e6]
    pat = 0
    bs = int(args.batch_size)
    l = len(data.train[0])
    order = np.arange(l)
    train_batch_num = int(l/bs)
    for e in range(1,args.epochs+1):
        tt = time.time()
        np.random.shuffle(order)
        if args.multi:
            x1, x2, y = data.train[0][order].copy(), data.train[1][order].copy(), data.train[2][order].copy()
        else:
            x, y = data.train[0][order].copy(), data.train[1][order].copy()
        for b in range(train_batch_num):
            print("\r%d/%d" %(b+1,train_batch_num), end='')
            if args.multi:
                b_x1, b_x2, b_y = x1[b*bs:(b+1)*bs], x2[b*bs:(b+1)*bs], y[b*bs:(b+1)*bs]
                model.train_on_batch([b_x1, b_x2], b_y)
            else:
                b_x, b_y = x[b*bs:(b+1)*bs], y[b*bs:(b+1)*bs]
                model.train_on_batch(b_x, b_y)
        rmse, corr, mape = evaluate(data.valid[-1], model.predict(data.valid[:-1], batch_size=bs))
        et = time.time()-tt
        print("\r%d | Valid | rmse: %.4f | corr: %.4f | mape: %.4f | time: %.2fs" %(e, rmse, corr, mape, et))

        if (corr-rmse) >= (best_valid[1]-best_valid[0]):
            best_valid = [rmse, corr]
            pat = 0
            # valid
            ss = "{} | Valid | rmse: {:.4f} | mape: {:.4f} | corr: {:.4f} | approx epoch time: {:.2f}s".format(e, rmse, mape, corr, et)
            # test
            rmse, corr, mape = evaluate(data.test[-1], model.predict(data.test[:-1], batch_size=bs))
            s = "{} | Test | rmse: {:.4f} | mape: {:.4f} | corr: {:.4f} | approx epoch time: {:.2f}s".format(e, rmse, mape, corr, et)
            print("\t"+s)
            flog.write(ss + "\n")
            flog.write(s+"\n")
            flog.flush()
            test_result = [mape, corr]


            # #can't use model.save(args.save) due to JSON Serializable error, so need to save like this:
            # yaml = model.to_yaml()
            # W = model.get_weights()
            #
            #
            # #train write mode
            # with open(args.save, "wb") as fw:
            #     pk.dump(yaml, fw, protocol=pk.HIGHEST_PROTOCOL)
            #     pk.dump(W, fw, protocol=pk.HIGHEST_PROTOCOL)


            # 保存模型结构和参数到文件
            tf.keras.models.save_model(model, args.save_pb, overwrite=True, include_optimizer=True)  # 默认生成 .pb 格式模型，也可以通过save_format 设置 .h5 格式
            print('模型已保存')

            #.h5
            model.save_weights(args.save_weights, overwrite=True)
            print('参数已保存')



            '''
            # Test loaded model
            with open(args.save, "rb") as fp:
                new_yaml = pk.load(fp)
                new_W = pk.load(fp)
            new_model = model_from_yaml(new_yaml)
            new_model.set_weights(new_W)
            #mape, corr = evaluate(data.test[1], new_model.predict(data.test[0]), data.mx[0])
            mape, corr = evaluate(data.test[-1], new_model.predict(data.test[:-1]))
            # print("data.test[-1]: ", data.test[-1])
            # print("data.test[2]: ", data.test[2])
            # print("data.test[0]: ", data.test[0])
            # print("data.test[1]: ", data.test[1])
            # print("data.test[0].shape: ", data.test[0].shape)
            print("\tLoaded Test | mape: %.4f | corr: %.4f " %(mape, corr))
            '''
        else:
            pat += 1
        if pat==args.patience: # early stopping
            break

    s = "End of Exp {}".format(exp)
    print(s)
    flog.write(s+"\n")
    flog.flush()
    flog.close()


    return test_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras Time series forecasting')
    parser.add_argument('--data', type=str, default='./data/hybrid_norm.npz', help='location of the data file')
    parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
    parser.add_argument('--hidSkip', type=int, default=10)
    parser.add_argument('--window', type=int, default=24*7, help='window size')
    parser.add_argument('--horizon', type=int, default=3)   #预测窗口。e.g., 预测t_i,则用(t_i-horizon-window, t_i-horizon)的数据预测。
    parser.add_argument('--skip', type=int, default=24, help='period')
    parser.add_argument('--ps', type=int, default=3, help='number of skip (periods)')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=3, help='The window size of the highway component')
    parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    #parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--multi', type=int, default=0, help='original(0) or multi-input(1) LSTNet')
    parser.add_argument('--add_model', type=int, default=0, help='model load(1) or not(0)')
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--save', type=str,  default='save/model.pt', help='path to save the final model')
    parser.add_argument('--save_pb', type=str, default='save_pb/test', help='path to save the final model')
    parser.add_argument('--save_weights', type=str, default='save_weights/model.h5', help='path to save weights of the final model')

    parser.add_argument('--max_value_csv', type=str, default='max_value/hybrid_max_value.csv', help='path to save the csv of max_value')
    parser.add_argument('--normalization_csv', type=str, default='hybrid.csv', help='path to save the csv of normalization')
    parser.add_argument('--save_predict_result_csv', type=str, default='save/predict_result.csv', help='path to save the csv of predict result')
    parser.add_argument('--model_name_number', type=int, default=0, help='the number of model name (model_num). e.g., model_0, model_1, etc.')


    parser.add_argument('--log', type=str,  default='logs/model.pt', help='path to save the testing logs')
    #parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--loss', type=str, default='mae')
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    parser.add_argument('--exps', type=int, default=1, help='number of experiments')
    parser.add_argument('--patience', type=int, default=10, help='patience of early stopping')
    args = parser.parse_args()



    # #raw_to_npz
    # max_value_name = args.max_value_csv
    # normalization_csv_name = args.normalization_csv
    # raw_to_npz(normalization_csv_name, max_value_name)
    #



    test = []
    for exp in range(1, args.exps+1):
        test.append(main(args, exp))
    test = np.array(test)
    avg = np.mean(test, axis=0)
    best = test[np.argmax(test[:,1]-test[:,0]), :]
    s = 'Average result | mape: {:.4f} | corr: {:.4f}'.format(avg[0], avg[1])
    ss = 'Best result | mape: {:.4f} | corr: {:.4f}'.format(best[0], best[1])
    with open(args.log, "a") as flog:
        flog.write(s+"\n")
        flog.write(ss+"\n")






