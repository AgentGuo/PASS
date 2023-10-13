#!/bin/bash
# 获取传入的参数
data_name=$1

# 测试模型
time python test_loaded_model.py --data "npz_data/$data_name.npz" --max_value_csv "max_data/$data_name.csv" --save_pb "./save_pb/$data_name"  --save_csv "./save/$data_name.csv"  --log "./logs/$data_name.test.log" --exps 5 --patience 15 \
    --normalize 1 --loss mae --hidCNN 100 --hidRNN 100 --hidSkip 50 --output_fun linear \
    --multi 1 --horizon 3  --highway_window 7 --window 14 --skip 7 --ps 3
