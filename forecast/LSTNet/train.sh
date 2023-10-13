#!/bin/bash
# 获取传入的参数
data_name=$1
mkdir logs
mkdir norm_data
mkdir npz_data
mkdir max_data
mkdir save
mkdir save_pb/$data_name
mkdir save_weight

# 数据预处理
python rawdata_to_npz.py --data_name "$data_name" --raw_data "../data/$data_name.csv" --npz_file_path "npz_data" --max_value_path "max_data" --norm_value_path "norm_data"
# 训练模型
time python main.py --data "npz_data/$data_name.npz" --max_value_csv "max_data/$data_name.csv"  --save_pb "./save_pb/$data_name"  --model_name_number 0  --log "./logs/$data_name.log"  --save_weights "./save_weights/$data_name" --exps 1 --patience 15 --normalize 1 --loss mae --hidCNN 100 --hidRNN 100 --hidSkip 50 --output_fun linear --multi 1 --add_model 0 --horizon 3  --highway_window 7 --window 14 --skip 7 --ps 3
# 测试模型
time python test_loaded_model.py --data "npz_data/$data_name.npz" --max_value_csv "max_data/$data_name.csv" --save_pb "./save_pb/$data_name"  --save_csv "./save/$data_name.csv"  --log "./logs/$data_name.test.log" --exps 5 --patience 15 \
    --normalize 1 --loss mae --hidCNN 100 --hidRNN 100 --hidSkip 50 --output_fun linear \
    --multi 1 --horizon 3  --highway_window 7 --window 14 --skip 7 --ps 3
