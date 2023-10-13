"""Console script for predict_with_performance_model_exp."""
import argparse
import logging
import time
import random
import string
from . import exp_main


def init_logger(exp_id):
    # 创建一个logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler('log/exp_%s.log' % exp_id)
    file_handler.setLevel(logging.INFO)

    # 创建一个handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def cli_main():
    """Console script for predict_with_performance_model_exp."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--appkey', type=str, help="test appkey", default="demo_appkey")
    parser.add_argument('--scene_id_list', nargs='+', type=int, help="quake scene id list")
    parser.add_argument('--adjust_time', type=int, help="quake adjust delay time(second)", default=3)
    parser.add_argument('--retry_times', type=int, help="http retry times", default=3)
    parser.add_argument('--mis_id', type=str, help="mis id", default="guopanfeng")
    parser.add_argument('--metric_file', type=str, help="metric file path", default="data/metric_qps.json")
    parser.add_argument('--metric_data_interval', type=int, help="metric data interval(min)", default=1)
    parser.add_argument('--predict_file', type=str, help="predict file path", default="data/predict_qps.json")
    parser.add_argument('--predict_data_interval', type=int, help="predict data interval(min)", default=5)
    parser.add_argument('--predict_horizon', type=int, help="predict horizon(minute)", default=12)
    parser.add_argument('--raptor_cookie', type=str, help="raptor cookie", default='')
    parser.add_argument('--env', type=str, help="env", default='')
    parser.add_argument('--request_type', type=str, help="request type", default='URL')
    parser.add_argument('--performance_table_file', type=str, help="performance table file path", default='')
    parser.add_argument('--scale_idc', type=str, help="scale idc", default='gh')
    parser.add_argument('--scale_image', type=str, help="scale image", default='')
    parser.add_argument('--single_pod_qps', type=int, help="single pod qps", default=400)
    parser.add_argument('--tp99_slo', type=int, help="tp99 slo", default=9999)
    parser.add_argument('--tp999_slo', type=int, help="tp999 slo", default=9999)
    parser.add_argument('--save_metric_folder', type=str, help="save metric folder", default='')
    parser.add_argument('--scale_switch', action="store_true", help="scale switch")
    parser.add_argument('--mode', type=int, help="run mode", default=1)
    parser.add_argument('--pressure_start_qps', type=int, help="pressure start qps", default=1)
    parser.add_argument('--pressure_end_qps', type=int, help="pressure end qps", default=1)
    parser.add_argument('--pressure_qps_step', type=int, help="pressure qps step", default=1)

    args = parser.parse_args()

    current_time = int(time.time())
    random.seed(current_time)
    exp_id = ''.join(random.choices(string.ascii_letters, k=6))
    logger = init_logger(exp_id)
    logger.info("exp_id = " + exp_id)
    logger.info("Arguments: " + str(args))

    if args.mode == 1:
        exp_main.predict_with_performance_model(args, logger)
    elif args.mode == 2:
        exp_main.pressure(args, logger)

    return 0
