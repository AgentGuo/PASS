"""Main module."""
from datetime import datetime, timedelta
from . import utils
import time
from .quake import Quake
from .predict import Predict
from .performance_model import PerformanceModel


def predict_with_performance_model(args, logger):
    quake = Quake(args.metric_file, args.metric_data_interval, args.scene_id_list, args.mis_id,
                  args.adjust_time, args.retry_times, logger)
    quake.start_build()
    predict = Predict(args.predict_file, args.predict_data_interval, args.predict_horizon, logger)
    performance_model = PerformanceModel(args.raptor_cookie, args.appkey, args.env, args.request_type, logger,
                                         args.performance_table_file, args.scale_idc, args.mis_id, args.scale_image,
                                         args.single_pod_qps, args.tp99_slo, args.tp999_slo, args.save_metric_folder,
                                         args.scale_switch)
    # 等待quake机器就绪
    logger.info('quake压测启动中...')
    time.sleep(120)
    # 单位时间间隔
    interval = timedelta(minutes=1)
    start_time = datetime.now()
    next_minute = datetime.now()
    logger.info('start time: %s' % start_time.strftime('%Y-%m-%d %H:%M:%S'))
    quake.set_start_time(start_time)
    predict.set_start_time(start_time)
    while 1:
        now = datetime.now()
        if now < next_minute:
            time.sleep((next_minute - now).total_seconds())
            continue
        # step1 调整压测qps
        quake.adjust_qps()

        # step2 预测
        predict_result = predict.get_predict_value()
        if len(predict_result) == 0:
            break

        # step3 性能模型
        performance_model.scale_with_performance_model(predict_result)

        next_minute += interval
    return 0


def pressure(args, logger):
    quake = Quake(args.metric_file, args.metric_data_interval, args.scene_id_list, args.mis_id,
                  args.adjust_time, args.retry_times, logger)
    quake.start_build()
    predict = Predict(args.predict_file, args.predict_data_interval, args.predict_horizon, logger)
    performance_model = PerformanceModel(args.raptor_cookie, args.appkey, args.env, args.request_type, logger,
                                         args.performance_table_file, args.scale_idc, args.mis_id, args.scale_image,
                                         args.single_pod_qps, args.tp99_slo, args.tp999_slo, args.save_metric_folder, args.scale_switch)
    # 等待quake机器就绪
    logger.info('quake压测启动中...')
    time.sleep(120)

    qps = args.pressure_start_qps
    while qps <= args.pressure_end_qps:
        quake.set_quake_qps(quake.scene_task_id, qps)
        time.sleep(300)
        performance_model.raptor.refresh_metric_data()
        tp99_map = performance_model.raptor.get_metric_data('tp99')
        tp999_map = performance_model.raptor.get_metric_data('tp999')
        sorted_keys = sorted(tp99_map.keys(), reverse=True)
        total_count = min(5, len(sorted_keys))
        violate_count = 0
        for i in range(total_count):
            if tp99_map[sorted_keys[i]] > args.tp99_slo or tp999_map[sorted_keys[i]] > args.tp999_slo:
                violate_count += 1
        violate_rate = violate_count / total_count
        logger.info('violate_rate = %.2f, violate_count = %d, total_count = %d' % (violate_rate, violate_count, total_count))
        if violate_rate > 0.1:
            if performance_model.raptor.get_instance_num() >= 13:
                break
            performance_model.scale_out(1)
            time.sleep(180)
        else:
            qps += args.pressure_qps_step
    quake.set_quake_qps(quake.scene_task_id, 0)
