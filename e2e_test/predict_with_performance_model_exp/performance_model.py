from .raptor import Raptor
from . import utils
import json
import requests
from datetime import datetime, timedelta
import math


class PerformanceModel:
    def __init__(self, raptor_cookie, appkey, env, request_type, logger,
                 performance_table_file, scale_idc, mis_id, scale_image,
                 single_pod_qps, tp99_slo, tp999_slo, save_metric_folder, scale_switch):
        self.scale_switch = scale_switch
        self.single_pod_qps = single_pod_qps
        self.tp99_slo = tp99_slo
        self.tp999_slo = tp999_slo
        self.scale_image = scale_image
        self.performance_table = None
        self.reversed_performance_table = None
        self.appkey = appkey
        self.env = env
        self.raptor = Raptor(raptor_cookie, appkey, env, request_type, logger, 3, save_metric_folder)
        self.logger = logger
        self.load_performance_table(performance_table_file)
        self.scale_idc = scale_idc
        self.last_scale_ts = datetime.now() - timedelta(minutes=5)
        self.last_reactive_scale_ts = datetime.now()
        self.mis_id = mis_id

    def scale_with_performance_model(self, predict_qps_list):
        # step0 刷新监控数据
        self.raptor.refresh_metric_data()
        # step1 获取现有实例列表
        instance_num = self.raptor.get_instance_num()
        # step2 查询性能模型所需实例数
        expect_instance_num = self.query_performance_model(instance_num, predict_qps_list)
        self.logger.info('current instance num = %d, performance model recommend instance num = %d' % (instance_num, expect_instance_num))
        # step3 执行扩缩动作
        if self.scale_switch:
            self.scale(expect_instance_num - instance_num)

    def scale(self, instance_num):
        if instance_num != 0:
            self.logger.info('preparing to perform scale')
        else:
            return
        # 扩缩容冷却时间
        if datetime.now() - self.last_scale_ts < timedelta(minutes=3):
            self.logger.info('due to the recent triggering of scale, temporarily skip this scale')
            return
        if instance_num > 0:
            self.scale_out(instance_num)
        else:
            if (datetime.now() - self.last_reactive_scale_ts) < timedelta(minutes=10):
                self.logger.info('due to the recent triggering of reactive scale, temporarily skip this scale in')
                return
            self.scale_in(abs(instance_num))
        self.last_scale_ts = datetime.now()

    def scale_in(self, instance_num):
        # 定义请求的URL
        url = 'http://xxxxx/xxxx'
        # 定义请求的JSON参数
        data = {
            'appkey': self.appkey,
            'idc': self.scale_idc,
            'env': self.env,
            'num': instance_num,
            'operator': self.mis_id
        }
        # 将字典转换为JSON格式
        json_data = json.dumps(data)
        headers = {'Content-Type': 'application/json',
                   'auth-token': 'xxxxx'}
        # 发送POST请求
        response = requests.post(url, data=json_data, headers=headers)

        # 判断请求是否成功
        if response.status_code != 200:
            self.logger.error('scale in %d instance failed, status code = %d' % (instance_num, response.status_code))
            return None
        self.logger.info('scale in %d instance, response = %s' % (instance_num, response.text))

    def scale_out(self, instance_num):
        # 定义请求的URL
        url = 'http://xxxx/api/scaleout'
        # 定义请求的JSON参数
        data = {
            'appkey': self.appkey,
            'idc': self.scale_idc,
            'env': self.env,
            'num': instance_num,
            'image': self.scale_image
        }
        # 将字典转换为JSON格式
        json_data = json.dumps(data)
        headers = {'Content-Type': 'application/json',
                   'auth-token': 'xxxx'}
        # 发送POST请求
        response = requests.post(url, data=json_data, headers=headers)

        # 判断请求是否成功
        if response.status_code != 200:
            self.logger.error('scale out %d instance failed, status code = %d' % (instance_num, response.status_code))
            return None
        self.logger.info('scale out %d instance, response = %s' % (instance_num, response.text))

    def MMs_qps(self, s, u, tail, T):
        # input #pod, tail RT, output QPS
        return s * u + math.log(1 - tail) / T

    def query_performance_model(self, current_instance_num, predict_qps_list):
        # step1 get metric qps
        qps = self.raptor.get_recent_max_qps()
        self.logger.info('raptor recent 3min max qps = %d' % qps)

        # step2 get predict qps
        if len(predict_qps_list) != 0:
            qps = max(qps, max(predict_qps_list))
        self.logger.info('after predict, qps = %d' % qps)

        # step3 check reactive scale
        tp99 = self.raptor.get_recent_min_tp('tp99')
        tp999 = self.raptor.get_recent_min_tp('tp999')
        self.logger.info('raptor recent 3 min tp99 = %d, tp999 = %d' % (tp99, tp999))
        if (tp99 > self.tp99_slo or tp999 > self.tp999_slo) and (datetime.now() - self.last_reactive_scale_ts) > timedelta(minutes=10):
            if tp99 != 9999:
                reactive_qps = self.MMs_qps(current_instance_num, self.single_pod_qps, 0.99, tp99 / 1000)
            else:
                reactive_qps = self.MMs_qps(current_instance_num, self.single_pod_qps, 0.999, tp999 / 1000)
            self.logger.info('trigger reactive scale, reactive_qps = %d, qps = %d' % (reactive_qps, qps))
            if reactive_qps > qps:
                self.last_reactive_scale_ts = datetime.now()
                qps = reactive_qps

        # step4 query performance table
        qps_list = list(filter(lambda key: key > qps, self.reversed_performance_table.keys()))
        if len(qps_list) == 0:
            min_qps = max(self.reversed_performance_table.keys())
        else:
            min_qps = min(qps_list)
        instance_num = self.reversed_performance_table[min_qps]
        return instance_num

    def load_performance_table(self, performance_table_file):
        tmp_dict = utils.load_json_file(performance_table_file)
        performance_table, reversed_performance_table = {}, {}
        for pod_num, qps in tmp_dict.items():
            performance_table[int(pod_num)] = qps
        for pod_num, qps in performance_table.items():
            if qps in reversed_performance_table:
                reversed_performance_table[qps] = min(reversed_performance_table[qps], int(pod_num))
            else:
                reversed_performance_table[qps] = pod_num
        self.performance_table, self.reversed_performance_table = performance_table, reversed_performance_table
