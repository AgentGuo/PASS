import time
import requests
import json
from datetime import datetime, timedelta
from . import utils


class Raptor:
    def __init__(self, raptor_cookie, appkey, env, request_type, logger, retry_times=3, save_metric_folder=''):
        self.env = env
        self.request_type = request_type
        self.retry_times = retry_times
        self.logger = logger
        self.appkey = appkey
        self.raptor_cookie = {i.split("=")[0]: i.split("=")[1] for i in raptor_cookie.split("; ")}
        self.metric_data = {'qps': {}, 'tp50': {}, 'tp90': {}, 'tp99': {},
                            'tp999': {}, 'tp9999': {}, 'instance_list': {}}
        self.save_metric_folder = save_metric_folder

    def refresh_qps_tp(self, ts):
        date = ts.strftime('%Y%m%d%H')
        # 定义URL参数
        params = {'domain': '%s.ptest' % self.appkey, 'date': date,
                  'reportType': 'hour', 'ip': 'All',
                  'type': self.request_type, 'isSecond': 'false'}
        for i in range(self.retry_times):
            try:
                # 发送GET请求
                response = requests.get(
                    'http://xxxx/xxxx',
                    params=params, cookies=self.raptor_cookie)

                # 检查状态码
                content = json.loads(response.text)
                qps_data_list = content["data"]["graphs"]["hit"]["rows"]
                tp_data_list = content["data"]["graphs"]["tpLine"]["rows"]
                break
            except requests.exceptions.HTTPError as err:
                self.logger.info("get tp99 filed, request params = %s, response = %s" % (str(params), str(err)))
                self.logger.info("attempting to retry, retry count = %d" % (i + 2))
                time.sleep(1)  # 等待1秒
        else:
            self.logger.info("Reached maximum retry count(%d)" % self.retry_times)
            return
        for minute, qps_data in enumerate(qps_data_list):
            ts_str = ts.replace(minute=minute).strftime('%Y-%m-%d %H:%M')
            if qps_data['Hits'] is not None:
                self.metric_data['qps'][ts_str] = qps_data['Hits'] / 60

        for minute, tp_data in enumerate(tp_data_list):
            ts_str = ts.replace(minute=minute).strftime('%Y-%m-%d %H:%M')
            for tp_type in ['tp50', 'tp90', 'tp99', 'tp999', 'tp9999']:
                if tp_data[tp_type] is not None:
                    self.metric_data[tp_type][ts_str] = tp_data[tp_type]

    def refresh_metric_data(self):
        now = datetime.now()
        self.refresh_qps_tp(now)
        self.refresh_instance_list(now)
        if len(self.save_metric_folder) != 0:
            utils.save_json_file(self.metric_data, '%s/%s.json' % (self.save_metric_folder, self.appkey))

    def get_history_metric_data(self, ts_from, ts_to):
        current = ts_from
        delta = timedelta(hours=1)
        while current < ts_to:
            self.refresh_qps_tp(current)
            current += delta
        if len(self.save_metric_folder) != 0:
            utils.save_json_file(self.metric_data, '%s/%s.json' % (self.save_metric_folder, self.appkey))

    def refresh_instance_list(self, ts):
        ts_str = ts.strftime('%Y-%m-%d %H:%M')
        # 定义请求的URL
        url = 'http://xxxxx/xxxx'
        # 定义请求的JSON参数
        data = {
            'appkey': self.appkey,
        }
        # 将字典转换为JSON格式
        json_data = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        # 发送POST请求
        response = requests.post(url, data=json_data, headers=headers)

        # 判断请求是否成功
        if response.status_code != 200:
            self.logger.error('get instance num failed, status code = %d' % response.status_code)
            return
        content = json.loads(response.text)['data']
        instance_list = []
        for instance_data in content:
            host_name = instance_data['setName']
            if len(instance_data['swimlane']) != 0 or len(instance_data['set']) != 0:
                continue
            if self.env in host_name:
                instance_list.append(host_name)
        self.metric_data['instance_list'][ts_str] = instance_list

    def get_instance_num(self):
        data = self.metric_data['instance_list']
        if len(data) == 0:
            return -1
        # 对字典的键进行排序，得到一个列表
        max_key = max(data.keys())
        return len(data[max_key])

    def get_recent_max_qps(self):
        data = self.metric_data['qps']
        # 对字典的键进行排序，得到一个列表
        sorted_keys = sorted(data.keys(), reverse=True)
        if len(sorted_keys) == 0:
            return 0
        elif len(sorted_keys) > 3:
            sorted_keys = sorted_keys[:3]
        qps = 0
        for ts in sorted_keys:
            qps = max(qps, data[ts])
        return qps

    def get_recent_min_tp(self, tp_type):
        data = self.metric_data[tp_type]
        # 对字典的键进行排序，得到一个列表
        sorted_keys = sorted(data.keys(), reverse=True)
        if len(sorted_keys) == 0:
            return 0
        elif len(sorted_keys) > 3:
            sorted_keys = sorted_keys[:3]
        tp = 1e9
        for ts in sorted_keys:
            tp = min(tp, data[ts])
        return tp

    def get_metric_data(self, metric_type):
        return self.metric_data[metric_type]
