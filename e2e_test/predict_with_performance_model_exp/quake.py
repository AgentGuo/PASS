from datetime import datetime, timedelta
import requests
import time
from . import utils
import json


class Quake:
    def __init__(self, metric_file, metric_data_interval, scene_id_list, mis_id, adjust_time, retry_times, logger):
        self.metric_data_interval = metric_data_interval
        self.metric_data = utils.load_json_file(metric_file)
        self.logger = logger
        self.retry_times = retry_times
        self.adjust_time = adjust_time
        self.mis_id = mis_id
        self.scene_id = scene_id_list[0]
        self.scene_id_list = scene_id_list
        self.scene_id_idx = 0
        self.scene_task_id = -1
        self.scene_backup_task_id = -1
        self.scene_task_update_time = datetime.now()
        self.start_time = datetime.now()

    def set_start_time(self, ts):
        self.start_time = ts

    def next_scene_id(self):
        self.scene_id_idx += 1
        self.scene_id = self.scene_id_list[self.scene_id_idx % len(self.scene_id_list)]

    def start_build(self):
        # 定义URL参数
        params = {'sceneId': str(self.scene_id), 'misId': self.mis_id,
                  'taskModel': str(0)}
        for i in range(self.retry_times):
            try:
                # 发送GET请求
                response = requests.get(
                    'http://xxxx/api/startBuild',
                    params=params)

                # 检查状态码
                self.logger.info("start build scene(%d), response = %s" % (self.scene_id, response.text))
                result = json.loads(json.loads(response.text)["data"])
                if result["success"]:
                    if self.scene_task_id == -1:
                        self.scene_task_id = result["sceneTaskId"]
                        self.scene_task_update_time = datetime.now()
                        self.logger.info("update scene_task_id = %d" % self.scene_task_id)
                    else:
                        self.scene_backup_task_id = result["sceneTaskId"]
                        self.logger.info("update scene_backup_task_id = %d" % self.scene_backup_task_id)
                    self.next_scene_id()
                break
            except requests.exceptions.HTTPError as err:
                self.logger.info("build scene(%d) filed, response = %s" % (self.scene_id, str(err)))
                self.logger.info("attempting to retry, retry count = %d" % (i + 2))
                time.sleep(1)  # 等待1秒
        else:
            self.logger.info("Reached maximum retry count(%d)" % self.retry_times)

    def update_scene_task(self):
        if (datetime.now() - self.scene_task_update_time) > timedelta(hours=8, minutes=3):
        # if (datetime.now() - self.scene_task_update_time) > timedelta(minutes=3):
            self.logger.info("before switching scene task, scene_task_id = %d, scene_backup_task_id = %d" %
                             (self.scene_task_id, self.scene_backup_task_id))
            tmp = self.scene_task_id
            # 关闭之前的quake压测
            self.set_quake_qps(tmp, 0)
            self.scene_task_id = self.scene_backup_task_id
            self.scene_backup_task_id = tmp
            self.scene_task_update_time = datetime.now()
            self.logger.info("after switching scene task, scene_task_id = %d, scene_backup_task_id = %d" %
                             (self.scene_task_id, self.scene_backup_task_id))
        elif (datetime.now() - self.scene_task_update_time) > timedelta(hours=8):
            self.start_build()

    def adjust_qps(self):
        self.update_scene_task()
        qps_idx = int((datetime.now() - self.start_time).total_seconds() / (60 * self.metric_data_interval))
        if qps_idx < len(self.metric_data):
            qps = self.metric_data[qps_idx]
        else:
            qps = 0
        self.set_quake_qps(self.scene_task_id, qps)

    def set_quake_qps(self, scene_task_id, qps):
        # 定义URL参数
        params = {'sceneTaskId': str(scene_task_id), 'qps': str(qps),
                  'runTime': str(self.adjust_time), 'misId': self.mis_id}

        # 最多重试retry_times次
        for i in range(self.retry_times):
            try:
                # 发送GET请求
                response = requests.get(
                    'http://xxxxx/xxxx',
                    params=params)

                # 检查状态码
                response.raise_for_status()

                self.logger.info("adjust qps to %d(scene_task_id = %d), response = %s" %
                                 (qps, scene_task_id, response.text))
                break
            except requests.exceptions.HTTPError as err:
                self.logger.info("adjust qps to %d failed, response = %s" % (qps, str(err)))
                self.logger.info("attempting to retry, retry count = %d" % (i + 2))
                time.sleep(1)  # 等待1秒
        else:
            self.logger.info("Reached maximum retry count(%d)" % self.retry_times)
