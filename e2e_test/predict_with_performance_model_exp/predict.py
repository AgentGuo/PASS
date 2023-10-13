from . import utils
from datetime import datetime, timedelta


class Predict:
    def __init__(self, predict_file, predict_data_interval, horizon, logger):
        self.predict_data_interval = predict_data_interval
        self.logger = logger
        self.horizon = horizon
        self.predict_data = utils.load_json_file(predict_file)
        self.start_time = datetime.now()

    def set_start_time(self, ts):
        self.start_time = ts
    def get_predict_value(self):
        from_ts = datetime.now()
        to_ts = (from_ts + timedelta(minutes=self.horizon))
        from_idx = int((from_ts - self.start_time).total_seconds() / (60 * self.predict_data_interval))
        to_idx = int((to_ts - self.start_time).total_seconds() / (60 * self.predict_data_interval))
        result = self.predict_data[from_idx+1:to_idx+1]
        self.logger.info('get predict result, length = %d, predict result = %s' % (len(result), str(result)))
        return result
