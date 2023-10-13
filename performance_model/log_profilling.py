import pandas as pd
import matplotlib
import functools

appkey = "demo_appkey"
data=pd.read_csv("demo_svc_metric_data.csv")

QoS_tp99 = 5
QoS_tp999 = 20
# Top_k = 3
# violation_threshold = 0.01
cap = 100
RATE = 0.99
performance_table = {}

cnt_group = data.groupby('cnt')
for cnt,group in cnt_group:
    qps_count = {}
    violation_count = {}
    for row in group.itertuples():
        qps = row[3] // cap
        # tp99 = row[-3]
        tp999 = row[-2]
        if qps not in qps_count:
            qps_count[qps] = 0
            violation_count[qps] = 0
        qps_count[qps] += 1
        if tp999 > QoS_tp999:
            violation_count[qps] += 1
    def rate(qps):
        total_count = 0
        total_violation = 0
        for k,v in qps_count.items():
            if k>qps:
                break
            total_count += v
            total_violation += violation_count[k]
        return (total_count-total_violation)/total_count

    lt = sorted(qps_count, key = lambda k:(rate(k) > RATE, k), reverse=True)
    if rate(lt[0]) > RATE:
        performance_table[cnt] = lt[0] * cap
    else:
        performance_table[cnt] = 0

    # lt = sorted(qps_count.items(), key = lambda kv:(rate(kv[0]), kv[0]), reverse=True)
    # if len(lt) > Top_k:
    #     for i in range(Top_k):
    #         qps = lt[i][0]
    #         if violation_count[qps]/qps_count[qps] < violation_threshold:
    #             performance_table[cnt] = qps * cap
    #             break
    # if cnt not in performance_table:
    #     last_qps = 0
    #     for qps,count in qps_count.items():
    #         if violation_count[qps]/count < violation_threshold:
    #             last_qps = qps
    #         else:
    #             break
    #     performance_table[cnt] = last_qps * cap

print(performance_table)

last_qps = 0
for k,v in performance_table.items():
    if v<last_qps:
        performance_table[k] = last_qps
    last_qps = performance_table[k]

print(performance_table)
