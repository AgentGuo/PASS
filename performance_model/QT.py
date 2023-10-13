import math
import json


def MM1_avgRT_pod(u, QPS, avgRT):
    # u: QPS a pod can process
    # QPS: predicted traffic of next period
    # avgRT: target average RT（s）
    # return #POD
    return math.ceil(QPS / (u - 1 / avgRT))
    # lambda = QPS/N, mu=u


def MM1_tailRT_pod(u, QPS, tail, target):
    # e.g. TP99 < 1s --> MM1_tailRT(u,QPS,0.99,1)
    # return #POD
    return math.ceil(QPS / (u + math.log(1 - tail) / target))


def MMs_tailRT_pod(u, QPS, tail, target):
    # 单机瓶颈，qps，尾延迟分位，尾延迟目标值（s）
    # return #POD
    return math.ceil((QPS - math.log(1 - tail) / target) / u)


def MMs_avgRT(s, u, QPS):
    # input s, output Ws, i.e. avgRT (s)
    # s: M/M/s, #server
    # lambda=QPS, mu=u
    rou = QPS / u
    rou_s = rou / s
    tmp1 = 0
    for n in range(s):
        tmp1 += (rou ** n) / math.factorial(n)
    tmp2 = (rou ** s) / (math.factorial(s) * (1 - rou_s))
    p0 = 1 / (tmp1 + tmp2)
    C_formula = tmp2 * p0
    Lq = (C_formula * rou_s) / (1 - rou_s)
    Ls = Lq + rou
    return Ls / QPS


def MMs_avgRT_pod(N, u, QPS, avgRT):
    # N:number of adjustable pods
    for s in range(2, N + 1):
        if MMs_avgRT(s, u, QPS) <= avgRT:
            return s
    return -1  # cannot satisfy target avgRT


def MMs_qps(s, u, tail, T):
    # input #pod, tail RT, output QPS
    return s * u + math.log(1 - tail) / T


def MM1_qps(s, u, tail, T):
    return (u + math.log(1 - tail) / T) * s


def table_init(table):
    lastins = 1

    for qps in range(50, 1200, 50):
        ins = MMs_tailRT_pod(100, qps, 0.99, 0.06)
        if ins > lastins:
            table[lastins] = qps - 1
            lastins = ins

if __name__ == "__main__":
    table = {}
    table_init(table)
    with open('performance_table_QT.json', 'w', encoding='utf-8') as f:
        # 将字典保存为JSON
        json.dump(table, f, ensure_ascii=False, indent=4)
    print(table)
