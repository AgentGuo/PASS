# Performance Model
The goal of the performance model is to build a mapping from QPS to the number of instances, and our performance model is built based on logs.
We have implemented two methods for building performance models:
* `QT.py`: building performance models using queuing theory.
* `log_profilling.py`: building a performance model based on metric log.