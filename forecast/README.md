# Forecast
Prediction is an important part of our autoscaling framework. We use prediction algorithms to predict some monitoring indicators of services, such as QPS, CPU usage, and instance count, and other timing information
## Forecast Description
We have used many open-source prediction algorithms, which are divided into online prediction and offline prediction.
* online prediction:
  * LSTNet: [https://github.com/Lorne0/LSTNet_keras](https://github.com/Lorne0/LSTNet_keras)
  * PatchTST: [https://github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)
* offline prediction:
  * RobustSTL: [https://github.com/LeeDoYup/RobustSTL](https://github.com/LeeDoYup/RobustSTL)
  * prophet: [https://facebook.github.io/prophet/docs/quick_start.html](https://facebook.github.io/prophet/docs/quick_start.html)
  * TiDE: [https://github.com/google-research/google-research/tree/master/tide](https://github.com/google-research/google-research/tree/master/tide)
  * seasonal index
