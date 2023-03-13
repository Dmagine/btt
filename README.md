# atdd
## 核心代码
### 在package文件夹里
#### atdd_tuner: nni自定义，决定下一个trial的参数组合，包含at等模型修复逻辑 (nni侧面)
#### atdd_assessor: nni自定义，比较当前运行的trial和历史的trial在当前epoch的一些指标，可以提前停止trial (nni侧)
#### atdd_manager: 包装用户侧的所有操作，初始化monitor和inspector  (用户侧)
#### atdd_monitor: 收集和简单计算单个模型运行时的各种指标，以文件形式传递给assessor和tuner (用户侧)
#### atdd_inspecter: 用at等规则诊断模型，可以提前停止trial (用户侧)
## 其他内容
### 是场景相关的模型代码和配置文件以及辅助测试脚本
