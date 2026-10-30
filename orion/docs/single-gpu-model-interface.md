# ALFWorld / ScienceWorld 单 GPU 工程检查

用户选择每个作业一张 GPU，两个环境并行；本轮不提交 WebShop。

`submit_model_interfaces.sh REVISION` 固定源码快照，提交两个独立 testqueue
作业，各 1 GPU、4 CPU、48 GB 内存，时间上限 20 分钟（不是预计耗时）。
使用已有共享环境和模型；不安装、不下载、不 clone、不修改训练代码。

每个作业检查两个任务，每个任务至多三轮：先用 Qwen3-1.7B 生成动作并调用
原环境辅助函数；学生进程退出后，用 Qwen3-8B 对保存的学生前缀和响应计算
逐 token 条件概率。两个模型不会同时驻留 GPU。
ALFWorld 使用已有 smoke 划分的前两个任务；ScienceWorld 使用 boil/melt 的
variation 0。这是短交互诊断，不是任务成功率评估。

后端为独立 Transformers 诊断实现，使用原 prompt/parser 辅助函数。
教师使用学生前缀，未验证原 OPD 的教师上下文构造和切片；未运行 vLLM/Ray
生产 workflow、梯度更新或 checkpoint 恢复。结果明确记录这些验证范围。
原 TCOD 的 config_validator 将学生推理、辅助教师和 trainer GPU 分开计数，
在线训练要求推理资源总数严格小于总 GPU 数，因而不能仅把 GPU 总数改成 1
就运行原生 OPD。单卡原生训练需要另行讨论资源调度改造。

结果在 `storage_ext/cjk/reports/tcod-opd/model-interface/<benchmark>-<job>/`：
`student.json` 保存交互，`scored.json` 保存教师评分，`status.json` 表示两阶段通过；
发生异常则写 `failure.json` 并非零退出。没有 `status.json` 不表示通过。
日志在 `cjk/logs/tcod-opd/model-interface/check-<job>.log`。
