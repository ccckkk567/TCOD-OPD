# 两张 H200 的 ALFWorld 训练资源方案（已批准，待 HPC 验证）

用户已明确批准此两卡共置方案，未批准之前的单卡串行改造。
集群指南列明 testqueue 为 H200 队列，使用 --gres=gpu:2；作业内检查实际设备，
不凭名称假定显存容量，不换收费分区。先2+2恢复验证，原50步pilot以后放行。

拟分配：GPU 0 运行一个学生vLLM引擎和一个教师vLLM引擎；GPU 1 运行原veRL
训练worker。学生1.7B、教师8B，推理TP=1，训练SP=1。保留原异步采样策略、
teacher上下文/切片、奖励处理、loss、有效训练batch和原更新逻辑。
不会执行整批生成→评分→更新的单卡串行重写。

必须修改的边界：
- config和validator新增默认关闭的两卡共置模式，明确物理GPU数与推理逻辑角色数。
- common/models/__init__.py 为两个推理actor提供同一张卡上的资源份额，严格限定
  在同一个 placement bundle；训练池仍独占另一张GPU，不伪报物理GPU数量。
- vLLM初始化顺序和显存预算分开控制，必须检查共享卡初始化/显存profiling及
  teacher API是否正常；预算只是待验证配置，不保证不会OOM。
- train启动器验证两张实际GPU，沿用原权重同步、保存与恢复接口；训练world size=1。
- 配置只按批准的资源范围改变：两个学生引擎缩为一个，各推理TP=1、训练SP=1；
  不自行减少有效batch、token预算或改变采样温度。必要时进一步说明再决策。

影响：并发数和张量规约方式变化，吞吐、样本到达顺序及实际滞后分布可能变化；
虽保留原异步策略，仍不保证与8卡逐样本或数值等价。共享vLLM初始化、显存峰值、
NCCL同步及world size=1恢复都需要真实HPC验证。不得用两个独立推理job的成功
来代替此验证。现有两条单卡E0基线可继续并行，不依赖本方案批准或完成。

实现入口为 `cluster.shared_inference_gpu=true`，默认false。一个完整GPU bundle
承载两个各申请0.5 GPU的actor，并逐一等待初始化、记录和核对实际GPU编号。
不能把0.5资源份额当成显存隔离；vLLM另设学生0.25、教师0.55的显存预算。
依据：[Ray placement groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)、
[vLLM per-instance memory budgets](https://docs.vllm.ai/en/v0.7.0/serving/engine_args.html)。
GPU 0/1是角色示意，实际编号由Ray分配，并非硬编码设备编号。

`submit_alfworld_recovery.sh REVISION` 现在只提交两卡配置：每段2张H200、
16 CPU、128G内存、4h上限；第一段成功才以afterok放行恢复段。
这是已批准的2+2工程验证，不自动提交50步pilot。
run_name为alfworld-s17-two-gpu-smoke，与之前8卡配置产物隔离。
正式baseline可与本链同时排队。未通过实际初始化/更新/保存/恢复之前不得
宣称两卡训练已验证；恢复到4步的COMPLETE也不等于逐位恢复等价。
