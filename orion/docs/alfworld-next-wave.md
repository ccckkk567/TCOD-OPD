# ALFWorld 第一批实验

当前证据：单卡诊断两个任务六轮生成/环境调用/教师评分通过。未验证原生训练、
恢复、原生教师上下文或后端等价性。研究允许没有预期差距，不以差距作为通过条件。

本批 E0：固定现有 analysis_dev 清单中按 task_id 排序后以 seed 42 打乱的前
32 个任务，保留任务清单；学生 Qwen3-1.7B 和教师 Qwen3-8B 各两次采样，
各 1 GPU、4h 时间上限，独立并行。冻结清单任务 1 GPU、5min，随后两条
评估以 afterok 依赖它。安装写任务使用 afterany 等待，运行时环境读锁检查。
采样显式 temperature=1、top_p=1、top_k=0、512 输出 token、最多30轮、
enable_thinking=False。独立 Transformers 结果不等于原生 TCOD backend。
不重用先前被模型默认值覆盖采样参数的短诊断作为正式基线。

提交入口：`submit_alfworld_baselines.sh REVISION`。不访问官方 test 或 confirm_holdout。
使用共享 Git 对象的固定 worktree，原 checkout 不变；模型、环境、数据不复制。
原生记录和独立评估均保留 TextWorld 对象的结构化序列化内容。

训练准备入口：`submit_alfworld_recovery.sh REVISION --approved-eight-gpu`，
仅在明确批准恢复 8 GPU 后调用。学生1.7B、教师8B，原配置2步，随后 afterok
恢复到4步，沿用此前批准配置和4h上限。该链可与 E0 并行，作业是否实际同时
运行由调度器决定。未加入50步训练，避免未检查真实恢复/评分证据就自动放行。

下一轮：检查真实训练记录、checkpoint 恢复证据与原生评分一致性，然后确定
50步实验及后续候选、前缀续写和诊断的依赖。4B/30B资产就绪情况未知，本批不
假定可用；E5正式干预、E6确认继续保持单独决策，不从初始结果挑有利任务。
