# vLLM EngineCore Ray 地址修复

2026-10-31日志：两个SHARED_INFERENCE角色均报告同一H200，随后两个EngineCore
分别出现Started a local Ray instance，最终Synchronizer.get_actor找不到synchronizer。
原作业同步器仍出现在调用栈中，故不能简单理解为没有创建同步器。

修复仅在默认关闭的两卡布局actor runtime_env注入当前作业真实GCS地址为
RAY_ADDRESS，让spawn出的非Ray EngineCore在首次Ray调用时连接原集群。
namespace仍由原调用参数传递，不改变NCCL/teacher上下文/采样或更新逻辑。
不使用address=auto，不硬编码其他作业地址，不新增重复同步器。
依据：https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html
当前地址接口：https://docs.ray.io/en/latest/_modules/ray/runtime_context.html

本地4项资源/配置回归检查通过，新增断言两个actor均拿到同一显式地址。
未在本地真实GPU/Ray集群验证，须HPC确认EngineCore连接现有集群和同步成功。
submit_ray_recovery_chain.sh使用新的ray-address-fix运行目录，保留失败现场，
按afterok提交2步、恢复到4步、50步pilot；不重装环境，不动其他项目任务。
