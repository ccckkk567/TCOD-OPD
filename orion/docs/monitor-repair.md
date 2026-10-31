# 监控和日志路径修复

日志中的SwanLab SettingsError发生在解析project环境字段；站点配置原来导出
SWANLAB_PROJECT纯字符串，而monitor已向SDK显式传project。删除站点导出，并在
训练启动器source凭据后unset此非秘密配置变量，保留API key及显式项目配置。
不改原训练采样和算法，不把JSONL回退当作SwanLab在线成功。

日志另显示checkpoint/log/explorer.log FileNotFoundError，无法仅从该片段确定
路径为何失效。启动器改为在ray.init前创建每job的logs/.../trinity目录，写入
Ray runtime_env；config validator后将配置中的save_dir也指向相同目录，
去除checkpoint/log软链接依赖。原checkpoint路径和保存逻辑不变。

用户要求继续完整链，submit_monitor_recovery_chain.sh提交2步→恢复4步→50步，
全部afterok。旧运行仍在队列时拒绝重复提交，不取消未知任务。保留失败现场，
新run_name以monitor-fix结尾。不重装环境。本地监控回归不是在线SDK集成验证。
