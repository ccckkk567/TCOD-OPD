# Orion HPC 上的 Agentic Vanilla OPD 机制分析：Codex 执行方案

**版本：v1.0 · 2026-10-30**
**交付对象：在本地运行、通过 SSH 提交 Orion SLURM 作业的 Codex。**
**研究底座：`kokolerk/TCOD` 的独立 Vanilla OPD；监控：SwanLab，不使用 W&B。**

> 本文是待执行的工程与实验方案，不是已完成的实验报告。已核对上传的 Orion 指南、TCOD 的指定版本关键代码、SwanLab 与 SLURM 文档；没有登录用户 HPC，没有提交作业，也没有完成 GPU 训练复现。文中的资源、任务数、训练步数是初始配置建议，不是测得的运行耗时或保证的最低资源。

## 阅读与执行导航

初次执行先读§0–3；环境与资产见§4–5；代码实现见§6–7；实验与日志见§8–10；SLURM资源和提交见§11–12；验收、分析与交付见§13–18。所有任务的已完成状态必须来自实际产物，不来自本文。

- [0. 给 Codex 的任务书：先读这一页](#section-0)
- [1. 来源、已知事实与需要现场确认的事项](#section-1)
- [2. 本地控制面、项目目录与最少配置](#section-2)
- [3. 并行实验图：哪些立即排队，哪些等上游](#section-3)
- [4. 阶段A：clone、环境和模型资产](#section-4)
- [5. 阶段B：三个benchmark独立准备](#section-5)
- [6. Codex 代码修改清单：小补丁、可审计、不顺手换算法](#section-6)
- [7. Codex 需要新增的模块及CLI契约](#section-7)
- [8. 训练与模型筛选的具体实验清单](#section-8)
- [9. 保存什么：数据schema与复现链](#section-9)
- [10. Metrics 与 SwanLab 看板](#section-10)
- [11. 初步资源profile：先测，再冻结](#section-11)
- [12. SLURM 脚本模板与依赖提交](#section-12)
- [13. 验收策略：三个等级，研究指标不当工程闸门](#section-13)
- [14. 分析方法、假设区分与结果表述](#section-14)
- [15. 调度异常、降级与恢复预案](#section-15)
- [16. Codex 工作包、执行顺序与完成定义](#section-16)
- [17. 最终交付结构与报告模板](#section-17)
- [18. 给 Codex 的精简执行指令](#section-18)
- [19. 依据与进一步核对入口](#section-19)

---

<a id="section-0"></a>

## 0. 给 Codex 的任务书：先读这一页

### 0.1 要完成什么

在 Orion 上建立一套**能够稳定运行、能够保存行为证据、能够并行排队**的小规模实验链路。先用固定 Teacher 做 Vanilla OPD，分析监督怎样转化为 Student 的行为与能力，再决定是否、以及如何设计 Teacher RL。

第一阶段**不训练新 Teacher、不实现新教学算法、不要求证明 Teacher 不适合 Student**。成功、失败和没有发现预期 gap 都是有效结果。

最终交付包括：修改后的代码与 patch、依赖锁、数据和模型清单、可重用 SLURM 脚本、作业 DAG 与 job IDs、SwanLab 记录、完整轨迹、checkpoint、分析图表、可重放案例和总结报告。

### 0.2 不可改变的研究边界

- 主训练：Student 自主生成并推进环境；Teacher 仅对这些相同 token 序列评分。不得混入 Teacher action、Teacher prefix、SFT 示范、环境奖励优势、课程截断或接管控制器。
- Teacher 的独立 rollout、Student-prefix continuation、2×2 分支都属于 **diagnostic**，不得进入 Student 主训练 buffer。
- 保留三个 benchmark：ALFWorld、WebShop、ScienceWorld。先完成 ALFWorld 闭环；另外两个的安装、下载和 smoke 可同时进行，不等 ALFWorld 训练结束。
- 采用相同协议比较教师和学生：任务、可见信息、动作接口、环境版本、剩余步数预算、生成设置均需明确记录。
- 不以某篇论文分数作为强制验收线；不以 Student 必须崩溃、Teacher 必须显著更强、必须发现动作排序反转作为通过条件。
- 相似工作不是停止探索的理由。贡献评价看明确增量、机制解释和证据，不追求“完全无先例”；也不能为了制造新颖性或预期结论，修改数据、挑选有利 seed 或隐去反例。

### 0.3 HPC 优先级与禁止事项

**队列实际名称是 `testqueue`，不是 `testsqueue`。**上传的 Orion 指南给出：GPU 首选 `testqueue`，单作业上限 **18 小时**；CPU 安装、下载、数据处理首选 `ioqueue`。这些是来源规则，不是本方案估计。[S1]

**Codex 在本地运行。**不得在 Orion 登录节点启动 Codex、Claude Code 等 AI 工具；不得在登录节点安装环境、下载大型模型/数据、跑测试/训练/批量分析。登录节点只做轻量文件管理、传输已准备的项目文件、`sbatch`、`squeue`、`sacct`、小文件状态查询。

环境统一采用 **module 提供的 Python + `venv` + `pip`**。不用 Anaconda/Miniconda，不向 module 的 base Python 安装。指南 §0 强调 venv，后文还列了 conda 示例；本方案选择满足 §0 限制的 venv 路径，不直接照抄论文 README 中的 conda 命令。[S1]

`testqueue` 不满足时，先减少单作业负载或断点分段；确需付费/其他队列时记录理由并取得用户许可，不自动转 `h200p`、`h200p-long` 或 `l40sp`。

### 0.4 执行方式：并行优先，但不做全组合扫描

先准备并行分支，再按产物依赖放行。默认第一波最多 **2 条训练链**，对应 1.7B 和 4B Student；两者不相互依赖。小模型 Teacher 初筛与 30B 参照评估并行。Teacher 的一次诊断结果可在相同 prefix/candidate/协议下复用，不重复计算。

不要让 WebShop 下载失败阻断 ALFWorld；不要让 30B 下载排队阻断 8B 初筛；不要让一个 Student 的训练失败阻断另一个的报告。

**这是实现任务，不是让 Codex 假装现有脚本已经存在。**下文 `opd_pilot.*`、`scripts/*`、`slurm/*` 是要新增并测试的接口契约；TCOD 自带路径会另外标明。先实现、测试，再提交调用它们的任务。

---

<a id="section-1"></a>

## 1. 来源、已知事实与需要现场确认的事项

### 1.1 已知的集群信息：以用户指南为准

| 项目 | 指南提供的信息 | 本方案如何使用 |
|---|---|---|
| 登录 | `orion-login01.acrc.a-star.edu.sg`，需要 A*STAR LAN/VPN | SSH 仅作控制面；不在登录节点执行 AI 工具 |
| 首选 GPU | `testqueue`，6 节点、每节点 8 H200，上限 18h | 默认单节点；作业 time 请求不超过 17:30:00，常规先设更短 |
| 首选 CPU | `ioqueue`，1 节点、56 CPU，上限 72h | CPU 小作业 2–8 核，避免一个安装任务占满节点 |
| 存储 | home 100GB；项目 scratch 示例 2TB；个人 development root 另行确认 | 模型、venv、checkpoint 不放 home；项目额度可能多人共享 |
| Python | `python/3.11.9` 等；`miniforge/24.11.3-2` | 首选 `python/3.11.9` + venv；缺失时现场选择受支持模块 |
| CUDA | 示例 `cuda/12.6.2` | 先核对 wheel、driver、编译工具链，不能把示例当任意 PyTorch wheel 的匹配保证 |
| 容器 | Apptainer 可用 | venv 无法合理解决时才准备受控容器方案，仍通过队列构建 |

指南没有给出：用户账号、项目 account、实际开发根目录、GPU 显存、各节点 RAM、当前 QoS/用户并发上限、计算节点外网/代理、JDK module 名称。这些**不得编造**。

先读取已有本地/远程配置、轻量查询可见 association 和分区。仅缺少无法推断的账号/account/根目录时向用户一次性询问；SwanLab key 不存在就先离线，不阻断准备。

### 1.2 代码基线

```text
REPO_URL=https://github.com/kokolerk/TCOD.git
TCOD_COMMIT=465eef4406ad0cff675b36bd46f37f28b1736ff9
```

已核对的关键内容：[S2–S5]

- `trinity/common/workflows/envs/TCOD/{alfworld,webshop,scienceworld}/OPD_workflow.py`：学生逐轮执行，教师在相同 token 序列评分。
- `trinity/algorithm/advantage_fn/on_policy_distill_advantage.py`：`A = kl_coef * (teacher_logprobs - old_log_probs)`，按响应 mask。
- `trinity/algorithm/algorithm.py`：OPD 默认是 PPO-style policy loss，无额外 reference KL、entropy loss、critic；完整默认值仍需导出 resolved config。
- `pyproject.toml`：`verl==0.7.0`；vLLM 范围 `>=0.10.2,<=0.14.1,!=0.12.0`；不是任意新版依赖都可替换。
- `trinity/utils/monitor.py`：**已有 `SwanlabMonitor`**，不应重新造完整日志框架；但当前实现即使 offline 也会要求环境 API key，需要修补。
- `trinity/cli/launcher.py`：若干模式函数捕获异常后只记录日志。必须确保工作失败时最终非零退出，避免 SLURM `afterok` 误放行。
- `trinity/common/config_validator.py`：GPU 总数会扣除 Student rollout 和 auxiliary Teacher 的 `engine_num × tensor_parallel_size`，剩余分配给 trainer；不能把申请4卡却配置8卡的 YAML 直接启动。

### 1.3 文档内约定

标为 **[来源]** 的是上传指南或已读官方资料支持的行为；标为 **[建议]** 的是本方案的实验/资源默认值；标为 **[待测]** 的是 Codex 必须在作业内验证的兼容性或统计性质。

不需要重新广泛检索论文后才能开始。文献用于解释实验结果，不用于强行决定实验应当出现什么。

---

<a id="section-2"></a>

## 2. 本地控制面、项目目录与最少配置

### 2.1 本地生成的控制包

Codex 在本地建立 `opd-orion-control/`，保存本方案、patch、脚本与配置；使用 rsync 上传**小型代码包**。不要在 Orion 登录节点安装或运行 Codex 代理。

本地需要填写/解析的 `site.local.env`（不得提交含秘密版本）：

```bash
ORION_HOST="<userid>@orion-login01.acrc.a-star.edu.sg"
ACCOUNT="<实际 SLURM account>"
PROJECT_ROOT="/<实际已分配根目录>/agentic-opd"
GPU_PARTITION="testqueue"
CPU_PARTITION="ioqueue"
PY_MODULE="python/3.11.9"
CUDA_MODULE="cuda/12.6.2"
SWANLAB_PROJECT="agentic-opd-gap-pilot"
SWANLAB_WORKSPACE=""
# 这是本项目的配置变量，不假定 SDK 自动识别它。
PILOT_SWANLAB_MODE="auto"
MAX_TRAIN_CHAINS=2
MAX_INFER_ARRAY_CONCURRENCY=2
MAX_DOWNLOAD_CONCURRENCY=2
```

必须拒绝包含 `<...>`、`projxxxx`、`your-development-root` 的正式提交；不能用本地 `$USER` 猜测远程账号。账号凭证走现有 SSH/VPN 机制，不写入文档。

密钥保存于用户已有安全位置，权限600。不要在共享可读目录创建含 key 的文件。不要用 `set -x`、`env` 或完整配置打印泄露 token。SwanLab 和 HF key 不写入 YAML、manifest、git patch 或命令参数。

### 2.2 远程目录

```text
$PROJECT_ROOT/
├── control/                       # 轻量提交脚本、site配置、任务规格
├── sources/TCOD-upstream/          # 固定上游，不在运行期改它
├── releases/<code_hash>/           # 每批作业指向不可变代码快照
├── patches/
├── envs/
│   ├── bootstrap/                 # huggingface_hub、轻量manifest工具
│   ├── train-<lockhash>/
│   ├── alfworld-<lockhash>/
│   ├── webshop-<lockhash>/
│   └── scienceworld-<lockhash>/
├── models/<model_key>/<revision>/
├── datasets/{alfworld,webshop,scienceworld}/
├── manifests/{models,envs,splits,probes,jobs}/
├── runs/<run_id>/{config,student_train,evaluation,diagnostics,metrics,swanlog}/
├── checkpoints/<run_id>/
├── exports/<run_id>/
├── reports/
├── logs/slurm/
├── status/
├── locks/
└── cache/{pip,hf,torch,triton}/
```

**不可变发布规则：**运行/排队中的作业不得指向会继续被 Codex 编辑的源码或 venv。bug修复生成新 `code_hash` 和 patch；不要在两条并行训练中间 `pip install -e` 同一正在变化的源码目录。

每个 job/task 的 `run_id`、SQLite buffer、Ray namespace、临时目录、SwanLab ID 和输出 shard 都独立。不同任务不能写同一个 `alfworld_opd_buffer.db`。

### 2.3 轻量登录节点操作范围

允许：`mkdir`、`rsync`项目文件、`sbatch`、`squeue`、`sacct`、`sinfo`、`scontrol show partition`、`test -f READY`、小文件 `cat/head/tail`。

不允许：模型下载、解压大数据、遍历整个scratch、编译、安装包、Python import-smoke、pytest、checkpoint merge、绘图、环境数据生成。

账号/partition检查可轻量进行，例如：

```bash
sinfo -p testqueue,ioqueue
scontrol show partition testqueue
squeue -u "$USER"
# 若有权限，用来确认本人已有 association；不查询其他用户秘密配置。
sacctmgr -n show assoc user="$USER" format=Account,Partition,QOS
```

`df` 只说明文件系统容量，不等于用户/项目配额；实际额度若不清楚，通过本站允许的 quota 命令或管理员确认。[S1]

---

<a id="section-3"></a>

## 3. 并行实验图：哪些立即排队，哪些等上游

### 3.1 默认 DAG

```text
本地实现控制包 + 上传
    |
    +--> J00 ioqueue: 现场能力/网络/配额检查 + bootstrap venv + clone/pin
           |
           +--> J10 ioqueue: 训练依赖安装 --> J11 testqueue: GPU编译/模型评分smoke
           |
           +--> J12A ioqueue: ALFWorld环境/数据 --> split_A --> J13A环境smoke
           +--> J12W ioqueue: WebShop small smoke；full准备另分支 --> split_W
           +--> J12S ioqueue: ScienceWorld/JAR/数据 --> split_S
           |
           +--> D17 ioqueue: Qwen3-1.7B
           +--> D04 ioqueue: Qwen3-4B
           +--> D08 ioqueue: Qwen3-8B
           +--> D30 ioqueue: 30B-A3B-Instruct-2507（不阻断8B路线）

J11 + 对应数据 + 对应模型
    +--> E0-S17 / E0-S04 / E0-T08：并行评估与学生前缀采集
    +--> E0-T30：独立较强教师参照，晚到不阻断其他结果

E0-S17前缀就绪 --> probe_S17/candidates --> E2/E3(step0) 诊断分支
E0-S04前缀就绪 --> probe_S04/candidates --> E2/E3(step0) 诊断分支

J11 + 数据A + T08 + 对应学生 + 该配对smoke + 初筛可解释
    +--> R17: OPD S17, step0→25 --> R17b: 恢复到50
    |                              \--> checkpoint评估/诊断（并行）
    +--> R04: OPD S04, step0→25 --> R04b: 恢复到50
                                   \--> checkpoint评估/诊断（并行）

每个完成的checkpoint --> CPU导出/检查 --> GPU评估/probe/sharded continuation
每组完整结果 --> ioqueue聚合分析 --> 分支报告
所有分支 afterany --> ioqueue总体报告（允许部分失败）

有明确线索 --> E5 局部更新对照 --> E6 独立任务/seed/环境确认
```

其中0→25→50分段是**在已通过恢复测试后**使用的排队优化；若恢复接口无法可靠保留优化器/scheduler/RNG，则改为一条连续50步作业，不做伪续训。总学习率schedule预设100步；在25步停止只是保存边界，不重启scheduler。不得使用25步终结schedule再把它当50步训练的一半。

### 3.2 并行策略与预算

- 第一波最多两条完整训练链：S17、S04，固定同一合格 Teacher；不是全部师生组合都跑。
- 8B Teacher的可用性按实际任务/学生前缀评估；若几乎全是无效动作且30B明显提供有效行为，生成可追溯决策记录，再改用30B。不能只按某个单seed分数挑“最容易制造gap”的教师。
- S17和S04都先做E0与smoke；有效者可独立进入训练。一个不可用不阻塞另一个。
- 数据安装三分支并行；全量WebShop若受内存/网络阻碍，继续ALFWorld与ScienceWorld，不把small结果混成正式WebShop结果。
- 训练step10/25权重一旦完整发布，可独立提交导出与评估，不必等step50。`afterok`只能依赖作业结束，不会在checkpoint产生时自动触发；由本地控制器轮询小型READY文件并幂等提交，或采用上述25步分段。
- CPU汇总用 `afterany`；真正需要上游资产/完整checkpoint的训练和评估用 `afterok`，并再次检查产物标记。
- 数组并发 `%2` 只限制**该数组**；不同数组不会共享并发额度。提交控制器另外限制本项目总GPU运行量/训练链数，依据实际账号QoS调整，不能把6个测试节点当个人配额。
- 建议软预算：首波运行GPU总量约8–12卡以内、完整训练最多2条、下载最多2个大模型并发。它是研究预算，不是集群授权；若用户配额更低，以配额为准。
- 只做必要串行：依赖安装→GPU编译；真实数据→manifest；采样→固定候选；step25→step50；完整checkpoint→export→评估。无依赖的分支不要人为串行。

### 3.3 作业和产物的幂等规则

`job_key = hash(stage, code_hash, env_lock, input_manifest_hashes, config, model_revision, seed, target_step)`。

提交前查 `manifests/jobs/registry.jsonl`：同key已完成且READY校验通过则复用；同key正在排队/运行则不重复提交。失败重试使用新 `attempt_id`，不覆盖旧日志。瞬态网络/节点错误最多自动重试两次；反复逻辑错误停该分支修复，不循环烧卡。

写产物先到临时目录，同文件系统原子rename后写 `READY.json`/`COMPLETE.json`，包含输入hash、文件清单、global step。**作业exit=0和文件存在都不等于资产完整**；两者一起核验。

---

<a id="section-4"></a>

## 4. 阶段A：clone、环境和模型资产

### 4.1 J00：在ioqueue内做bootstrap与上游clone

本地先上传 `control/`，登录节点只预建日志目录并提交。以下内容应放在 CPU batch job 内，不在登录shell直接执行：

```bash
set -euo pipefail
: "${SLURM_JOB_ID:?This script must run in a SLURM allocation}"
: "${PROJECT_ROOT:?}"
module load python/3.11.9
mkdir -p "$PROJECT_ROOT"/{sources,envs,models,manifests,status,locks,cache/pip,cache/hf,reports}
python -m venv "$PROJECT_ROOT/envs/bootstrap"
source "$PROJECT_ROOT/envs/bootstrap/bin/activate"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
python -m pip install --upgrade pip
python -m pip install huggingface_hub PyYAML
python -m pip freeze > "$PROJECT_ROOT/manifests/bootstrap.lock.txt"

repo="$PROJECT_ROOT/sources/TCOD-upstream"
if [[ ! -d "$repo/.git" ]]; then
  git clone --no-checkout https://github.com/kokolerk/TCOD.git "$repo"
fi
git -C "$repo" fetch origin 465eef4406ad0cff675b36bd46f37f28b1736ff9
git -C "$repo" checkout --detach 465eef4406ad0cff675b36bd46f37f28b1736ff9
test "$(git -C "$repo" rev-parse HEAD)" = 465eef4406ad0cff675b36bd46f37f28b1736ff9
```

幂等完善：已有venv不重复初始化；已有模型不重下；dirty上游不得强制reset，先保留diff。Codex在本地审阅同一commit，生成patch，上传后在CPU作业创建release并应用。检查patch后保存 `git diff --binary`、上游SHA和release tree hash；不向上游push。

J00同时记录：Python版本、module列表、OS/CPU架构、可见JDK、ioqueue与testqueue配置、网络出口测试、存储可用量、账号可见限制。GPU与driver信息在J11记录。若计算节点不能上网，ioqueue准备wheelhouse；不得把下载搬到登录节点规避规定。

### 4.2 J10/J11：训练venv安装与GPU验证

**不要照抄README的conda命令。**使用同一module创建train venv。建议先尝试源码允许且注明较稳定的 `vllm==0.10.2`，保留 `verl==0.7.0`，`flash-attn==2.8.1`。具体Torch、Transformers、Ray和SwanLab版本让兼容性解析后固定，不随每个job升级。[S2]

实现以下流程，而不是声明一组未验证的新版本组合“必定可用”：

1. CPU作业内创建 `train-build/` venv；写初始constraints：`vllm==0.10.2`、`verl==0.7.0`、`transformers>=4.51,<5`，同时遵守源码其他要求。
2. `pip install --dry-run --report ...` 预解析依赖。保存解析报告，不必因为无关optional库缺失而重建全部栈。
3. 安装固定release的 `.[vllm]`、SwanLab、pytest和实际需要的分析包；不要无意义安装所有dev、megatron、数据处理扩展。
4. `pip check`、`pip freeze`，记录Torch CUDA wheel版本和Python ABI。依赖wheel自带CUDA运行时与host driver匹配；GPU扩展编译用的toolkit必须与安装的Torch合理匹配。若 `cuda/12.6.2` 不适用，查可用module并记录，不强行混编。
5. J11在testqueue内安装/验证FlashAttention，限制 `MAX_JOBS` 与CPU核数。无合适wheel、编译受限时，允许使用源码支持的其他attention后端完成smoke，写入环境锁；不要假装仍是相同后端。
6. 在GPU节点验证 `torch.cuda.is_available()`、模型加载、一条generation、给定token的teacher scoring、一次小batch前向/反向。不要只测试import。
7. 发布不可变 `train-<lockhash>/READY.json`；后续不再向该venv `pip install`。若有修复，生成新环境hash。

建议训练初始precision为资源允许的原框架稳定模式，**不要仅为省显存把默认trainer精度静默改为BF16**。优先通过microbatch、checkpointing、序列并行降低占用。Student rollout/Teacher scoring 可按baseline BF16；保存所有实际dtype。出现数值偏差时再做局部精度对照。

GPU smoke的目标是发现系统性错位/异常，不要求两后端logprob逐bit一致。不用单一超小绝对误差阈值卡死pilot。

### 4.3 模型下载任务：独立排队、不重复下载

默认先下载四个明确checkpoint：

| key | repo ID | 默认作用 |
|---|---|---|
| `s17` | `Qwen/Qwen3-1.7B` | 主Student A |
| `s04` | `Qwen/Qwen3-4B` | 并行Student B |
| `t08` | `Qwen/Qwen3-8B` | 低成本Teacher候选 |
| `t30` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 较强Teacher参照/回退 |

不把 `Qwen3-30B-A3B`、`Instruct-2507`、`Thinking-2507`混为一个checkpoint。不自动换成量化版本。Qwen3.5先不加入默认矩阵；若后续需要，以独立兼容性分支接入，不混用不同tokenizer的原生token ID评分。

下载器接口：

```bash
python -m opd_pilot.assets download-model --key s17 --models-manifest ... --output ...
```

实现依据官方 `HfApi.model_info` 与 `snapshot_download`：[S9]

```python
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

# 本片段用于实现下载器；repo_id、root由任务spec传入。
# 首次解析revision写入锁文件，此后的重试必须复用锁中的SHA。
info = HfApi().model_info(repo_id=repo_id, revision="main")
revision = info.sha
local_dir = Path(root) / model_key / revision
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    max_workers=4,
    ignore_patterns=["*.gguf", "*.onnx", "*.tflite"],
)
```

同模型下载由锁保护；写 `_ASSET_READY.json` 前检查config、tokenizer、所有权重index引用的shard存在。不要把一次网络失败后留下的目录当完整模型。使用断点缓存，禁止 `force_download=True` 反复重下。若公开模型不需要认证，不强制索取HF token。

模型资产manifest记录repo、不可变revision、文件清单、字节数、config、tokenizer hash。全部权重hash可在ioqueue低优先级做，不阻塞第一轮GPU smoke；关键小文件与权重shard完整性需先检查。

**Student/Teacher兼容性检查：**不只比词表大小；还要比较token→ID映射、merges/normalizer、special tokens和实际输入模板。Teacher对Student token IDs评分时必须具有相同token语义；不满足就停该配对，不通过重新tokenize后按位置硬拼来“修复”。

### 4.4 存储与下载上限

默认保存完整BF16权重，不保存重复的同checkpoint副本。step0引用原模型；导出后的Student只保留必要步数；大模型不复制到每个run目录。建议首波预算上限600GB（包含环境、模型、checkpoint、数据），这是项目预算，必须受实际剩余共享配额约束。

不删除别人的文件；不自动清除失败run的唯一原始证据。空间不足先停止新下载/新checkpoint扩展、保留当前可恢复状态并报告。home仅放必要小配置，不放模型缓存。

---

<a id="section-5"></a>

## 5. 阶段B：三个benchmark独立准备

### 5.1 模拟器隔离方案

分别创建ALFWorld、WebShop、ScienceWorld venv，避免历史模拟器依赖降级训练栈。最小接入采用 **JSON-RPC/JSONL subprocess worker**：训练/诊断进程使用统一EnvClient，worker由模拟器自己的Python执行。

要新增的接口契约：

```python
class EnvClient:
    def reset(self, task_spec: dict) -> dict: ...
    def step(self, action: str) -> dict: ...
    def available_actions(self) -> dict: ...
    def metadata(self) -> dict: ...
    def close(self) -> None: ...
```

返回显式包含 `observation, raw_reward, score, done, success, info, technical_status`；某字段未知为null，不伪造。保持上游parse/prompt行为，RPC只是进程隔离，不改变Agent策略。

worker协议：请求ID、命令、payload；stdout只传JSON，库日志重定向stderr；设超时、异常码和finally清理。子进程必须用参数列表启动而非拼接shell，显式传入venv和JAVA_HOME，清理训练侧PYTHONPATH/LD_LIBRARY_PATH污染。不要在每一步重启JVM或载入整个商品库。并行session必须隔离可变状态；同session操作串行化，避免共享全局RNG/缓存把一个episode的reset影响到另一个episode。

若某个环境直接装入训练venv已验证无冲突，可将本地adapter作为优化路径；必须与RPC对同一action trace结果一致，再启用。默认优先正确和隔离，不把实现复杂的通用RPC平台当研究任务。

### 5.2 ALFWorld

- 在ioqueue clone官方repo并锁定commit，或锁定pip发行版；不要追随浮动main。候选版本可以从已验证环境复用，但以当前Python3.11兼容smoke为准。
- 检查 `alfworld-download --help`，再使用它实际支持的数据目录选项；不要假设所有版本都有 `--data-dir`。下载路径或缓存均重定向项目存储。
- 保留官方 `json_2.1.1` game目录；为每个task记录game文件与相关资产身份/hash。
- 数据清单从真实game文件枚举；优先复用上游预处理逻辑，但审计是否存在硬编码路径、Teacher成功过滤。
- 默认pilot不依赖 `train_expert.jsonl` 的教师成功筛选。若上游只能提供此清单，先明确其语义，再生成未经Teacher成功筛选的训练任务池。
- 模型输入可使用当前可见observation和admissible actions；planner/expert隐藏输出只准用于独立诊断，不注入主Student/Teacher上下文。
- 成功由环境可验证的任务完成条件决定；同时保存reward、done及可用的won信息。

### 5.3 WebShop

- 官方repo固定commit；先审阅 `setup.sh`/requirements，禁止直接在train venv执行其安装脚本。[S6]
- `small`只用于下载、索引、RPC和动作解析smoke；正式pilot使用完整商品数据/对应索引。small不与full对比，不把small曲线称为正式benchmark性能。
- 官方README说明默认可能只加载1000商品；必须检查运行时 `num_products`、文件路径和索引规模，而不只看文件是否下载完。[S6]
- JDK按锁定的Pyserini等依赖要求选择。Orion指南未指定JDK模块；用 `module avail` 查询，或在CPU作业安装用户空间OpenJDK发行包，记录来源/版本/checksum，不使用conda/sudo。
- 优先一个worker/server加载一次商品与搜索索引，管理多个隔离session；先1–2个并发episode。不要每个Ray actor独立复制整套数据导致内存爆炸。
- TCOD README含很高的WebShop内存提示，但这不是所有实现的必需内存结论；先记录实际MaxRSS、商品数与worker数，按节点实际RAM申请资源，不能假设testqueue提供1TB。
- 数据支持不足时暂停WebShop正式训练，保留small工程smoke状态，不阻断另外两环境。
- 主指标独立记录 `episode_done`、`score`、`success = perfect_score`。购买终止不等于任务成功。

### 5.4 ScienceWorld

- 固定官方repo/包与JAR checksum、Java版本；`task_name, variation_id, simplification`是任务身份的一部分。
- 优先用现有打包JAR，不在登录节点编译。确需构建则ioqueue任务执行。
- 核对TCOD中的compact action呈现已生效：使用动作模板与对象列表，避免把数千完整组合塞进每步prompt。
- 逐步保存原始reward、当前score、最高score、最终score和done；主分析用明确的最终score与成功语义，历史best_score另列。
- 同一(task,variation)的不同simplification不得分散到训练和开发/确认集造成泄漏。
- 不把gold planner、future state或不可见对象列表无意注入模型。环境提供的可见动作信息保持各模型一致。

### 5.5 固定任务协议与pilot清单

每环境生成 `protocol.json`，记录数据版本、goal shuffle/seed、rendering、history、action列表、parser、成功定义、max_env_steps、prompt/response预算。

初始规模：[建议]

| 清单 | 数量建议 | 选择方式 |
|---|---:|---|
| train_pilot | 256 | 从预定训练任务池随机分层，不按Teacher成功筛 |
| analysis_dev | 64 | 与train disjoint，按任务类别分层 |
| confirm_holdout | 32 | 与前两者disjoint；探索过程中不挑case、不调配置 |
| smoke | 每环境2–4 | 仅工程测试，不纳入研究效果 |
| 原benchmark test | 暂不反复使用 | 最后或扩展阶段才运行完整评价 |

若清单过小，比例缩减并记录，不强行重复任务凑数。默认从原训练池中划出pilot train/dev/confirm，官方test留出；已有论文协议另存 `paper_reference_protocol`，不静默将自建split声称为论文原划分。

WebShop必须记录目标顺序、商品全集/子集、搜索索引和session ID。ALFWorld记录seen/unseen来源。ScienceWorld按(task,variation)隔离simplification。三者的测试分数不可混成一个未说明权重的总分。

默认工程起点：ALFWorld 30步、WebShop15步、ScienceWorld30步；这是沿用小规模协议的起点，不是自然任务上限。若ScienceWorld主要被horizon卡住，可新建50/100步诊断protocol；所有模型在新protocol中同条件重测，不在同一run中悄悄更改。

---

<a id="section-6"></a>

## 6. Codex 代码修改清单：小补丁、可审计、不顺手换算法

### 6.1 P00：启动器、退出码、产物完整性

修改 `trinity/cli/launcher.py` 的 bench/explore/train/both 等错误路径：日志后重新抛异常或返回显式失败，并让CLI最终非零退出。检查 `ray.wait` 后的异常是否通过 `ray.get` 真正传播，不能只等待结束就判成功。[S4]

新增包装器在退出前验证：本次实际optimizer update达到目标、最后checkpoint完整、必要记录存在。只有通过才写 `COMPLETE.json`。提前达到安全停止点写 `SEGMENT_COMPLETE.json` 并记录实际步数，不把它叫整个50步完成。

CPU分析的afterany分支可在缺失输入时生成partial report并exit0，但必须列出缺失/失败，不写伪实验数值。启动器打印resolved config前做敏感字段脱敏。

### 6.2 P01：环境与异常指标

修改各 `OPD_workflow.py`/utils，接入EnvClient与统一Recorder，同时尽量保留原来的prompt格式和loss接口。

- `episode_done`、`task_success`、`termination_reason`、`raw_score`分别保存。
- `termination_reason`至少区分：`success`、`env_terminal_failure`、`horizon`、`prompt_overflow`、`generation_truncated`、`model_service_error`、`env_error`、`cancelled`。
- 真正模型生成的非法动作、空动作是行为错误，继续按原环境规则处理；虚构dummy response不作为模型样本训练。
- 若技术失败发生在episode中段，默认主训练不接收该不完整episode，保留失败记录和丢弃计数。不要丢掉正常模型失败轨迹。
- 若超过prompt预算，默认记录技术终止，不让dummy token继续30轮；发生频繁时一次性决定更长预算或固定历史窗口，生成新protocol，重新baseline，不静默逐模型采用不同策略。
- 确认Teacher/Student评分相同token、相同temperature约定，响应切片的 `prompt_length - 1` 与后端语义正确。单看长度相等不够，需用短序列做逐token位置对照。
- 保留每次传入模型的真实chat与token；不要从obs/action反向“差不多”重建训练时prefix。

### 6.3 P02：记录器

新增轻量 `opd_pilot/records.py`，采用per-worker shard，完成后异步/CPU合并。禁止多个worker无锁写同一个JSONL/SQLite文件。

最低需要三类关联ID：`episode_uid`、`turn_uid`、`update_uid`；再保存 `sampling_student_revision, scoring_teacher_revision, update_student_revision`。

每turn记录真实输入、输出、动作、score、token logprobs；每update记录消费的turn IDs、mask、advantage及有效token数。即使只存最终聚合指标，也不能丢掉原始行为证据。

重用prompt通过内容hash去重；长数值数组保存NPZ/Parquet，标量JSONL。后台writer使用有界队列，积压时降低非必要诊断采样/表格上传，不静默丢弃训练样本与关键record。

### 6.4 P03：SwanLab沿用上游接口并修补

现有 `SwanlabMonitor` 强制读取 `SWANLAB_API_KEY`，offline也会raise；本项目至少修改如下：[S3,S7]

1. 规范化 `cloud`→`online`；支持 `online/offline/disabled`，默认由本站配置决定。
2. `offline`不login、不要求key；`online`未配置凭证或联网失败，在有限超时后记录警告并转offline，不进行交互式输入。
3. 不自动 `save=True` 把密钥写到共享环境；使用job运行时注入的环境变量。任何记录、pprint都脱敏。
4. 本地 `metrics/*.jsonl` 始终先落盘。SwanLab是观察层，不是唯一原始证据库。
5. Trainer、Explorer、benchmark各自使用独立run ID和logdir；同一role只允许一个聚合writer；不让每个DDP rank建同名run。
6. 使用实例 `self.logger.log` 而不是依赖可能指向其他run的全局logger；close可重复调用、半初始化失败时也不再抛异常。
7. 相同指标同一step聚合后上传一次。SwanLab对重复同key/step保留先到值，不能用后到的完整统计“覆盖”先前局部值。[S8]
8. role的checkpoint恢复使用已保存run ID与 `resume="allow"`；若锁定SDK不支持相应离线恢复，建立新session并记录 `parent_run_id + global_step_offset`，不覆写旧轨迹。
9. 新增监控掉线/模式切换指标与warning。日志/表格API不支持时回退本地CSV，不因可视化失败终止正常训练。

配置示意（由生成器填实际目录，不把秘密写入YAML）：

```yaml
project: agentic-opd-gap-pilot
group: alfworld-t08-seed42
name: s17-opd-v1
monitor:
  monitor_type: swanlab
  cache_dir: /ABSOLUTE/RUN_DIR/monitor
  monitor_args:
    mode: offline    # auto预检成功后生成online；SDK本身不接收auto
    workspace: null
    logdir: /ABSOLUTE/RUN_DIR/swanlog
    tags: [vanilla-opd, alfworld, s17, t08, pilot]
```

不要在多个role都传入同一个 `id`。项目名、group和tags一致以便比较，role不同要独立writer。SDK实际版本在第一次smoke后锁定。允许安装包依赖中存在wandb，但运行不得 `wandb.init` 或上传W&B；设置 `WANDB_MODE=disabled` 作为防护，而不是依赖它代替配置。

离线同步必须由CPU作业运行，使用与记录时相同的SwanLab版本；优先同步已close的run目录。官方文档提示不同日志版本不可任意混用。[S7]

```bash
# 位于ioqueue作业内，且使用锁定的相同SDK版本。
# RUN_LOG是确实存在的run-*目录，不是任意上级目录。
swanlab sync "$RUN_LOG"
# 若要同步到已存在的原实验，应读取保存的真实ID：
# swanlab sync "$RUN_LOG" --id "$REAL_SWANLAB_ID"
```

计算节点联网正常时在线观察；联网受限就离线记录，段结束后同步。可选近实时CPU转发器只能tail独立metrics流并写入独立mirror run，不能多个进程同时resume同一run或每分钟扫描全目录。没有网络时不要承诺实时云端曲线。

### 6.5 P04：checkpoint、恢复和安全停止

- 保存step0指向原模型的manifest；step10保存模型权重；step25、50保存可恢复训练状态并导出权重。保留当前及上一完整checkpoint，禁止清理正在被评估使用的文件。
- scheduler从一开始按规划总步数100构建；50步作为pilot观察点，25步仅分段点。
- full checkpoint至少包括actor/FSDP shards、optimizer、scheduler、RNG、global update、采样版本映射、数据游标/已消费ID。异步队列若无法完整恢复，应清空不兼容旧buffer、记录resume边界和freshness变化，不称为bitwise恢复。
- 先做2步+恢复2步smoke；至少验证step继续、optimizer state存在、LR不重置、teacher固定、记录不重复。可用极小模型/少量batch对比连续4步与2+2恢复，容许数值差异，不要求轨迹bitwise相同。若恢复机制不稳，用连续50步，不让恢复工程拖死pilot。
- 在time limit前请求安全保存：信号处理只设置flag；所有rank在共同更新边界进行collective checkpoint，不能在任意一个Python signal handler中直接FSDP保存。
- 对SLURM `--signal=B:USR1@600`，batch shell trap写 `STOP_REQUESTED` 文件并通知driver；driver在安全边界保存。10分钟是排队安全余量，不是保存一定能完成的保证。[S10]
- 只有所有文件完整且metadata落盘才写READY；eval job不读正在写入的checkpoint。

### 6.6 P05：独立评估与诊断路径

新增teacher-free评估runner：Student评估不需要为了构造 `teacher_logprobs` 再加载Teacher。复用相同prompt/parser/环境adapter，不偷偷简化历史。

诊断runner应能：加载固定prefix→重放真实环境→加载指定完整候选响应→执行其action→分别由Teacher/Student续写。数据写diagnostics，不触碰训练buffer。

评估、采样、评分分离成独立可分片任务，模型可先后加载；不要求一张卡同时装Teacher和Student。

### 6.7 P06：最小测试，不做漫长大而全CI

在本地做静态和mock测试；真实依赖/模拟器/GPU测试全部通过队列。

必须覆盖：token切片/mask、同checkpoint评分对照、无教师接管、异常不进入loss、WebShop done≠success、ScienceWorld final与best分离、SwanLab offline无key、关键数据无泄漏、两并发run不共写资源、checkpoint恢复与CLI失败退出。

每环境2–4条短smoke即可进入pilot；少数无法重放的状态可以排除分支诊断，但必须记为不可测。不要为了100%全部case可重放而阻止主训练。

**实现优先级：**P00错误传播、P01正确评分/环境指标、P02关键记录、P03最小SwanLab与P04已用到的checkpoint恢复是首条训练需要的部分。高级top-K日志、完整梯度归因、长期在线dashboard、自动大规模超参搜索都不是启动前置条件。单个模拟器兼容问题连续两轮修补仍未解决，标BLOCKED并继续其他环境；不重新写整个框架。

---

<a id="section-7"></a>

## 7. Codex 需要新增的模块及CLI契约

新增包 `opd_pilot`，不要把全部逻辑塞进TCOD workflow。下表为**待实现接口**，不是声称上游已有这些命令。

| 模块/入口 | 必须实现的职责 | 最小输出 |
|---|---|---|
| `opd_pilot.assets` | clone/pin、下载模型、资产检查 | asset manifest + READY |
| `opd_pilot.envs.worker` | 指定venv内环境reset/step/metadata | JSON协议，stderr日志 |
| `opd_pilot.envs.client` | 进程/HTTP代理、超时、session隔离 | 统一状态与异常 |
| `opd_pilot.prepare` | 任务分层与split、协议锁定 | train/dev/confirm JSONL + hashes |
| `opd_pilot.configs` | 从上游OPD生成完整可运行配置 | resolved YAML + GPU mapping |
| `opd_pilot.smoke` | 评分/更新/恢复/评估smoke | gate_report.json |
| `opd_pilot.evaluate` | 原模型/checkpoint独立评估、行为记录 | episodes/turns + summary |
| `opd_pilot.probes` | 固定/动态prefix抽样、候选生成/交叉评分 | immutable probe/candidate bank |
| `opd_pilot.continue` | 固定响应×续写者分支 | branch结果与replay验证 |
| `opd_pilot.export` | 完整checkpoint转可评估HF资产 | export manifest + READY |
| `opd_pilot.update_probe` | 一步teacher评分干预 | 对照更新与行为结果 |
| `opd_pilot.analyze` | 聚合、配对统计、图表、case册 | report.md + CSV/Parquet/PNG |
| `opd_pilot.audit` | 检查数据/版本/样本用途、研究声明边界 | audit.json |
| `scripts/submit_wave.sh` | 轻量sbatch提交、依赖与job registry | DAG + job IDs |
| `scripts/reconcile_jobs.sh` | 轻量状态同步、重试/partial报告触发 | 状态清单，不长期驻留登录 |

统一调用形态：

```bash
# 以下在相应分配节点中执行；每条spec必须已完成占位符替换与验证。
python -m opd_pilot.evaluate --spec /abs/specs/e0_s17_alf.json
python -m opd_pilot.probes --spec /abs/specs/probe_s17.json
python -m opd_pilot.continue --spec /abs/specs/continue_s17_step0.json
python -m opd_pilot.analyze --spec /abs/specs/report_s17.json
```

CLI共同要求：`--spec`；支持 `--validate-only`；输出 `status.json`；明确exit code；接续task/shard只跳过已校验完成项；随机种子由spec+任务身份派生，不能依赖进程启动顺序。

配置生成器应直接调用/复用锁定版本的Config loader/validator，不猜字段名。尚未实现的逻辑必须抛 `NotImplementedError` 或BLOCKED，不返回假分数、假READY。

---

<a id="section-8"></a>

## 8. 训练与模型筛选的具体实验清单

### 8.1 E0：所有候选初筛可并行

ALFWorld默认32个固定dev任务，每模型2条采样。E0-S17、E0-S04、E0-T08均依赖各自模型和ALFWorld环境，不依赖其他模型评估结束。E0-T30先相同16–32任务，可晚到。

WebShop与ScienceWorld环境就绪后，各用16任务×1–2次，先评估一Student和候选Teacher；不强制每模型全benchmark笛卡尔积。

同时从Student真实dev轨迹抽取约16个前缀，Teacher各续写2次，观察教师是否能处理Student状态。Teacher资格不设80%等硬阈值；能产生可解析行为、在部分任务/前缀给出有效进展即可继续。若T08在原点和前缀都被工程/格式错误主导，先修协议；确认其真实能力不足且T30更有效，再以记录的决策切换Teacher。

模型大小与teacher质量是不同因素。首波并行的S17和S04用同一Teacher，便于观察Student规模；不要同时换Student和Teacher却把效果归因于规模。

### 8.2 E1：两条有限规模Vanilla OPD训练链

首波计划：

| run | Student | Teacher | 环境 | seed | step |
|---|---|---|---|---:|---:|
| `R17` | Qwen3-1.7B | 合格的T08，必要时统一改T30 | ALFWorld | 42 | 50 |
| `R04` | Qwen3-4B | 与R17相同 | ALFWorld | 42 | 50 |

两条可同时排队，不要求S17跑完才试4B。若S17完全被格式错误主导，也保留其E0结果，只让4B训练；这不是研究失败。

在已有OPD YAML上固定：

```yaml
mode: both
algorithm:
  algorithm_type: on_policy_distill
  advantage_fn: multi_turn_opd
  advantage_fn_args:
    kl_coef: 1.0
  policy_loss_fn: ppo
  kl_loss_fn: none
  kl_penalty_fn: none
  entropy_loss_fn: none
  repeat_times: 1
  optimizer:
    lr: 1.0e-6
```

原配置采用 `staleness_control: max_staleness=2`；先保留并记录真实版本差，出现疑问时单做较严格同步对照，不将整个实验偷换为别的采样规则。

建议有效batch先复用上游16/64配置并核对实际episode/turn/token数；H200内存不足先调microbatch和梯度累积，不先减总有效batch；确需改变则记录。不得把名义train_batch_size=64直接解释成64条完整轨迹。

初始训练采样温度1.0，top-p/top-k用不截断支持的明确设置；评估温度0.4作为独立固定protocol。Teacher评分必须对应学生训练采样所使用的temperature/归一化语义；若SDK返回未经temperature的logprob，统一标注并在公式中使用实际语义，不隐瞒。

prompt/response预算从上游合理起点设置（例如prompt 10240、response512），先统计溢出。若native thinking或格式导致普遍截断，一次性选择合适预算/模式、重做E0；不能只给Teacher更长预算却称完全控制。Teacher可有更大硬件context上限，但实际可见内容/有效token预算在主比较中一致。

第一轮优先小Student全参数训练；若资源只能LoRA，单独标为 `update_space=lora`，不把LoRA限制解释成模型的绝对能力上限。不进行额外cold-start训练；若必须做格式SFT，用同样SFT后的权重作为所有分支共同S0并完整记录，不与原始S0结果混用。

### 8.3 E2：固定probe、动态probe和候选生成

每Student初始从约24个dev任务选48–72个固定prefix：约2/3预定义随机/分层选择，约1/3探索性高分歧/循环/失败前状态。主统计用随机子池，探索性子池单独报告。

另外建立 `common_probe_pool`：从两Student的随机子池取去重子集，用于跨size在相同历史上的比较。各Student自有池用于描述它真正访问的状态；不能把不同状态上的差异说成纯能力差异。

Teacher/Student各采当前轮完整response，初始每prefix各2个；解析并语义规范化action。完整response保留，不能把长短不同的response总logprob直接当action概率排名。交叉打分时，在相同prefix+相同candidate上比较两模型；同时报总NLL和token均值，名字不要错写成Teacher entropy。

固定candidate bank发布后，step0/25/50对同一candidate评分和续写，不每个checkpoint重新选“最有意思”的动作。重复candidate记录频率，不强迫Teacher和Student一定生成不同动作。

### 8.4 E3：2×2 continuation，不把接力成功当成学习证明

固定Student访问的历史 `h`、环境状态、已使用步数。Teacher和Student各提出一个完整候选响应 `r_T/r_S`，执行其中action，再让两种执行者各自继续：

| 已固定的当前响应 | Teacher继续 | Student继续 |
|---|---|---|
| r_T | Q_T(h,r_T) | Q_S(h,r_T) |
| r_S | Q_T(h,r_S) | Q_S(h,r_S) |

**干预单位是完整响应，不只是action字符串。**它包含的解释也会进入历史。只有后续另做固定解释/去解释对照，才能进一步声称差异来自纯环境动作。

环境恢复规则：优先原生可靠snapshot；否则reset后重放完整action前缀，核对观测、合法动作和关键状态。比较时恢复同一个真实状态，再注入同一candidate及新环境观测。不得只把历史文本发给模型而不还原环境。

重放不一致的状态标为 `replay_mismatch`，不参与同状态因果比较，但仍可保留普通行为日志。不可重放率本身需报告。不能要求全部环境所有状态100%可重放才能开始。

剩余预算 = 原episode最大步数 − 已用步数 − candidate执行步数；Teacher/Student相同。若candidate已终止任务，记录 `terminal_candidate=true`，其结果可用于总体行为评价，但不算后续承接能力的证据。

初始只做16个prefix，每prefix两个固定候选，每候选两种continuation policy，各2次：共128条continuation。可以拆为4–8个GPU shard，每个shard一次加载一个executor模型连续处理多条，避免每条轨迹重新加载模型。候选生成先于分支执行；相同Teacher分支在相同prefix/candidate/protocol下缓存复用。

指标：

\[
\Delta_T(h)=Q_T(h,r_T)-Q_T(h,r_S),\qquad
\Delta_S(h)=Q_S(h,r_T)-Q_S(h,r_S).
\]

观察 `Delta_T>0, Delta_S<0` 是否出现；探索性出现不等于已确认。两次采样的点估计只能发现线索，有意思的状态用新continuation seeds补到4–8次。记录两者全败、同action、terminal candidate、技术失败等分母，不只报反转比例。

不能用“Teacher成功、Student失败”单独证明Teacher策略不适合Student；需比较不同candidate在相同续写者下的相对结果。仍然要承认：当前Student承接不了，不代表训练后永远学不会。

### 8.5 E4：固定probe上的选择—承接动态

对step0/25/50 Student重复固定probe评分、当前动作采样和同candidate continuation。step10至少做轻量评分/动作分布；不要求全部深度分支重跑。

分别测量：

- **选择**：Student自发生成的语义动作频率、Teacher候选/动作的条件似然变化；单一生成文本的概率不等于所有同义动作的总概率。
- **承接**：给定同一完整candidate后，Student成功率/score、剩余步数、错误类别。
- **访问状态变化**：在每checkpoint另外取少量动态prefix，和固定池分开统计。

“选择Teacher路线增加，承接能力未增”只是候选机制；还要排除训练太短、candidate不代表路线、prompt差异和选择性抽样。没有这种模式也正常，保留学习同步改善的反例。

### 8.6 E5：按需要做局部更新干预，不自动铺开

当E1–E4出现可复查线索时，从同一Student checkpoint（优先25）复制model+optimizer+scheduler+RNG，两分支使用**刚由该Student采样的同一批数据**，分别用T08/T30或预先定义的两种Teacher评分条件，做一个实际optimizer update。

两分支相同学习率、有效token、batch顺序、loss聚合/clip；保存全部输入和actual update，测固定probe概率/动作/少量continuation。可加同样流程不更新的对照。

若继续5–10步，每支必须重新采自己的on-policy数据；不能重复固定旧数据后仍声称在做同一在线流程。单次评分干预只能支持局部更新的影响，不能凭两个大小不同Teacher将效应归为单一机制。

如调整Teacher历史、去除学生旧解释或仅替换某类状态监督，必须使用无新增隐藏事实的变换；匹配advantage尺度/有效token或至少显式报告差异。随机同量位置对照可用于排除单纯减小更新量。无需第一轮同时做所有消融。

### 8.7 E6：确认，不是重新挑最有利的setting

优先把最稳定的一条现象移到confirm_holdout，用新rollout seeds复查；再补第二训练seed；然后按收益选择第二环境/模型。保留原来探索与确认标记。第二环境安装已并行完成，不需要此时再从零搭建。

将50步延长到100只是检查时间尺度，不用“50步没学会”证明不可学习。第二轮Teacher RL objective设计属于后续研究，不在本执行任务中自动启动。

---

<a id="section-9"></a>

## 9. 保存什么：数据schema与复现链

### 9.1 Run manifest

必需字段：`schema_version, run_id, attempt_id, code_hash, upstream_commit, patch_hash, env_lock_hash, protocol_hash, split_hash, student_repo/revision, teacher_repo/revision, tokenizer_hashes, model_mode, dtype, sampling_config, loss_config, seeds, slurm_job_id/array_id, resources_requested, resources_observed, purpose, update_space, parent_run, checkpoint_parent`。

保存完整resolved config（脱敏），以及源码实际用到的model参数和generation_config。模型名相同不等于revision相同。

### 9.2 Episode / Turn / Update

| 记录 | 最低字段 |
|---|---|
| Episode | task身份、env版本、seed、purpose、是否进入Student训练、episode UID、开始/结束、总步数、所有score/reward、成功/终止原因、technical flags |
| Turn | parent episode、turn index、精确messages或hash引用、prompt/response token IDs、prompt_length、采样配置、原始response、解析action、validity、observation、可见动作、环境info白名单 |
| 信号 | sampled old logprob、Teacher logprob、实际trainer重算old/current logprob（若框架使用）、response mask、teacher_valid mask、实际advantage、segment mask、采样与评分版本 |
| Update | consumption IDs、重复使用次数、actual optimizer step、有效turn/token、实际loss/clip/ratio/grad、丢弃样本及原因、checkpoint引用 |
| Branch | parent prefix UID/hash、候选完整响应与来源、action、executor revision、预算、replay check、seed、完整执行结果 |

同名 `old_log_probs` 在不同后端可能来自采样或重算，必须标明来源，不统一改名覆盖。采样不同行为温度的logprob不得直接相减后当作相同分布的KL。

### 9.3 数据用途标记

```json
{
  "purpose": "diagnostic",
  "used_for_student_update": false,
  "run_id": "alf-s17-t08-seed42",
  "parent_prefix_id": "...",
  "student_checkpoint": "step_025",
  "teacher_revision": "immutable-hf-sha",
  "protocol_hash": "..."
}
```

训练buffer写入处assert `purpose == student_train`。`diagnostic`数据可以未来研究使用，但本pilot不得自动回流。

### 9.4 存储策略

每worker shard独立，周期flush；压缩JSONL保存文本，NPZ/Parquet保存数值。只有小清单用JSON；不把整个模型logits、hidden states、梯度存下来。

固定probe可以保存少量top-K及tail质量（若后端支持），不作为主训练loss改变。不要为全量logits计算把pilot拖成一个新infra项目。

完整checkpoint保留25/50和最近恢复点；10保存weights即可。已被分析引用的checkpoint不得自动删除。轨迹/信号先保留全部pilot，评估隐私后只向SwanLab上传小样本摘要，不上传模型权重或大量完整数据集。

---

<a id="section-10"></a>

## 10. Metrics 与 SwanLab 看板

### 10.1 必须上传的标量

| 分组 | 建议key | 解释 |
|---|---|---|
| 训练 | `train/policy_loss`, `train/grad_norm`, `train/clip_fraction`, `train/ratio_mean`, `train/effective_tokens`, `train/updates` | 缺少某个值则null/不报，不伪造0；update_norm可后续抽样计算 |
| 监督 | `signal/logratio_mean`, `signal/logratio_p10`, `signal/logratio_p90`, `signal/positive_fraction` | 统计实际Teacher−Student信号，不当作任务优势 |
| 分段 | `signal_action/logratio_mean`, `signal_reasoning/logratio_mean`, `signal_format/logratio_mean`, `signal_stop/logratio_mean` | 特殊token、格式与决策拆开；信号占比不是梯度贡献占比 |
| 任务 | `eval/success_rate`, `eval/score_final_mean`, `eval/score_best_mean`, `eval/steps_mean`, `eval/success_steps_mean` | 各env/split分别run或明确key前缀，episode等权 |
| 质量 | `quality/parse_error_rate`, `quality/invalid_action_rate`, `quality/prompt_overflow_rate`, `quality/generation_truncation_rate`, `quality/service_error_rate`, `quality/replay_mismatch_rate` | 技术异常与行为错误分开 |
| 行为 | `behavior/action_tokens_mean`, `behavior/reasoning_tokens_mean`, `behavior/repeat_action_rate`, `behavior/no_progress_rate` | 无进展判据必须有环境特定定义，不能只靠字符串重复 |
| 新鲜度 | `freshness/model_version_lag_mean/max`, `freshness/teacher_version_count` | 本pilotTeacher固定应为1版本；Student可有限滞后 |
| 资源 | `cost/generated_tokens`, `cost/scored_tokens`, `cost/env_steps`, `cost/gpu_hours_allocated`, `cost/wall_seconds`, `cost/max_rss_gb` | 训练、诊断、评估分开计费/计数 |
| probe | `probe/action_agreement`, `probe/candidate_nll_delta`, `probe/adoption_frequency` | candidate和语义动作分母明确 |
| continuation | `branch/mean_delta_teacher`, `branch/mean_delta_student`, `branch/reversal_fraction_exploratory`, `branch/n_valid_prefixes` | 反转点估计不作为显著机制结论 |

精确episode数、task数、token数随聚合结果一起上传，避免曲线均值没有分母。训练的turn均值不替代任务均值；长episode不能因turn多被自动赋更高评估权重。

### 10.2 KL / entropy 的命名边界

- `mean(log S_old - log T)`记录为sampled log-ratio统计。只有在实际采样分布、temperature及支持条件对应时才称KL Monte Carlo估计。
- 有限样本估计可为负，保留原值；不能把它截成零，也不能因出现负数判断实现一定错。
- 在Teacher样本上计算Student NLL是交叉评分，不是Student自身entropy。
- 按token平均和按轨迹求和同时保留；轨迹总KL随长度变化，不直接解释为能力退化。
- 对完整candidate的总logprob与长度归一化分数分别命名。语义动作概率需通过动作采样频率或合法动作空间的明确计算，不等同于一条固定文本的概率。
- 主线不需要精确full-vocab entropy；已有后端便宜可得则记录，top-K近似注明tail/覆盖，不为这个指标阻塞执行。

### 10.3 看板组织

项目：`agentic-opd-gap-pilot`。Group包含env、teacher和protocol短hash；实验name包含student、seed、stage、code短hash。Trainer/Explorer/Evaluation独立run，metadata中写共同 `logical_run_id`，不要并发写相同run ID。

步数：Trainer用真实global update；Explorer用采样batch序号并另报对应Student版本；评估run用checkpoint_step；continuation用batch/shard进度，聚合后另建summary run。不要让不同含义的step混在同一曲线上。

云端记录标量和少量表/图；原始证据以本地文件为准。对敏感task/用户名/绝对路径做脱敏，公开项目默认false/私有（以账号支持为准）。

---

<a id="section-11"></a>

## 11. 初步资源profile：先测，再冻结

以下是**SLURM申请上限建议，不是运行时间预测**。节点内存、GPU显存和账号上限未在指南给出，需要现场检查；超过可申请资源时缩小并发，不改用未经授权队列。

| profile | partition | GPU | CPU | RAM建议 | time请求 |
|---|---|---:|---:|---:|---|
| site_probe | ioqueue | 0 | 2 | 8G | 00:30:00 |
| clone/bootstrap | ioqueue | 0 | 4 | 16G | 02:00:00 |
| model_download | ioqueue | 0 | 4 | 24G | 06:00:00 |
| install_train | ioqueue | 0 | 8 | 48G | 04:00:00 |
| install_env/index | ioqueue | 0 | 8 | 64G起测 | 08:00:00 |
| gpu_smoke | testqueue | 1 | 8 | 48G | 01:30:00 |
| eval/continue ≤8B | testqueue | 1 | 6–8 | 48–64G | 03:00:00 |
| eval/continue 30B | testqueue | 2，实测可降1 | 8–12 | 96G | 04:00:00 |
| train4 | testqueue | 4 | 16 | 128G | 12:00:00 |
| train8 | testqueue | 8 | 24–32 | 256G | 16:00:00 |
| aggregate/export | ioqueue | 0 | 4–8 | 32–96G按模型 | 04:00:00 |

WebShop实际RAM与共享server设计有关；full数据如果64G不够，先单worker实测并申请合理RAM，不盲目复制数据，也不声称需要固定1TB。CPU分析不申请GPU；GPU作业不要用多余的卡只为了获得CPU内存。

### 11.1 4卡Student训练的明确分配

锁定版本按总GPU扣除explorer角色计算trainer资源。[S5]

```text
总计4卡 = Student trainer 2 + Student rollout(1 engine × TP1) 1 + Teacher scoring(1 engine × TP1) 1
```

可用于S17/S04 + T08的初始profile。单Teacher评分服务能否承受长度/并发，必须在H200实测。Trainer的FSDP/sequence-parallel度必须整除其2卡，不能沿用需要4trainer卡的配置。

```yaml
cluster:
  node_num: 1
  gpu_per_node: 4
explorer:
  runner_per_model: 2
  rollout_model:
    engine_num: 1
    tensor_parallel_size: 1
  auxiliary_models:
    - engine_num: 1
      tensor_parallel_size: 1
```

上述只是资源override片段；模型、路径、dtype、预算等由完整生成器补齐并通过validator。30B不默认只需激活参数对应显存：若TP1/4卡方案不稳，独立使用8卡profile（4 trainer + 2 rollout + 2 teacher），不要挤占另一个job的可见GPU。

### 11.2 SLURM可见设备与Ray隔离

保留SLURM设置的 `CUDA_VISIBLE_DEVICES`；不硬编码物理0–7、不手工覆盖为所有GPU。`cluster.gpu_per_node`取本job真实申请数，不取整节点物理卡数。

每job独立Ray cluster。明确 `cluster.ray_address`，禁止 `address=auto`误连别的实验；每job独立Ray namespace和短的node-local temp目录。锁定Ray文档指出同节点并行cluster需要处理端口冲突。[S11]

实现 `RayRuntime`：在分配节点内选并锁定本job端口块，检查当前Ray版本CLI支持的head、node-manager、object-manager、metrics/agent及worker端口参数；记录映射。file lock只防本项目冲突，还需端口bind预检与冲突重试；不要仅用JobID取模就声称绝不会冲突。

建议Ray head及trainer均运行在同一个 `srun --ntasks=1` job step内，由Ray自行分配本job GPU；不要外层启动8个srun进程，每个都创建完整Ray集群。

不得 `ray stop --force`、`pkill -u $USER` 或清空全局 `/tmp/ray`。清理只针对当前job登记的PID/process group及其临时目录。Dashboard默认不开；本地调试端口使用受控SSH tunnel，不能在登录节点启动服务。

---

<a id="section-12"></a>

## 12. SLURM 脚本模板与依赖提交

本节给出控制面骨架。Codex应将它们变成文件并做 `bash -n` 和mock测试；`run_stage.sh`及各Python CLI按第7节实现后才能正式提交。

### 12.1 通用batch wrapper：`control/slurm/job.sbatch`

调度资源通过 `sbatch` 命令行传入，避免在 `#SBATCH` 中写不会展开的 `$PROJECT_ROOT`。SLURM只保存batch脚本本身，不会自动冻结外部代码/spec；所以任务spec必须指向不可变release。[S10]

```bash
#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Must execute as a batch job}"
: "${PROJECT_ROOT:?PROJECT_ROOT is missing}"
if [[ $# -ne 2 ]]; then
  echo "usage: job.sbatch STAGE ABSOLUTE_SPEC_PATH" >&2
  exit 64
fi
STAGE="$1"
SPEC="$2"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPD_JOB_ATTEMPT="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export OPD_STATUS_DIR="$PROJECT_ROOT/status/jobs/$OPD_JOB_ATTEMPT"
mkdir -p "$OPD_STATUS_DIR"
export OPD_STOP_FILE="$OPD_STATUS_DIR/STOP_REQUESTED"

request_stop() {
  date -u +%FT%TZ > "$OPD_STOP_FILE"
}
trap request_stop USR1

# 这里只运行一个driver；Ray/FSDP分布式由driver管理。
# run_stage.sh须实现资源检查、密钥读取/脱敏、module/venv加载、
# 专属Ray启动清理、状态记录和输出校验。
srun --ntasks=1 --kill-on-bad-exit=1 \
  bash "$PROJECT_ROOT/control/scripts/run_stage.sh" "$STAGE" "$SPEC" &
child=$!
forward_term() {
  request_stop
  kill -TERM "$child" 2>/dev/null || true
}
trap forward_term TERM INT

# USR1可打断wait，不能误判child已完成。
while true; do
  set +e
  wait "$child"
  rc=$?
  set -e
  if kill -0 "$child" 2>/dev/null; then
    continue
  fi
  break
done
exit "$rc"
```

`run_stage.sh` 不允许用 `|| true` 吞掉训练异常；finally清理可以best-effort。超过目标walltime安全停止但未达target step时返回明确RESUMABLE状态和非零码（例如75），不要触发下游“已完成25步”的afterok。控制器据最新完整checkpoint另提交恢复任务，并重建受影响依赖。

CPU任务不必提前10分钟停止，wrapper收到信号可忽略已进入原子写入完成区的情况；具体stage决定是否poll STOP。不要在下载器写半个文件时创建READY。

### 12.2 轻量提交函数：`control/scripts/submit_lib.sh`

该脚本只构建sbatch参数和保存小型job registry，可在登录节点执行；不加载Torch、不读大数据、不绘图。没有写入SwanLab/HF密钥。

```bash
#!/bin/bash
set -euo pipefail
: "${ACCOUNT:?}"
: "${PROJECT_ROOT:?}"
mkdir -p "$PROJECT_ROOT/logs/slurm" "$PROJECT_ROOT/manifests/jobs"

submit_stage() {
  local stage="$1" spec="$2" partition="$3" gpus="$4"
  local cpus="$5" mem="$6" wall="$7" dependency="${8:-}" array="${9:-}"
  local out jid
  local -a args=(
    --parsable --account="$ACCOUNT" --partition="$partition"
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem"
    --time="$wall" --signal=B:USR1@600 --job-name="opd_${stage}"
    --export=ALL --chdir="$PROJECT_ROOT"
  )
  if (( gpus > 0 )); then args+=(--gres="gpu:$gpus"); fi
  if [[ -n "$dependency" ]]; then args+=(--dependency="$dependency"); fi
  if [[ -n "$array" ]]; then
    args+=(--array="$array"
      --output="$PROJECT_ROOT/logs/slurm/%x-%A_%a.out"
      --error="$PROJECT_ROOT/logs/slurm/%x-%A_%a.err")
  else
    args+=(--output="$PROJECT_ROOT/logs/slurm/%x-%j.out"
      --error="$PROJECT_ROOT/logs/slurm/%x-%j.err")
  fi
  out=$(sbatch "${args[@]}" "$PROJECT_ROOT/control/slurm/job.sbatch" "$stage" "$spec")
  jid="${out%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Bad job id: $out" >&2; return 1; }
  # 正式实现增加提交去重和registry写锁；这里是最小骨架。
  printf '%s\t%s\t%s\t%s\t%s\n' "$jid" "$stage" "$spec" "$dependency" "$array" \
    >> "$PROJECT_ROOT/manifests/jobs/submitted.tsv"
  printf '%s\n' "$jid"
}
```

传递PROJECT_ROOT、ACCOUNT前明确 `export`，禁止含未替换占位符的路径进入函数。在完整实现里，增加资源profile白名单、walltime≤partition上限、job_key去重、失败后依赖无效处理。`--kill-on-invalid-dep=yes`可在本站支持且行为已确认时启用，未支持时由reconcile报告并取消本项目失效依赖；不要误取消别人的job。

### 12.3 第一次资源准备的并行提交示例

以下spec由本地Codex预生成并上传；bootstrap内容不依赖尚未构建的训练venv。大数据处理仍在作业内。

```bash
# 登录节点允许的轻量提交脚本；source仅加载无秘密的站点变量。
source "$PROJECT_ROOT/control/site.runtime.env"
export PROJECT_ROOT ACCOUNT
source "$PROJECT_ROOT/control/scripts/submit_lib.sh"
S="$PROJECT_ROOT/control/specs"

J00=$(submit_stage bootstrap "$S/bootstrap.json" ioqueue 0 4 16G 02:00:00)
J10=$(submit_stage train_env "$S/train_env.json" ioqueue 0 8 48G 04:00:00 "afterok:$J00")
J11=$(submit_stage gpu_compile "$S/gpu_compile.json" testqueue 1 8 48G 01:30:00 "afterok:$J10")

JA=$(submit_stage env_alf "$S/env_alf.json" ioqueue 0 8 48G 06:00:00 "afterok:$J00")
JW=$(submit_stage env_web "$S/env_web.json" ioqueue 0 8 64G 08:00:00 "afterok:$J00")
JS=$(submit_stage env_sci "$S/env_sci.json" ioqueue 0 8 48G 06:00:00 "afterok:$J00")

# 分开提交便于只依赖某个模型，而不是等四个模型全部下载完。
D17=$(submit_stage model "$S/model_s17.json" ioqueue 0 4 24G 06:00:00 "afterok:$J00")
D04=$(submit_stage model "$S/model_s04.json" ioqueue 0 4 24G 06:00:00 "afterok:$J00")
# 限制大下载同时最多两项；CPU安装/其他环境不需要跟着串行。
D08=$(submit_stage model "$S/model_t08.json" ioqueue 0 4 24G 06:00:00 "afterok:$J00,afterany:$D17")
D30=$(submit_stage model "$S/model_t30.json" ioqueue 0 4 24G 06:00:00 "afterok:$J00,afterany:$D04")

E17=$(submit_stage eval "$S/e0_s17_alf.json" testqueue 1 8 64G 03:00:00 "afterok:$J11:$JA:$D17")
E04=$(submit_stage eval "$S/e0_s04_alf.json" testqueue 1 8 64G 03:00:00 "afterok:$J11:$JA:$D04")
E08=$(submit_stage eval "$S/e0_t08_alf.json" testqueue 1 8 64G 03:00:00 "afterok:$J11:$JA:$D08")
E30=$(submit_stage eval "$S/e0_t30_alf.json" testqueue 2 12 96G 04:00:00 "afterok:$J11:$JA:$D30")

# 这份报告即使某候选失败也应给出可用结果，不阻塞其他准备分支。
R0=$(submit_stage report "$S/e0_report.json" ioqueue 0 4 16G 01:00:00 \
  "afterany:$E17:$E04:$E08:$E30")
```

以上下载依赖是简单限流办法，不代表T08在科学上依赖S17下载；正式控制器可使用带限流的独立队列/array并依赖对应array element，减少一个下载失败阻断另一个下载。不要因为非必要限流依赖失败，停止所有模型资产准备。

E0总体报告可等待全部候选，但**T08训练决策不等待E30**。T08配对的小结可独立afterany(E17,E08)生成。若需要早评ScienceWorld/WebShop，以同样结构分别依赖JS/JW，不依赖JA训练。

### 12.4 训练分段与评估并行示例

在对应pair smoke、数据清单和GO/WARN记录完成后：

```bash
# G17/G04是各配对smoke与决策作业的job id；task spec已冻结。
R17A=$(submit_stage train "$S/train_s17_0_25.json" testqueue 4 16 128G 12:00:00 "afterok:$G17")
R04A=$(submit_stage train "$S/train_s04_0_25.json" testqueue 4 16 128G 12:00:00 "afterok:$G04")

R17B=$(submit_stage train "$S/train_s17_25_50.json" testqueue 4 16 128G 12:00:00 "afterok:$R17A")
R04B=$(submit_stage train "$S/train_s04_25_50.json" testqueue 4 16 128G 12:00:00 "afterok:$R04A")

X17=$(submit_stage export "$S/export_s17_25.json" ioqueue 0 8 64G 02:00:00 "afterok:$R17A")
X04=$(submit_stage export "$S/export_s04_25.json" ioqueue 0 8 96G 02:00:00 "afterok:$R04A")
V17=$(submit_stage eval "$S/eval_s17_25.json" testqueue 1 8 64G 03:00:00 "afterok:$X17")
V04=$(submit_stage eval "$S/eval_s04_25.json" testqueue 1 8 64G 03:00:00 "afterok:$X04")

# CAND17是已发布candidate bank对应的任务job id；四个shard不拆成每个episode一个job。
B17=$(submit_stage continuation "$S/branch_s17_25_shards.json" testqueue 1 8 64G 03:00:00 \
  "afterok:$X17:$CAND17" "0-3%2")
A17=$(submit_stage report "$S/report_s17_25.json" ioqueue 0 4 24G 02:00:00 \
  "afterany:$V17:$B17")
```

`CAND17`必须已经完成具体候选生成，不能依赖未生成的占位job。Teacher分支可另外一次性计算并复用；30B分支使用对应2卡profile，不塞进1卡数组。不同资源需求分成不同array。

失败重试后旧 `afterok` 不会神奇跟随新job；reconcile需要用新job ID重新提交/更新下游。对自己的无效pending依赖做明确标记，不无限等待。总报告依赖所有计划leaf jobs的afterany，不只是trainer结束。

### 12.5 轻量作业观察

```bash
squeue -u "$USER" -o '%.18i %.22j %.10P %.8T %.10M %.6D %R'
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS -P
scontrol show job "$JOB_ID"
```

Codex本地按合理间隔查看，不每秒轮询SLURM。远程只读小型status/READY；大日志分析作为ioqueue任务。排队不是失败，不要频繁取消重提交以“刷优先级”。[S10]

---

<a id="section-13"></a>

## 13. 验收策略：三个等级，研究指标不当工程闸门

### 13.1 STOP：本分支必须先修

- token语义不一致或评分切片系统性错位；错误更新prompt/环境token。
- 实际混入Teacher执行数据，却宣称Vanilla OPD。
- train/dev/confirm/test泄漏，或读取不可见planner信息进入主模型输入。
- NaN、无法更新、checkpoint损坏、重复/覆盖导致关键数据不可追溯。
- 超长dummy被当正常action训练；持续无法区分技术异常和任务失败。
- 使用未分配GPU、违规登录节点工作、提交到未经许可付费队列。

停的是受影响分支，不是所有并行任务。明确区分STOP原因与模型能力差。

### 13.2 WARN：记录、降低结论强度，但继续

- Student初始成功率低、Teacher提升不大、50步没有显著改善。
- 个别environment任务重放不一致；仅将其排除同状态干预，其余照常。
- 少数超时/网络失败；记录全样本与剔除技术失败后的结果。
- SwanLab离线；本地原始记录完整就继续。
- 某些可选指标（精确entropy、全梯度、hidden-state）未实现。
- 不同后端的微小数值偏差；只要量级记录并不淹没主差异，无需bitwise一致。

不设Teacher≥80%、Student≥10%、必须提升5pp、必须出现反转或p<0.05的硬线。一次seed可探索，确认再加seed。

### 13.3 GO：足以开展机制发现

至少有可解释的环境交互、真实Student更新、Teacher评分、独立eval、可追踪日志和一部分可靠重放状态。工程底线满足后尽快进入E1–E4，不把环境搭建无限延长成全部工作。

`gate_report.json`字段：stage、GO/WARN/STOP、evidence paths、affected branch、next action、未满足项、实际资源/步数。它是状态记录，不是要求Reviewer批准每一个微小选择。

---

<a id="section-14"></a>

## 14. 分析方法、假设区分与结果表述

### 14.1 先描述事实，再解释

对每条观察写：

```text
Observation：具体状态/行为/指标发生了什么。
Hypothesis：哪种机制可能解释。
Alternative：还有哪些解释（工程、预算、教师错误、覆盖、优化等）。
Intervention：改变了哪个变量，哪些变量保持一致。
Result：结果和不确定性，包含反例。
Scope：结论适用于哪些模型/状态/训练阶段。
```

不得把LLM对轨迹的分类当真实因果标签。LLM可辅助索引案例，最终通过实际执行/控制对照支撑。没有证据时标“未区分”，不为论文故事补原因。

### 14.2 五类主要候选机制

| 候选机制 | 需要的证据 | 会削弱该解释的结果 |
|---|---|---|
| Teacher在Student prefix上不可靠 | Teacher自己续写也困难；偏好的candidate实际无效；条件修正后改善 | Teacher续写可靠，Student仍在相同candidate后失败 |
| Student承接价值与Teacher不同 | 同状态候选相对排序随executor改变，重复采样仍有信号 | 只因候选直接终结任务、预算/解释文本差异或随机失败 |
| 选择与执行学习不同步 | 固定probe上采纳频率增加，条件成功率未同步改善 | 两者同步增长；只是在不同新状态上的分布变化 |
| 监督覆盖/作用位置不足 | 行为未明显改变，或Teacher目标几乎不在Student采样支持中 | 相关行为已频繁出现且概率更新明确 |
| 有限更新/优化或数值因素 | 同数据更新对照、clip/ratio/版本/precision差异解释结果 | 对齐实现后仍出现相同的状态条件化结构 |

模型小、Teacher更大或Teacher经过蒸馏，本身不构成机制证据。不要自动断言小模型“不可能学会”。50步无增益只说明当前预算与设置未观察到增益。

### 14.3 统计与对照

以task/episode为主要单位。多个turn、token、branch不是独立task样本。主汇总先每task聚合，再做配对差值和task-cluster bootstrap区间；若task很少，优先展示原始点和区间，不强行给显著性结论。

rollout seed与training seed分开。第一次training seed42可用于发现，确认任务与新continuation seeds不参与挑case；必要时补training seed43。

技术失败率并列报告：全部提交episode的operational结果，以及剔除明确技术失败后的behavior分析。不得只展示后者。不可恢复状态同样报告分母和类型。

对candidate价值反转只给 `exploratory/confirmed_on_new_repeats` 标签；重复验证用新随机样本，不用同一批数据既选最强反转又当确认。若候选相同、结果全败、剩余预算太少，标识信息不足。

### 14.4 计算成本与公平性

每种模型/teacher条件同时报：实际updates、有效训练token、环境action数、Teacher评分token、诊断额外rollout、申请GPU小时与实测运行时间。不同模型相同步数不等于同计算成本，既给step对齐也给token/资源对齐图。

Teacher相关改进必须连接到Student独立执行。Teacher采样质量或混合续写成功，只说明某一环节，不直接称为教学能力提升。

### 14.5 第一轮必须产出的四组图/表

1. **训练动态**：独立success/score、技术失败、分段信号、动作长度/循环、update指标，含S0。
2. **固定状态2×2价值图**：Delta_T vs Delta_S，标prefix数量、误差/不确定性及不可用分支。
3. **选择—承接动态**：step0/25/50同候选的采纳频率与Student条件成功率；固定池和动态池分开。
4. **10–20个可重放case**：有效学习、可疑不匹配、Teacher不可靠、工程假象及推翻原假设的反例都保留。

附机器可读 `metrics.csv/parquet`、`prefix_results.parquet`、`case_index.json`，不能只交PNG和SwanLab截图。

---

<a id="section-15"></a>

## 15. 调度异常、降级与恢复预案

| 情况 | 允许的处理 | 不允许的处理 |
|---|---|---|
| 模型下载掉线 | 续传、同SHA重试、记录asset状态 | 改为未经确认的第三方权重且不记 |
| SwanLab无网络/key | 自动offline，本地日志，后续ioqueue sync | 因监控无法联网把训练整个取消或泄露key |
| GPU OOM | 降microbatch/推理并发、保持有效batch；必要时较大profile | 顺手改Teacher、任务、loss后与原run混为同条件 |
| 30B资源不足 | 先做T08路线；T30只做顺序诊断或TP2 | 把MoE激活参数当总权重显存 |
| WebShop full内存过高 | 共享数据server、降低worker、独立申请可用RAM | small结果冒充full，或用登录节点当环境服务 |
| 到达time limit | 更新边界checkpoint、记录实际step、重提交正确依赖 | 只换权重继续却称optimizer恢复；重置schedule隐瞒 |
| 某branch异常 | 标缺失，报告partial，其他分支继续 | 缺失填0、整个DAG无条件卡住 |
| 没观察到预期gap | 记录负结果，确认基线有效后调整假设/预算 | 故意挑崩溃配置、只保留有利case |
| tokenizers不兼容 | 更换同token语义配对或单独设计跨词表方法（本轮不做） | 直接传不同token IDs并比较logprob |
| Ray冲突 | per-job端口/temp/PID隔离，重启本job | `ray stop --force`杀同节点其他实验 |

每个异常先区分基础设施与真实行为。允许合理修复，不要求“复现所有旧bug”；修复后的协议与原文不同之处必须写清。

---

<a id="section-16"></a>

## 16. Codex 工作包、执行顺序与完成定义

| 工作包 | 能与什么并行 | 最低完成定义 |
|---|---|---|
| W0 本地控制/站点变量 | 源码审阅、patch设计 | 脚本无占位符误提交；遵守登录限制 |
| W1 环境与资产 | 各模型下载、三个模拟器分支 | 版本锁、READY、最小smoke |
| W2 P00–P04补丁 | 数据清单准备 | 错误传播、日志、SwanLab offline、评分/恢复测试 |
| W3 E0及pair smoke | S17/S04/T08/T30评估 | 有原始轨迹与可解释对比，不强求成功率 |
| W4 E1并行训练 | E2/E3 step0、另一Student训练 | 目标50步或可恢复partial；独立checkpoint |
| W5 E2–E4诊断 | 不同prefix shard、已完成checkpoint评估 | 固定候选、真实状态恢复、清晰数据用途 |
| W6 CPU分析 | 各分支独立聚合 | 四类图/表、case册、局限与反例 |
| W7 E5/E6 | 按具体线索再安排 | 一项明确对照与新样本确认，不自动全开 |

**推荐最短可执行路径：**W0→(W1/W2并行)→E0/pair smoke→S17和S04可用分支并行50步＋少量step0诊断→完成checkpoint固定probe分析→报告。不要等所有高级诊断模块写完才开始第一条训练。

Codex可自主修复路径、兼容性、缓存、资源等工程问题并记录；改变主训练setting、引入Teacher RL、改数据协议以取得预期结果、扩大到大规模训练/付费队列，须先说明并获得用户同意。

由于尚无研究结果，本方案不规定Teacher RL的reward。未来可能从信息条件、程序兼容、续写价值或有限更新机制推导目标；本阶段只保证有足够证据判断哪一种解释值得深入。

---

<a id="section-17"></a>

## 17. 最终交付结构与报告模板

```text
reports/pilot-v1/
├── EXECUTION_STATUS.md           # 真实完成/失败/排队/跳过列表
├── ENVIRONMENT_AND_PROTOCOL.md   # 来源、依赖锁、代码patch、模型和split
├── RESULTS.md                    # 结果、区间、成本、未支持的判断
├── MECHANISM_NOTES.md            # 现象→解释→替代解释→干预→结果
├── jobs.tsv                     # job key、ID、依赖、状态、重试
├── swanlab_runs.json             # role/run_id/url或offline目录
├── metrics.parquet
├── prefix_results.parquet
├── figures/
├── cases/
└── ARTIFACT_INDEX.json           # 所有证据路径与hash
```

`RESULTS.md`至少回答：

- 哪些师生配对、环境和实际资源完成了什么？未完成为什么？
- Vanilla OPD是否改变Student关键动作，而不仅是格式和文本？
- Teacher在Student访问状态上是否仍能提供可靠行为？
- 是否观察到候选价值排序随续写者变化？有多少独立任务/重复支持？
- Student的选择与后续执行是否同步改善？哪些反例存在？
- 哪些发现可归因于实现、预算或协议，哪些仍值得研究？
- 下一步最有信息量的一个对照是什么？不要列十个未经筛选idea。

严禁把计划值写成测得值；未运行的实验写 `not_run`；因排队未完成写 `pending`；基础设施失败写 `technical_failure`，不填成模型0分。

---

<a id="section-18"></a>

## 18. 给 Codex 的精简执行指令

> 从本地执行本方案。先读取上传的Orion指南和本文，使用真实账户/根目录，禁止在登录节点运行Codex、安装环境或做计算。固定TCOD指定commit，在本地完成最小patch与控制脚本，上传后通过ioqueue准备环境、模型和三个benchmark；GPU统一优先testqueue且不超过18小时。利用独立job与dependency并行完成S17/S04初筛及可用配对的50步Vanilla OPD，Teacher只评分。Teacher生成与2×2 continuation只作diagnostic，禁止回流训练。沿用并修复TCOD SwanLab接口，offline不要求key，本地记录始终保留。先完成少量可解释的真实实验，不以论文分数、显著性或预设gap为门槛；工程底线不让步，研究验收宽松。保存所有配置、模型/data身份、精确tokens、环境状态重放信息、训练信号、checkpoint及作业记录。从固定prefix和训练前后对照建立证据，保留正例、负例和未区分结论。不要直接设计Teacher RL reward，不追求绝对原创，不因已有相似工作否定推进空间，也不要为了支持预期机制选择性呈现结果。

---

<a id="section-19"></a>

## 19. 依据与进一步核对入口

**[S1] 用户上传：`acrc-orion-cluster-guide(3).md`（指南标注更新2026-10-05）。**
本方案的队列、18小时上限、登录节点禁令、module和存储规则来自该文件。重点是§0、§2、§3、§4、§6.1–6.2。保留原文件在control/docs，不擅自修改集群规定。

**[S2] TCOD固定版本与安装声明。**

```text
https://github.com/kokolerk/TCOD/tree/465eef4406ad0cff675b36bd46f37f28b1736ff9
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/pyproject.toml
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/README.md
```

**[S3] 已核对TCOD SwanLab接口。**注册表已有swanlab；实现强制读取key，需要offline修补。

```text
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/utils/monitor.py
```

**[S4] 已核对启动器错误处理与Ray入口。**

```text
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/cli/launcher.py
```

**[S5] 已核对GPU分配检查。**

```text
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/config_validator.py
```

**[S6] 官方benchmark入口。**WebShop small/full与默认1000商品说明来自官方README；其conda步骤不适用于本项目HPC安装规则。

```text
https://github.com/alfworld/alfworld
https://github.com/princeton-nlp/WebShop
https://github.com/princeton-nlp/WebShop/blob/master/README.md
https://github.com/allenai/ScienceWorld
```

**[S7] SwanLab初始化、离线同步。**使用锁定SDK对应的API；`logdir`、mode、id/resume由当前文档核对，离线日志格式随版本可能变化。

```text
https://docs.swanlab.cn/api/py-init.html
https://docs.swanlab.cn/en/guide_cloud/experiment_track/sync-logfile.html
```

**[S8] SwanLab指标与step语义。**重复同指标同step会忽略后到值，需要单writer聚合。

```text
https://docs.swanlab.cn/guide_cloud/experiment_track/log-experiment-metric.html
https://docs.swanlab.cn/en/api/py-log.html
```

**[S9] Hugging Face下载API。**使用不可变revision，缓存/目录规则按锁定SDK处理。

```text
https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
```

**[S10] SLURM官方sbatch与array。**dependency、array并发、信号、变量不展开等由实际安装版本核对。

```text
https://slurm.schedmd.com/sbatch.html
https://slurm.schedmd.com/job_array.html
```

**[S11] Ray on Slurm与多任务端口隔离。**不直接套用新版symmetric-run到旧Ray；先核对本环境CLI。

```text
https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html
```

**[S12] 本研究固定的OPD代码入口。**

```text
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/TCOD_examples/alfworld/opd.yaml
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/algorithm/algorithm.py
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/algorithm/advantage_fn/on_policy_distill_advantage.py
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/workflows/envs/TCOD/alfworld/OPD_workflow.py
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/workflows/envs/TCOD/webshop/OPD_workflow.py
https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/workflows/envs/TCOD/scienceworld/OPD_workflow.py
```

**版本声明：**本方案不以任何浮动依赖/模型的最新状态为运行保证；Codex执行时将实际解析的版本锁住。来源支持的代码行为与本方案提出的实验规模、统计分析和资源建议已区分。未取得的HPC现场信息必须查询，不编造。
