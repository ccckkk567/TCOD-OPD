# TCOD ALFWorld 每步上下文修正执行文档

版本日期 2026 年 10 月 31 日
适用对象 在实际研究仓库中执行修正的 Codex
审计基准 `kokolerk/TCOD@465eef4406ad0cff675b36bd46f37f28b1736ff9`
任务范围 ALFWorld 独立 Vanilla OPD 的输入协议及其训练评估一致性

## 1 执行目标与授权边界

将指定的 ALFWorld Vanilla OPD 运行修正为以下协议：每次环境决策重新构造一条独立 user message，包含固定任务目标、当前 observation、当前 admissible actions，以及此前最近最多两组 observation/action。第 1、2、3 次及后续决策分别使用 0、1、2、2 组历史。

继续由 Student 自主生成并推进环境；Teacher 对每个 turn 中 Student 实际采样的原始 token 序列评分；每个 turn 保留一个 Experience。已有的 OPD 目标、response mask、评分位置对齐与训练算法保持原义。

**本次交付是执行方案，不表示用户的本地代码已经修复，也不表示已经运行训练。** 方案依据公开固定版本审计；当前会话没有用户服务器上实际仓库、私有补丁、已批准配置和运行日志的完整快照。因此执行者必须先做本地映射，不能把下文候选文件名当作已存在的本地文件。

用户此前要求：改变原 TCOD 行为的代码、配置或包装脚本，先说明范围，再取得明确批准。按以下规则执行，避免重复询问：

1. 若用户只是提供本文供审阅，先完成只读审计和具体修改清单，形成可审查的补丁草案；应用行为变更前取得一次明确批准。
2. 若用户已经明确要求“按这份文档执行修正”，视为批准本文规定的实现与验证范围；完成本地核对后继续，不按文件或测试逐次询问。
3. 真实训练沿用本会话已有的授权阶段和预算。若 E1 的 2＋2 恢复训练已经获准，在修正协议和前置验收通过后执行；本文不自动授权扩展到新的长训练、额外消融或其他环境。
4. 本轮拟新增的行为范围必须说明完整：显式上下文模式、独立完整模板输入、训练与实际评估入口共用构造函数、必要的验证与运行标识。不要把它简化表述成“只改一行”。

本次不新增摘要模型、检索记忆、Teacher rollout 干预、额外特权状态或新的蒸馏损失。不迁移至 verl-agent，也不顺带修改 B2F、F2B、ScienceWorld 或 WebShop 的行为。

## 2 已确认的代码事实

### 2.1 作者公开说明与本方案的地位

Issue #9 中，作者明确说明评估使用独立的每步模板和最近两组 observation/action，训练也使用固定历史并按 turn 形成 experience。该 issue 页面没有可直接应用的完整修复补丁，也未显示关联分支或 PR。[S1]

本文将这一说明落实为可执行的工程规格。**它属于依据作者公开说明重建的协议，不应称为作者官方补丁，也不能声称逐字复原其未公开的全部实验脚本。** 特别是首步是否使用 NO_HIS 的内部实现细节，没有完整官方补丁可以逐行确认；本文明确选择所有步骤使用完整模板。

### 2.2 固定版本中问题的具体位置

以下行号用于定位固定版本；本地有差异时按符号和语义重新定位。

| 文件与位置 | 代码事实 | 修正含义 |
| --- | --- | --- |
| `OPD_workflow.py` 第 104 至 110 行 | `format_messages()` 返回空列表 | 原生路径没有 system prompt，无须新增 |
| 同文件第 125 至 128 行 | 分别建立 task description、history、memory 和 turn_responses | 四者承担不同职责，不要合并 |
| 同文件第 143 至 162 行 | `len(history) < 2` 时使用 NO_HIS | 前两次决策不把已有的一步历史放入当前模板 |
| 同文件第 164 至 170 行 | memory 累积旧 user 与完整 assistant response | 虽然模板内部窗口为两步，实际模型输入仍包含旧对话 |
| 同文件第 179 至 181 行 | 先记录动作前 observation 与 parsed action，再执行环境动作 | 当前配对时序正确，应保留 |
| 同文件第 195 至 232 行 | 已对每个 turn 的 Experience 分别评分并返回列表 | 不需要再次实现整轨迹拆分 |

来源为固定版本的 workflow 与模板文件。[S2][S3]

### 2.3 只清空 memory 会产生第二步信息缺失

`ALFWORLD_TEMPLATE_NO_HIS` 没有独立的 `task_description` 字段，也没有 history 字段。初始 observation 通常包含任务，原实现通过保留第一条 user message，让第二步仍然可以找到任务。

假设第二步 observation 只是“柜门关着”，不复述任务。若只删除 memory 累积，却继续走 `len(history) < 2` 的 NO_HIS 分支，第二次决策将失去固定任务目标及第一组 observation/action。[S2][S3]

因此两项修改必须一起实施：

- 停止跨 turn 累积对话消息。
- 每次都用已有的完整 `ALFWORLD_TEMPLATE`，填入当时实际存在的 0、1 或 2 组历史。

第一步也改为完整模板是本文的明确设计选择：它提供统一、可测试的输入契约。现有模板字符串无需重写，空历史用 `""`，历史数量用 `0`；无需添加“暂无历史”等新的提示词。

## 3 修正后的输入契约

令 `r` 为从 0 开始的当前决策索引，`o_r` 为此次动作执行前的 observation，`a_r` 为此次解析得到的动作，`H_r` 为此前已经执行的 observation/action 记录。

本步输入只由以下内容构成：

- reset 时提取的固定任务目标；
- `o_r`；
- 当前环境返回的 admissible actions；
- `H_r` 的最后至多两个元素；
- 已执行步数 `r` 和当前步数 `r + 1`。

输入消息列表恰好为一条 `role="user"` 消息。保留模型现有 chat template 的正常 token 包装，不人为增加 system message、assistant prefill 或上一轮完整回复。

| 第几次决策 | step_count | history_length | 历史配对 | 当前观察 |
| --- | ---: | ---: | --- | --- |
| 1 | 0 | 0 | 空 | O0 |
| 2 | 1 | 1 | O0 与 A0 | O1 |
| 3 | 2 | 2 | O0 与 A0；O1 与 A1 | O2 |
| 4 | 3 | 2 | O1 与 A1；O2 与 A2 | O3 |

必须保持以下语义：

1. history 保存的是动作执行前的 observation 和实际 parsed action；不能把执行后的 observation 配到前一步 action。
2. 当前动作尚未生成，不能提前进入本步 history；未来观察不得进入当前 prompt。
3. history 中没有上一轮完整 reasoning。它只使用现有 `_format_history` 的 observation/action 格式。
4. 完整 history、turn_responses 和原始审计日志可以继续保留；模型只读取 history 的末尾窗口。
5. admissible actions 继续使用原来的嵌套 list 解包、过滤 `help`、原始顺序、引号与换行形式。
6. 不删除初始 observation 中重复出现的任务文本，不清洗或总结 observation。
7. `HISTORY_LENGTH` 仍为 2。本次不开放窗口搜索或增加另一个可调 K 参数。

**固定历史条目数不等于固定 token 数。** 当前 observation、动作列表以及步数文本仍可能变长。因此验收检查窗口语义及 token 预算，不能要求真实任务每步 token 数完全相等，也不能保证所有任务永不溢出。

## 4 建议的实现结构

### 4.1 显式保留旧模式并启用修正模式

在 Vanilla OPD 的 `workflow_args` 中增加一个小范围选项：

```yaml
context_mode: per_step_h2
```

只接受以下两个值：

| 值 | 行为 |
| --- | --- |
| `legacy_accumulated` | 保留固定版本原有消息累积与 NO_HIS 分支 |
| `per_step_h2` | 本文定义的每步独立完整模板与两步历史 |

缺省值为 `legacy_accumulated`，避免旧配置、旧结果复查和正在使用的其他入口被静默改变。修正运行必须显式填写 `per_step_h2`，并在启动审计中验证实际生效值。

这是为本项目保留旧协议可复现性而作的工程选择，不是作者要求。若本地已有等价、可验证的协议选择机制，应复用，不再叠加第二套开关。

`legacy_accumulated` 分支应保持原输入内容和时序，尤其不要在抽取 helper 时把 legacy 的前两步也换成完整模板。用同一条固定模拟轨迹对比修改前后的 legacy messages。

### 4.2 在 reset 中独立解析模式

该 workflow 声明 `can_reset=True`，且构造函数会调用 `reset(task)`。[S2]

模式解析和合法性校验应进入 `reset(task)`，而不是只在 `__init__` 设置一次：

```python
mode = task.workflow_args.get("context_mode", "legacy_accumulated")
if mode not in {"legacy_accumulated", "per_step_h2"}:
    raise ValueError(f"Unsupported ALFWorld context_mode: {mode}")
# 校验通过后，再更新 task 状态与 self.context_mode。
```

不能使用 `self.context_mode` 作为后续任务的缺省值，否则对象复用时，评估行为会取决于它上一次执行过什么任务。也不能根据 `task.is_eval` 自动切换模式。

已有 `temperature` 与 `max_env_steps` 的 reset 处理方式是相邻问题，先记录实际有效值，本次不顺手扩大为全部 workflow 参数重构。

### 4.3 新增一个纯输入构造函数

候选新文件：

`trinity/common/workflows/envs/TCOD/alfworld/context.py`

若本地已有合适的纯函数模块，放入该模块即可。不要为了增加 helper 修改公共框架或创建新 workflow 注册体系。

推荐接口语义如下。此代码是实现参考，不是官方补丁，也不应脱离本地类型与导入规则直接盲贴：

```python
def build_alfworld_step_messages(
    *,
    task_description,
    observation,
    admissible_commands,
    history,
    step_index,
):
    commands = admissible_commands
    if commands and isinstance(commands[0], list):
        commands = commands[0]

    actions_text = "\n ".join(
        f"'{command}'" for command in commands if command != "help"
    )
    recent_history = history[-HISTORY_LENGTH:]
    user_content = ALFWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=step_index,
        history_length=len(recent_history),
        action_history="\n".join(recent_history),
        current_step=step_index + 1,
        current_observation=format_observation(observation),
        admissible_actions=actions_text,
    )
    return [{"role": "user", "content": user_content}]
```

helper 从原 `utils.py` 导入模板、常量和 observation 格式函数；不修改模板正文。它不能访问模型、环境、随机数、文件或外部摘要服务，也不能修改传入的 history 或动作列表。

启动审计确认实际 `HISTORY_LENGTH == 2`。若本地不是 2，报告协议冲突；不要仅靠模式名称假装仍在使用两步。

### 4.4 接入 Vanilla OPD 的现有循环

对 `per_step_h2`：

1. 按当前 observation、当前 info、固定 task description 和已有 history 调 helper。
2. 将得到的 messages 直接传给原 `self.model.chat_async(..., **kwargs)`。
3. 保留模型返回的原 Experience，不能 decode 后重新 tokenize 重建。
4. 不把本次 assistant response 追加到下一步消息列表。
5. 保留原 `parse_action`、`history.append(_format_history(...))` 和 `env.step` 时序。
6. 保留 `turn_responses`、终局赋值、EID、Teacher scoring 和返回结构。

对 `legacy_accumulated`，继续运行原来的累积路径。现有无 system prompt 的行为不变。

同步更新文件头和评分处把 prefix 描述为“累计全部历史”的过时注释，使注释能区分两种模式；不改变 registry 名 `OPD_alfworld_workflow`，不因为类名中含 VerlAgent 就更换实现框架。

## 5 必须保留的 OPD token 链路

### 5.1 Experience 已经按 turn 保存

固定实现已经在每次 Student generation 后把返回值加入 `turn_responses`，并逐条 Teacher 评分。[S2] 当前问题是每条 Experience 的 prompt 过长，不是缺少逐 turn 样本结构。

继续使用当前 turn 的：

- `response.tokens`；
- `response.prompt_length`；
- `response.logprobs`；
- `response.action_mask`；
- `response.response_text`；
- 原有 EID 和 reward 处理。

不能将整个 episode 拼成一个 Experience，也不能把本轮 reasoning 从训练 response 中裁掉，仅对 action token 做蒸馏。

### 5.2 Teacher 应评分完全相同的 token IDs

固定 vLLM 路径保存实际 prompt token IDs 与实际生成 token IDs。Teacher 通过 `logprobs_async` 直接接收这条完整序列，不再自行组织另一份 prompt。[S2][S5]

定义：

- `P = response.prompt_length`；
- `N = len(response.tokens)`；
- `R = N - P`。

对正常样本应成立：

```text
P > 0
R > 0
teacher_input_ids == response.tokens.tolist()
len(student_response_logprobs) == R
len(teacher_full_logprobs) == N - 1
len(teacher_response_logprobs) == R
len(response.action_mask) == R
```

Teacher 不应使用完整旧历史、Teacher 自己的续写、未来环境反馈或重建的 token 序列。若本地 scorer 与公开 vLLM 实现不同，先验证其实际返回契约，不能机械套用形状。

### 5.3 保留 prompt_length 减一的切片

公开 `vllm_model.py` 的 `logprobs` 返回 `prompt_logprobs[1:]`，长度为 `N-1`；返回数组第 0 项对应完整输入序列的 token 索引 1。[S5]

因此第一个 response token 的输入索引为 `P`，在返回数组中的位置为 `P-1`。现有切片正确：

```python
teacher_response_logprobs = teacher_full_logprobs[P - 1:]
```

例：`P=5`、`R=3`、`N=8`，Teacher 返回 7 个位置；response 应取数组索引 4、5、6。若改为 `[P:]`，会漏掉第一个生成 token。

测试必须使用按位置可区分的假评分检查首尾，不能只检查两边长度相同。不得用截短、补零或删除首 token 来掩盖不一致。

### 5.4 mask 与 trainer 的含义

普通单 turn Experience 的 action mask 覆盖本轮全部生成 token；历史 observation/action 在 prompt 中，不属于本轮 response loss。batch padding 应保持无效 mask。[S6][S8]

`to_data_proto` 将采样时的 Student logprobs 保存为 `rollout_log_probs`。实际优势函数使用 `old_log_probs`；在部分配置下 trainer 会重新计算它，而不是直接用采样值。[S7][S8][S9]

本次只记录并核验这条链路，保持原 `multi_turn_opd`、KL 系数、rollout correction 与 old-logprob 设置，不借 context 修复改写算法。文档和最终报告不能无条件宣称“训练时一定直接用 Teacher logprob 减采样时 Student logprob”。

Teacher 直接消费 Student IDs 的前提是词表与 token ID 语义兼容。记录并核验双方 tokenizer、added tokens、特殊 token IDs 与实际映射；不能仅凭模型都叫 Qwen 就认定兼容。若不兼容，停止把这条直接评分路径当作有效 OPD，另报问题，不通过重分词绕过。

## 6 训练与评估必须使用同一输入协议

### 6.1 先找实际入口

固定 `TCOD_examples/alfworld/opd.yaml` 里的评估 tasksets 被注释。配置解析对 eval 会优先使用 `default_eval_workflow_type`，缺省后才回退到默认 workflow。[S4][S10]

执行者必须列出：

- 实际训练入口及 registry 映射；
- Student baseline 与 checkpoint 评估入口；
- Teacher 独立评估入口；
- 本项目已有的轨迹重放、同历史评分和诊断入口。

检查解析后的实际配置，不只检查 YAML 文件名。特别检查所有启用的 eval tasksets 是否分别显式设置了 `context_mode: per_step_h2`；不能假定它们会继承训练 task 的 `workflow_args`。

### 6.2 不使用通用 benchmark 代替 TCOD 评估

`benchmark/config/alfworld-template.yaml` 指向另一个 `alfworld_workflow`。对应通用实现带 system prompt，累计全对话，输入及 experience 组织方式与 TCOD OPD 不同。[S11][S12]

不要为了完成本任务直接运行这份通用配置，然后把结果命名为修正后的 TCOD baseline。优先让现有项目 evaluator 调用同一个纯 helper；若评估直接走 Vanilla OPD，则核验该分支实际选择了修正模式。

Teacher 独立评估与 Student 评估应在同一任务、同一历史下生成相同的消息内容。若比较同一个模型的训练和评估，且 tokenizer、chat template、thinking 设置相同，则 prompt token IDs 也应相同。不同模型的独立评估不应无条件要求 tokenizer 输出相同。

### 6.3 不顺带改变其他实验因素

保留当前已批准的：

- Student、Teacher 模型及 checkpoint；
- tokenizer、chat template、thinking 相关开关；
- 生成温度、top-p、top-k、token 上限；
- admissible actions 信息与 action parser；
- 环境版本、split、任务集合及评估随机种子；
- optimizer、KL 系数、批量、资源布局和训练日程。

作者关于 non-think 的说明不等于授权删除模板中的 `<think>` 指令。发现不一致时单列记录，本补丁不同时改变多个因素。

训练与评估温度可以沿用原先不同的设置；需要一致的是目标输入协议，不要求所有采样参数被改成相同。

## 7 文件修改范围

下表是候选范围。执行者先对应到本地真实路径，再给出最终改动清单。

| 文件类别 | 允许的改动 | 约束 |
| --- | --- | --- |
| TCOD ALFWorld `OPD_workflow.py` | context_mode 解析、分支接入、相关注释、少量审计字段 | 保留 legacy 输入及现有 OPD 算法链路 |
| ALFWorld 新增 `context.py` 或现有纯函数模块 | 独立完整模板 builder | 不增加模型、环境或摘要依赖 |
| 本地实际 evaluator 与必要诊断入口 | 复用 builder、显式选择并记录协议 | 不改通用 benchmark 的默认行为 |
| 新运行配置或已有受控覆盖层 | 显式 per_step_h2、独立运行与存储标识 | 从当前已批准本地配置派生 |
| 有针对性的 tests | 四步输入、token 对齐、legacy/reset/配置一致性 | 按仓库现有测试组织方式 |
| 本次运行 manifest 与验收报告 | 协议、来源、真实验证结果 | 字符串不塞入数值 metrics |

共享 `utils.py` 中的模板字符串、`HISTORY_LENGTH`、`parse_action` 原则上无需修改。公共 `model.py`、`vllm_model.py`、`experience.py`、advantage 与 trainer 文件本次作为审计对象，不作为默认修改对象。

本地若有另一套已获批准的独立 Vanilla OPD 模块，优先修改真正被运行的路径，并记录其与上游固定版本的差异；不要在未使用的上游文件中完成一个无效补丁。

## 8 必要的验证

这些验证针对输入协议和训练数据完整性。它们不会证明研究假设成立，也不以成功率提高作为补丁正确性的条件。

### 8.1 四步模拟轨迹

使用 fake env 和记录调用的 fake Student，冻结四步 observation、admissible actions 与预定 response。模型记录 messages 时必须深拷贝，避免后续列表修改污染快照。

夹具要求：

- O0 同时包含唯一任务标记 `TASK_SENTINEL` 和独立观察标记 `OBS_0_SENTINEL`。
- 若夹具使用真实 `_extract_task`，将任务字段放在 O0 末尾，把观察标记放在它之前；该函数会将任务前缀之后的全部文本作为任务，避免意外把观察标记纳入固定目标。
- O1、O2、O3 分别含自身观察标记，不再复述任务。
- 每轮完整 reasoning 有专属 `THINK_i_SENTINEL`，与 parsed action 分开。
- 当前可行动作列表使用可辨识的当前状态内容。
- 所有待检查标记都应避免自然出现在任务、可行动作或其他 observation 中。

必须验证第 3 节表格，并逐步检查：

- 实际调用恰好一个 user message；
- 每步有固定任务，不要求任务文字恰好出现一次；
- 第二步含第一组 observation/action；
- 第四步不含 `OBS_0_SENTINEL`，只保留两组最近历史；
- 旧 reasoning 标记不进入后续输入；
- 观察与动作编号相配，当前输入没有未来 observation；
- history 和命令列表未被 helper 原地修改。

不能用“某个普通动作字符串没出现”断言未来信息不泄漏，因为该动作可能合法地出现在当前 admissible actions 中。

### 8.2 Experience 与 Teacher 对齐

四轮模拟生成应得到四个 Experience，Teacher 分别接收四条对应的原始 token 序列。使用不同 prompt 和 response 长度覆盖 padding 情况。

检查：

- 原 token IDs、prompt_length、采样 logprobs 未被重建；
- Teacher 入参逐 token 相等；
- 第 5 节长度关系成立；
- Teacher response 切片首项确实对应 `P-1`；
- 原 run 标识和 step 0、1、2、3 保持规则；
- prompt 中的历史不进入 response mask；
- 本轮实际生成 token 有有效 mask，padding 无效；
- reward 的现有广播语义未因改 context 改变。

若 batch 转换依赖真实 verl/torch，运行仓库已有的轻量数据管道测试或一个小型集成测试即可，不启动分布式训练来证明纯 shape 关系。

### 8.3 legacy 与 reset 一致性

必要场景：

1. 同一固定模拟轨迹下，新 legacy messages 与修改前逐 turn 一致。
2. 同一 workflow 对象依次 reset 到显式 per_step、未填写 mode、显式 per_step 三个任务，模式依次为 per_step、legacy、per_step。
3. 第二个 episode 不继承第一个任务的目标、history 或消息。
4. 不支持的 mode 明确报错，不静默回退。

### 8.4 配置与评估一致性

给训练和实际评估入口同一份冻结的任务、observation、admissible actions、history、step。比较 messages；在相同 tokenizer 与渲染参数下进一步比较 token IDs。

故意构造“训练 per_step、eval 漏 mode”的新运行配置，启动审计必须报出协议不一致。可把这一检查放入已有运行验证脚本，无需创建大型配置系统。

### 8.5 长轨迹与真实 smoke

先用 30 步等长人工观察验证窗口外标记不会回流、旧 assistant 消息不会累积。步数数字长度等微小变化是正常的，不使用过于严格的 token 数恒定断言。

随后在已有授权和环境条件下执行小范围真实无更新 smoke：

- 真实 ALFWorld reset/step；
- 实际 Student 生成；
- 实际 Teacher scoring；
- 覆盖 0、1、2、超过 2 组已完成历史的输入。

自然提前成功的 episode 不应被强行继续；窗口覆盖可由模拟轨迹补足。记录真实任务 ID、步号、原 messages 样例、prompt token 数、response token 数、Teacher 返回与切片长度、有效 mask 数，以及异常状态。

通过 smoke 只能报告“输入与评分链路通过”，不能报告“论文成功率已复现”。

## 9 overflow 与 dummy 的处理边界

固定 `model.py` 的溢出处理会返回 `prompt_truncated` Experience，不执行实际模型生成；其 response 是截断边界上的 prompt token，logprobs 为零。`Experience` 对这种记录设置零 action mask。[S13][S6]

所以“dummy 一定被当作正常 token 学习”不是准确描述。但现有 OPD 循环仍可能解析它、调用 env.step、执行 Teacher scoring，并在未应用 response mask 的 workflow KL 汇总中纳入它。[S2]

本方案固定以下处理：

1. **核心补丁不改公共截断实现，也不暗中改变 legacy 的回退行为。**
2. 在验证 harness 中，对真实 Student 返回值立即检查 `truncate_status`、有效 response mask 和长度。若为 prompt_truncated，在返回 workflow 进行 action parsing 之前记录错误并中止本次验收。该 guard 只服务验证，可通过包装模型调用实现。
3. 使用一个人工 dummy 夹具验证这个 guard 确实阻止其被当成正常动作执行。它验证的是验收入口，不要把测试结果误报成公共 workflow 的生产截断策略已改变。
4. 中止时保存计划任务数、已运行任务、失败任务 ID 和原因，不计算一个悄悄剔除失败任务后的“更高成功率”。若 runner 会自动重试或吞掉异常，使用直接调用验证 harness，或以已有方式令该次验收明确失败。
5. E1 启动前，确认现有日志或本次审计记录能观察每个返回 Experience 的 `truncate_status`、有效 response token 数和异常计数；没有可用观测时，补充不改变动作和训练语义的审计记录。验证 harness 的中止能力不代表生产 workflow 已获得相同保护。后续授权训练若检测到 prompt_truncated 或无有效训练 token，将该次验收标为失败并停止继续扩展实验，不靠修改 token 上限、删除动作列表或关闭保护来通过验收。
6. 如果真实独立 prompt 仍超限，生产环境的 fail-fast、显式终止或其他回退策略属于新的行为决策，先报告触发输入和拟改范围，再按已有授权处理。

Student 与 Teacher 的预算应分别检查。公开 Teacher 评分实现还会生成一个探测 token，输入序列长度须满足其 `max_model_len - 1` 契约。[S5]

response 达到 max_tokens、动作标签缺失、空 response 和 prompt overflow 是不同情况，应分别记录。后端未提供某个状态时标为不可观测，不能填成“零次”。

## 10 运行隔离与恢复训练

### 10.1 新协议单独建立运行

从当前已批准的本地配置派生修正配置。上游 `opd.yaml` 示例含 250 步等默认值，不能用它覆盖本项目的 E1 和 pilot 约束。[S4]

新运行至少隔离：

- run 名、输出目录和 checkpoint 目录；
- experience buffer 名称、数据库或队列 namespace；
- 轨迹缓存、Teacher score 缓存及评估输出；
- 协议标识和配置快照。

上游配置包含 buffer 名称及存储配置，实际后端和持久化方式以本地解析结果为准。仅换 checkpoint 目录不能证明旧经验已隔离，应核验 buffer、队列 namespace 和缓存均属于新运行。不要删除旧 buffer；新建独立标识并核验起始状态。

旧完整历史 Experience 不能通过替换 prompt_text 直接冒充新协议 on-policy 样本，因为其 response 是在另一份输入上采样的。可以在新的输入下重新评分做离线诊断，但必须单独标明，不能送进新 baseline 的 on-policy 训练批次。

### 10.2 保存足够的来源信息

使用现有 run manifest 或简单 JSON，记录：

- 上游基准 commit、本地 HEAD、未提交补丁指纹；
- context_mode、HISTORY_LENGTH、builder 与模板指纹；
- 实际训练和评估 workflow；
- 模型 checkpoint、tokenizer 和 chat template 标识；
- thinking、生成和 Teacher 评分相关有效配置；
- 数据 split、任务清单及随机种子；
- run、buffer、checkpoint 位置；
- 训练总预算、学习率日程、恢复点与实际更新计数。

在 Experience 已有 `info` 或 manifest 中记录字符串协议名。数值 `metrics` 只保存兼容聚合器的数值。记录配置意图之外，还要保存有限数量的真实模型输入以验证它确实生效。

### 10.3 E1 的 2 加 2

沿用既定要求：先 2 次真实 optimizer update，保存后恢复，再 2 次 update；pilot 至多 50 次更新，学习率日程按既定 100 步配置，保留原 8 GPU 等资源设置。

特别注意：

- 2＋2 发生在同一份修正协议下，不能将旧协议的前两步与新协议的后两步拼接。
- 同步恢复 optimizer、scheduler、global step 及框架要求的训练状态；只加载模型权重不等于恢复训练通过。
- 第 3、4 次更新的学习率与步数应接续既定日程，不能重启成新的 2 步 schedule。
- 不把 buffer 的 step 数、环境动作数、rollout 数或日志条目数当成 optimizer update 数。
- 若现有中止/恢复机制会改变 scheduler horizon，先查明并给出独立、明确的最小处理方案，不发明未经确认的 CLI 参数。
- 同一协议 2＋2 之间可以按框架要求恢复属于该 run 的状态；禁止混入旧协议经验。
- 2＋2 是恢复和更新链路验收，不自动批准继续跑满 50 次。

旧 checkpoint 可以在新 context 下评估，标签应为“旧协议训练 新协议评估”。正式的新协议训练对照从同一个原始 Student 初始化开始。

## 11 本地执行顺序

### 阶段 0 只读定位并形成实际修改清单

在用户指定的研究仓库中：

1. 读取适用的 `AGENTS.md` 与已有项目执行说明。
2. 检查当前分支、HEAD、未提交改动、实际运行配置和已有补丁。
3. 核对 workflow 注册映射、继承关系、辅助模型与评估入口。
4. 将本文的固定代码事实映射到本地符号；若上游或本地已经修复，不重复补丁。
5. 输出精确文件清单、拟改内容、验证方法和配置差异。需要批准时，具体到这些变更后再询问一次。

可用以下只读命令定位；`TCOD_REPO` 必须替换为已确认的实际路径：

```bash
TCOD_REPO=/ABSOLUTE/PATH/TO/TCOD
git -C "$TCOD_REPO" status --short
git -C "$TCOD_REPO" rev-parse HEAD
git -C "$TCOD_REPO" diff --stat
rg --files "$TCOD_REPO" -g 'AGENTS.md' -g '*alfworld*' -g '*OPD*' -g '*opd*'
rg -n 'OPD_alfworld_workflow|OnPolicyDistillVerlAgentAlfworldWorkflow|ALFWORLD_TEMPLATE|HISTORY_LENGTH' "$TCOD_REPO/trinity" "$TCOD_REPO/TCOD_examples"
rg -n 'default_eval_workflow_type|eval_tasksets|context_mode' "$TCOD_REPO/trinity" "$TCOD_REPO/TCOD_examples"
```

另按本地目录布局查找已有评估、replay、score 与研究脚本。不要读取凭据文件，不需要用户重新提供已经能在仓库中确认的路径。

### 阶段 1 实现本文的最小范围

在已获批准范围内使用独立工作分支或不干扰现有工作的隔离目录：

- 新增纯 builder；
- 在 Vanilla OPD 加模式解析和接入；
- 连接真实评估入口；
- 派生并显式标识新配置；
- 增加本次必要测试与审计输出。

不要 reset、清理或覆盖用户未提交的无关工作。若依赖本地未提交补丁，记录并保留它们，不能直接从旧上游 commit 建一个缺少这些前置工作的运行环境。

### 阶段 2 完成低成本验证

运行第 8 节有针对性的测试、相关静态检查和 `git diff --check`。现有测试框架若可复用，沿用其命令，不新建另一套测试架构。

从实际修改前代码或已确认的基准提取 legacy 快照进行比较。不要用新实现调用自身生成“expected”结果。

### 阶段 3 真实无更新 smoke

完成本地环境和模型链路检查，保存可审核的 messages、token 长度及评分对齐证据。只使用验证所需的少量任务，保持已批准的模型与资源布局。

环境或资源缺失时，交付已经完成的补丁、测试和阻塞证据，明确哪些真实验证尚未运行。不要降低模型规模、减少到不同 GPU 配置或换模拟结果冒充真实验证。

### 阶段 4 执行已获准的 E1

前置验证通过后，按有效授权执行第 10 节的 2＋2。保存两个阶段日志和 checkpoint 来源，报告真实 update、scheduler 连续性和协议一致性。

不要从这一步直接开展额外的 memory 消融、Teacher RL 或 50 步长训练。

## 12 验收表与停止条件

| 验收项目 | 通过标准 |
| --- | --- |
| 输入结构 | per_step 模式每次实际调用只有一条独立 user message |
| 第二步信息 | 当前 observation 不复述目标时，仍有任务及第一组历史 |
| 窗口与时序 | 历史最多两组、动作前 observation 配对、无旧 reasoning 累积 |
| 逐 turn 样本 | 每轮真实生成对应一个 Experience |
| Teacher 对齐 | 输入 IDs 完全相同，切片与 response 位置一致 |
| mask 与 padding | 本轮 response 有效，prompt 和 padding 不被当作生成监督 |
| legacy 与 reset | 旧模式保持原样；任务切换不泄漏协议或历史 |
| 训练评估一致性 | 实际入口与有效配置均选定同一协议 |
| 真实 smoke | 有真实生成与评分证据，无 dummy 冒充模型输出 |
| 运行隔离 | 新旧协议的 buffer、缓存、输出和恢复链路可区分 |
| 2＋2 | 若已获授权执行，真实更新与 scheduler 接续，协议未变化 |

以下情况不能宣称修复验收完成：

- 无法确定本地实际执行的 workflow 或评估入口；
- 修改范围超出已批准的输入修正；
- Teacher 与 Student token 语义不兼容；
- 只能通过重分词、截短评分或填零才能对齐；
- 仍存在 prompt_truncated 或无有效 token，却被统计为正常样本；
- 训练与评估使用不同 context 模式；
- 新协议混入旧缓存或旧 experience；
- 2＋2 只恢复权重、未恢复必要训练状态；

done/success 口径存在冲突时，可以报告 context 输入与 token 链路已通过，但不得报告环境成功率、reward 口径或依赖它们的正式 baseline 验收通过。该问题单列处理，不把 reward 重写强行捆入本次 context 补丁。

如果发现训练统计或 success 定义的独立问题，完成本次不受影响的修正和测试，将阻塞明确列出。不能为了让检查变绿而悄悄更改原算法、数据或统计分母。

## 13 相邻问题的处理

本次审计发现或提示了以下问题，需记录但不自动纳入 context 补丁：

- workflow 在 `done` 时把 `_final_reward` 设为 1.0，而非直接使用环境返回 reward；本地应核验实际 wrapper 的 done 与 success 语义。[S2]
- trainer 可能重算 `old_log_probs`；该设置不能被 context 修复顺带修改。[S9]
- workflow 的 `kl_divergence` 是采样 token 的 logprob 差值求和；其数值随响应长度、轨迹长度和采样变化，不能单独证明机制改善或蒸馏成功。[S2]
- advantage 返回的部分 metrics 不一定被 trainer 汇总；不能把缺少某个 trajectory KL 指标直接判定为 context 修复失败。[S9]
- 两步截取仍可能丢失更早 observation 中发现、但当前未重现的事实；这是协议本身的研究限制，不等于本次实现错误。正确摘要、完整历史与不同窗口的比较留到后续单独实验。

修正后若成功率上升，只能先得出在该评估协议下表现变化；不能据此直接确认此前提出的 policy 与 competence 迁移失配假设。

## 14 Codex 最终交付要求

最终报告应写明完成情况，而不是只回复“已修改 memory”。

交付以下内容：

1. **修改清单**：实际路径、函数、改动理由，明确是否修改了 evaluator 或包装脚本。
2. **可审查 diff**：基准 commit、本地 patch 来源与独立提交标识；不包含无关改动。
3. **配置差异**：新旧 mode、实际 workflow、buffer、输出目录，以及所有保留的研究参数。
4. **输入证据**：同一模拟轨迹第 1 至第 4 步的实际 messages；真实 smoke 的有限样例。
5. **token 对齐证据**：P、N、R、Teacher 输入哈希或 ID 对比、切片位置、有效 mask 数。
6. **验证结果**：逐项标注通过、失败或未执行，并给出命令、日志位置和阻塞原因。
7. **训练结果**：仅在实际执行后填写 update 数、checkpoint、恢复与 scheduler 连续性；没有运行则明确写未运行。
8. **限制与回滚**：哪些相邻问题未处理，如何回到 legacy 模式；保留旧日志和数据，不使用破坏性的重置命令。

完成标准是输入协议、训练样本和评估入口的一致性得到证据支持。最终成功率、论文复现程度及 Teacher–Student Gap 结论，需要后续正式实验另行判断。

## 15 原始资料

除 Issue #9 与作者明确链接的模板外，下列代码链接均固定到本次审计 commit。链接用于核对语义；执行者仍以本地实际版本映射为准。

[S1]: https://github.com/kokolerk/TCOD/issues/9
[S2]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/workflows/envs/TCOD/alfworld/OPD_workflow.py
[S3]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/workflows/envs/TCOD/alfworld/utils.py
[S4]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/TCOD_examples/alfworld/opd.yaml
[S5]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/models/vllm_model.py
[S6]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/experience.py
[S7]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/algorithm/advantage_fn/on_policy_distill_advantage.py
[S8]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/trainer/verl/utils.py
[S9]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/trainer/verl_trainer.py
[S10]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/config_validator.py
[S11]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/benchmark/config/alfworld-template.yaml
[S12]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/workflows/envs/alfworld/alfworld_workflow.py
[S13]: https://github.com/kokolerk/TCOD/blob/465eef4406ad0cff675b36bd46f37f28b1736ff9/trinity/common/models/model.py
[S14]: https://github.com/kokolerk/TCOD/blob/17a8af222017fb78a44a50e5711bfa08543de36b/trinity/common/workflows/envs/TCOD/alfworld/utils.py

- [S1 作者澄清与讨论原文][S1]
- [S2 Vanilla OPD workflow][S2]
- [S3 固定版本模板与 history 工具][S3]
- [S4 上游 ALFWorld OPD 示例配置][S4]
- [S5 vLLM generation 与 logprobs 契约][S5]
- [S6 Experience 与 mask][S6]
- [S7 OPD advantage][S7]
- [S8 Experience 到 verl batch][S8]
- [S9 trainer 的 old logprob 与 advantage 调用][S9]
- [S10 训练评估配置回退规则][S10]
- [S11 通用 ALFWorld benchmark 配置][S11]
- [S12 通用 ALFWorld workflow][S12]
- [S13 公共模型的 prompt 截断处理][S13]
- [S14 Issue 作者指向的模板版本][S14]
