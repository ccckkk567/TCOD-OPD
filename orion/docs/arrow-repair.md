# 2026-10-31 Arrow/NumPy 修复

实际日志：TCOD config.check_and_update → file_reader → datasets → pyarrow
在导入时失败，报要求NumPy≥2而实际为1.26.4。日志未给出已安装Arrow版本，
修复job先通过metadata读取并记录，不把推测版本当事实。此次没有到训练更新阶段。

只安装PyArrow21.0.0预编译wheel，使用--no-deps保留其余环境；预先核对
所有已安装包对Arrow的版本约束，不满足就停止。随后pip check、导入原TCOD
文件reader、ALFWorld/TextWorld，以及NumPy→Arrow→datasets转换验证。
官方21.0支持NumPy≥1.16.6：
https://arrow.apache.org/docs/21.0/python/install.html#dependencies
当前26.0要求NumPy≥2：https://arrow.apache.org/docs/python/install.html
本地仅验证脚本语法和生成配置，实际wheel兼容性需要HPC修复作业验证。

`submit_arrow_repair_chain.sh REVISION` 先拒绝仍有opd-*排队/运行的情况，
固定worktree后提交4个job：修复1GPU/15min，首次训练2GPU/4h，恢复2GPU/4h，
pilot2GPU/18h。均afterok，任一失败阻止后续运行；修复独占共享环境锁。
不得取消队列中其他实验。失败目录保留，新的run_name追加-arrow-fix，
修复前后pip freeze存于logs/tcod-opd/env/arrow-<job>/。
不改变TCOD算法、截断、提示词或训练数据，不凭此次修复宣称训练已验证。
