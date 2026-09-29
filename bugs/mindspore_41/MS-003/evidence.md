# MS-003 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`2`
- 退出码 0：`2`；非零：`0`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative`
- `python reproducer.py --mode graph`

代表性控制命令：

- 复现器内部独立参考或跨模式对照

复验说明：复现器使用稳定的独立 float64 公式作为参考，并在两个执行模式得到相同误差。
