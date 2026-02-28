# codes_e_new_20251225

一个全新、可复现实验工程：多模型训练 + 集成（OE）+ 全套评估图表 + Excel 全量导出。

## 快速开始

1) 进入目录：

`cd codes_e_new`

2) 运行一次完整实验（会在 `runs/` 下生成一个新的 run 目录）：

`python main.py`

3) 快速自检（更少 epoch，用于验证代码流程）：

`python main.py --quick`

## 数据

默认使用 `data/factor1016.xlsx`。

## 输出结构（每次运行一个 run_id）

`runs/<run_id>/`
- `config.json`：本次运行的完整配置（可复现）
- `models/`：每个模型的 best checkpoint
- `excel/`：所有可重绘数据（指标/预测/训练历史/集成机制等）
- `figures/`：评估图（loss/R2、散点图、残差图、指标对比等）

## 重要原则：避免信息泄漏

- `train/val` 用于训练与早停（模型选择/调参只看验证集）
- `test` 仅用于最终泛化评估与出图/导出，不参与调参
