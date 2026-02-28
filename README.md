# codes_e_new

预测钢渣碳化效率：多模型训练 + 集成（OE）。

## 快速开始

1) 进入目录：

`cd codes_e_new`

2) 运行一次完整实验（会生成 `runs/`，并且在runs目录下生成一个新的 run 目录）：

`python main.py`

3) 快速跑通（epoch很小，用于验证代码是否跑通）：

`python main.py --quick`

## 数据

默认使用 `data/The sample data used in the study.xlsx`。

## 输出结果（每一次运行，都会在runs目录下生成一个 run_id）

`runs/<run_id>/`
- `config.json`：本次运行的完整配置
- `models/`：每个模型的 best checkpoint
- `excel/`：所有生成的数据，可用于绘制图形以及用于模型效果分析对比等，包含指标、预测、训练历史、集成机制等
- `figures/`：绘制出的评估图，包括loss、R2、散点图、残差图、指标对比等

## 重要原则：避免信息泄漏

- `train/val` 用于训练与早停（模型训练与调参）
- `test` 用于最终泛化评估与分析，不参与调参
