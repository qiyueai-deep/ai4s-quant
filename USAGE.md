# 项目使用说明（PyCharm）

本项目 = 从零实现的 LLM 权重量化研究（主线）+ 两个已归档的历史版本。
所有数字由仓库代码在单机 CPU 上产出，不联网、无 GPU、不依赖 transformers。

## 一、环境准备（PyCharm）

1. 打开 PyCharm → `File → Open` → 选择 `ai4s` 文件夹。
2. `File → Settings → Project: ai4s → Python Interpreter`：
   建议新建一个 venv（Python ≥ 3.10），然后在 PyCharm 的 Terminal 里执行
   ```
   pip install -r requirements.txt
   ```
   依赖只有三个包：`numpy>=1.26`、`torch>=2.0`、`matplotlib>=3.8`。
   注意：本机实测环境**没有 scipy 也能跑**，全部线性代数用 numpy/torch 实现。
3. 运行配置无需任何参数；若要改种子或输出目录，用 Run Configuration 的
   Parameters 传入命令行参数（见下文）。

## 二、一条最小验证路径（约 5 分钟）

仓库已自带训练好的模型（`artifacts/target_q.pt`），首次运行只需两步：

| 步骤 | 在 PyCharm 中怎么跑 | 耗时 | 产物 |
|---|---|---|---|
| 1. 单测 | 右键 `tests_quant.py` → Run（或 `python tests_quant.py -v`） | 约 1 分钟 | 17 条断言全绿 |
| 2. 主实验 | 右键 `experiments_quant.py` → Run | 约 1 分钟 | `results_quant/` 下 6 张表 + 1 张图 + `06_summary.json` |

跑完后打开 `results_quant/06_summary.json` 与 `results_quant/fig_quant.png`
即可核对 README 中的全部数字。

## 三、完整复现路径（含从头训练，约 5 分钟）

如果想验证"从语料到模型到量化"全链路：

```
python -m lmlab.train --force --corpus hard --d-model 192 --n-layers 4 --suffix q
python experiments_quant.py
python experiments_quant.py --seed 7 --out results_quant_seed7
```

- 训练一步约 4 CPU 分钟，产物写入 `artifacts/`（checkpoint、tokenizer、manifest）。
- 全链路确定性：语料、初始化、校准窗口、评测窗口全部固定种子，同参数重跑
  `summary.json` 逐字节一致。第三条命令换种子复现，用于确认结论不翻转。

## 四、命令行参数

| 脚本 | 参数 | 说明 |
|---|---|---|
| `lmlab.train` | `--force` | 覆盖已有 checkpoint 重训 |
| | `--corpus easy/hard` | 语料难度（hard 才能让量化前沿有信号） |
| | `--d-model --n-layers` | 模型宽度与深度 |
| | `--suffix q` | 产物文件名后缀（如 `target_q.pt`） |
| `experiments_quant.py` | `--seed` | 全链路随机种子（默认 0） |
| | `--out` | 结果输出目录（默认 `results_quant`） |
| `tests_quant.py` | `-v` | 逐条显示断言 |

## 五、目录结构

```
ai4s/
├── README.md               研究说明：四条命题、全部实测数字、边界
├── PORTFOLIO.md            求职材料：简历措辞、面试问答、JD 对齐
├── USAGE.md                本文件（PyCharm 运行指南）
├── STAR.md                 项目讲解（STAR 法则，面试叙述用）
├── requirements.txt        三个依赖
├── lmlab/                  基础设施
│   ├── data.py             确定性高熵语料生成（easy / hard 两种难度）
│   ├── model.py            字符级 Transformer + tokenizer + checkpoint 读写
│   └── train.py            从头训练入口（python -m lmlab.train）
├── quant/                  量化主线
│   ├── calibration.py      hook 收集每层输入，H = 2XXᵀ（含 fp32 累加）
│   ├── rtn.py              RTN（per-tensor / per-channel）+ MSE 最优截断搜索
│   ├── gptq.py             GPTQ 从零实现（Cholesky 逐列舍入 + 误差回传）
│   ├── theory.py           闭式误差预测、方向性对齐 κ、注水式比特分配
│   └── pipeline.py         fake-quant 前向 + val CE 评测
├── experiments_quant.py    E1-E4 四组实验（主实验入口）
├── tests_quant.py          17 条断言（含 GPTQ 恒等式、暴力对照等）
├── artifacts/              自训练模型 target_q.pt + tokenizer + manifest
├── data/                   确定性语料（可由代码再生成）
├── results_quant/          实验产物：5 张表 + 对账表 + 图 + summary.json
├── _legacy_specdec/        上一版投机解码项目（原样归档，不属主线）
└── theory_lab/             更早的统计物理项目（原样归档，不属主线）
```

## 六、常见问题

- **想只跑某组实验？** `experiments_quant.py` 的 `main()` 里按 E1-E4 分了
  独立函数，注释掉不需要的调用即可。
- **换机器/换 Python 版本？** 只需满足 `requirements.txt`；torch CPU 版即可，
  无需 CUDA。
- **训练中断了怎么办？** 删掉 `artifacts/` 里对应带 `q` 后缀的文件后
  重跑带 `--force` 的训练命令。
- **`_legacy_specdec/` 与 `theory_lab/` 是什么？** 历史版本原样快照，
  与主线无关，可整体删除而不影响主线任何功能。
