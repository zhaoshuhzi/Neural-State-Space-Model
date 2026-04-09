# Neural State-Space Model for Stimulation-Activated Region Discovery

一个适合上传到 GitHub 的最小可运行项目，用 **Neural State-Space Model (NSSM)** 对源空间 EEG / ROI 时间序列进行建模，并根据刺激输入 `u(t)` 估计 **刺激激活脑区**。

该仓库对应的整体思路与论文图一致：

- **Spatiotemporal Encoder**：把 ROI / source-space EEG 编成时序特征
- **Neural State-Space Model**：学习潜在神经动力学状态 `z_t`
- **Stimulus Control Path**：显式把刺激输入 `u(t)` 注入系统
- **Dynamic Network Decoder**：可选输出动态功能网络
- **Activation Discovery**：通过刺激开 / 关对比得到每个 ROI 的激活分数

## 1. 项目结构

```bash
.
├── README.md
├── requirements.txt
├── train.py
├── infer_activation.py
└── nssm
    ├── __init__.py
    ├── data.py
    ├── models.py
    └── utils.py
```

## 2. 模型核心

状态更新：

```math
z_{t+1} = (1-g_t) \odot z_t + g_t \odot \tanh(W_z z_t + W_e e_t + W_u u_t)
```

观测读出：

```math
\hat y_t = C z_t + D u_t
```

其中 `D u_t` 是显式刺激通路，用来增强“刺激驱动激活区域”的可解释性。

## 3. 安装

```bash
pip install -r requirements.txt
```

## 4. 快速开始

训练合成数据示例：

```bash
python train.py \
  --dataset synthetic \
  --outdir runs/demo \
  --epochs 20 \
  --batch-size 32 \
  --encoder-dim 64 \
  --latent-dim 32
```

使用训练好的模型做激活脑区推断：

```bash
python infer_activation.py \
  --checkpoint runs/demo/best_model.pt \
  --dataset synthetic \
  --outdir runs/demo_infer
```

## 5. 真实数据格式

把真实数据整理成 `npz` 文件，推荐包含以下字段：

- `X`: `[num_samples, seq_len, num_rois]`
- `U`: `[num_samples, seq_len, stim_dim]`
- `Y`: `[num_samples, seq_len, num_rois]`，可选；没有时默认 `Y=X`
- `A`: `[num_samples, seq_len, num_rois, num_rois]`，可选动态网络监督
- `roi_names`: `[num_rois]`，可选 ROI 名称

示例：

```python
import numpy as np

np.savez(
    "my_data.npz",
    X=X,
    U=U,
    Y=Y,
    roi_names=np.array(roi_names, dtype=object),
)
```

训练命令：

```bash
python train.py \
  --dataset npz \
  --data-path my_data.npz \
  --outdir runs/real_data \
  --epochs 50 \
  --batch-size 16 \
  --encoder-dim 128 \
  --latent-dim 64
```

如果有动态功能网络标签 `A`：

```bash
python train.py \
  --dataset npz \
  --data-path my_data.npz \
  --outdir runs/real_data_dfn \
  --predict-network \
  --lambda-adj 0.1
```

## 6. 如何发现刺激激活脑区

默认策略：

1. 用真实刺激 `u(t)` 运行模型，得到 `ŷ_t^stim`
2. 把同一批样本的刺激置零，再运行一次，得到 `ŷ_t^no-stim`
3. 对刺激开始后的时间窗，计算每个 ROI 的平均差异：

```math
score_r = \mathbb{E}_{t \ge t_{stim}} |\hat y_{t,r}^{stim} - \hat y_{t,r}^{no-stim}|
```

`score_r` 越大，表示该 ROI 对刺激越敏感，越可能是 **刺激激活区域**。

另外还会导出：

- `direct_stimulus_weights.json`

它来自模型里的显式刺激通路，可作为“刺激直达 ROI / latent state”的辅助解释指标。

## 7. 说明

这个仓库目前定位为：

- 方便直接上传 GitHub
- 方便你基于自己的 EEG / source 数据继续改
- 保留“刺激输入 → 潜在神经动力学 → 激活脑区”的可解释路径

真实研究里通常还会继续补充：

- 更严格的 source reconstruction
- subject-level cross-validation
- permutation / bootstrap 显著性检验
- ROI-level 多重比较校正
- 脑表面可视化
