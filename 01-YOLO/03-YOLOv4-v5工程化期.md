# YOLOv4–v5：工程化期

> **核心问题**：YOLO 从「学术模型」变成「工业标配」的这一步，是哪些具体改动完成的？

---

## 1. YOLOv4（2020）

**论文**：*YOLOv4: Optimal Speed and Accuracy of Object Detection*, arXiv:2004.10934
**作者**：Aleksei Bochkovskiy, Chien-Yao Wang, Hongyang Mark Liao

### 1.1 论文的核心贡献：一个分类法

v4 论文最著名的部分不是某个模块，而是 **Bag-of-Freebies (BoF) 与 Bag-of-Specials (BoS)** 的分类：

| 分类 | 定义 | 关键特性 | 例子 |
|---|---|---|---|
| **BoF** | 只影响**训练**的技巧，不改变推理时的网络结构 | **可零成本迁移到任何模型/任何 backbone** | Mosaic、MixUp、Cutout、DropBlock、cmBN、label smoothing、CIoU 损失、DropPath |
| **BoS** | 只对**特定模型**有效的结构改动 | 迁移性差，依赖具体架构 | CSPDarknet-53、PANet、Mish、SPP 变体、SiLU |

**为什么这个分类如此重要？**
当你从 v5 升级到 v8 时：BoF 类技巧（增强、损失、优化器）几乎可以无损迁移；BoS 类（CSP 换 C2f、SPP 换 SPPF）需要逐个核对。这套分类帮你区分「哪些改动值得抄，哪些改动会破坏新架构」。

### 1.2 BoF 清单

#### 数据增强

| 技术 | 做法 | 增益 |
|---|---|---|
| **Mosaic** | 4 张图拼成 1 张训练图 | +1.6~3.0 mAP |
| **RandomAffine / RandomPerspective** | 随机仿射 + 透视变换 | 小目标鲁棒性 |
| **HSV 抖动** | 随机 H/S/V 偏移 | 光照鲁棒性 |
| **RandomFlip** | 水平翻转 | 免费 |
| **MixUp** | 两图线性混合 | +~1 mAP |
| **Cutout** | 随机遮挡图像区域 | 正则化 |
| **DropBlock** | 在特征图上随机丢弃块 | 卷积网络专用正则 |

#### 训练技巧

- **cmBN（Cross mini-Batch Normalization）**：小 batch 训练时用跨 batch 统计量，缓解 BN 在小 batch 下不稳
- **权重衰减调整**：不同模型用不同策略（v4 论文给出一张查找表）
- **梯度累积**：模拟大 batch
- **余弦学习率调度**
- **Label Smoothing**
- **多尺度训练**

#### 损失函数

- **CIoU Loss**（替代 GIoU，见 §6.3）

### 1.3 BoS 清单

| 模块 | 说明 |
|---|---|
| **CSPDarknet-53** | Cross Stage Partial，把梯度路径分成两路再拼接，减少重复梯度 |
| **PANet** | 自底向上 + 自顶向下双向融合，增强浅层细节 |
| **Mish 激活** | $x \cdot \tanh(\text{softplus}(x))$，平滑、易求导 |
| **SPP 变体** | SPP + SPPF（YOLOv5 的简化版） |
| **SiLU / Swish** | $x \cdot \sigma(x)$ |

### 1.4 性能

| 指标 | 数值 |
|---|---|
| COCO test-dev mAP@0.5 | **43.0** |
| COCO test-dev mAP@0.5:0.95 | 29.3 |
| 速度 | 65 FPS (Tesla V100) / 77 FPS (1080p GPU) |
| 对比 YOLOv3 | 33.0 → 29.3 (mAP@.5:.95 因分辨率/训练差异不完全可比) |

---

## 2. YOLOv5（2020–2022）

**仓库**：`ultralytics/yolov5` ｜ **论文**：无（官方明确不发论文）
**版本节奏**：v1.0 (2020-06) → v3.0 (2020-10) → v4.0 (2021-01) → v5.0 (2021-07) → v6.0 (2022-03) → v7.0 (2022-12)

### 2.1 v5 的真正贡献：训练配方

v5 的精度提升**几乎全部来自训练工程**，而非网络结构。这是理解 v5 的关键：

| 技巧 | 作用 | 实现要点 |
|---|---|---|
| **AutoAnchor** | 换数据集自动重算 anchor | 训练前对数据集所有 GT 框做 k-means（默认 1000 次迭代），算出 3 组 anchor 存到 `data/*.yaml` |
| **Mosaic** | 4 图拼接 | 数据加载器中实现，`close_mosaic=10` 在最后 10 epoch 关闭 |
| **MixUp** | 两图混合 | alpha=0.2，概率 0.1 |
| **Copy-Paste** | 实例级复制粘贴 | 概率 0.0（默认关闭，YOLOX 的贡献） |
| **EMA** | 指数滑动平均 | decay=0.9999，评估用 EMA 权重 |
| **AMP** | 混合精度训练 | 自动 GradScaler |
| **Cosine LR + warmup** | 学习率调度 | warmup 3 epoch，lr 从 0.001 线性升到 0.01 |
| **Label Smoothing** | 正则 | 0.0（默认关闭） |
| **多尺度训练** | 尺度泛化 | `scale=0.5` 随机缩放 ±50% |
| **Rectangular Inference** | 矩形推理 | 推理时按长宽比 padding，减少无效计算 |
| **TTA（Test-Time Augmentation）** | 测试增强 | 三倍缩放 + 翻转，mAP +1~2 |

### 2.2 v5 的网络结构

#### Backbone：Focus + C3

**Focus**：把 4×4 的像素信息集中到通道维（早期版本）

```
输入 640×640×3
  ↓ slice 成 4 份 320×320×3  →  concat → 320×320×12
  ↓ Conv 3×3 s=2                       320×320×32
  ↓ Conv 3×3 s=2                       160×160×64
  ↓ C3                                160×160×64
  ...
```
> v6.0 之后 Focus 被 **6×6 Conv** 替代（参数量和延迟更优，v5 的 Focus 是在没有 ConvNeXt 之前的选择）。

**C3（CSP Bottleneck ×3）**：

```
        ┌──────────── C3k (3×3) ────────────┐
x ──┬──→│                                    ├──→ concat → 1×1 Conv → out
    │   └────────────────────────────────────┘
    └──────────────── Shortcut ─────────────────┘
```
- 3 个 bottleneck，其中第 1 个用 1×1 conv 把通道减半
- 一路走瓶颈（提取特征），一路走 shortcut（保留原信息），concat 后融合
- **作用**：缓解梯度消失 + 减少重复梯度 + 降低参数

#### Neck：SPPF + PAN

**SPP（Spatial Pyramid Pooling）**：v3 的 SPP（4 个不同尺度的 MaxPool）→ v4 改进版 → v5 的 **SPPF**（3 个连续的 5×5 MaxPool，等效于 9/5/3 三个尺度，更快）

```
        ┌──────────────┐
x ──┬──→│ Conv 1×1     ├──→ concat → Conv 1×1 → out
    │   └──────────────┘
    └──→ Cat([MaxPool5(x), MaxPool5(MaxPool5(x)), MaxPool5(MaxPool5(MaxPool5(x)))])
```

**PAN**：FPN 自顶向下（语义）+ 自底向上（定位）双向路径

#### Head：耦合头（v5）vs 解耦头（v6+/v8）

v5 用**耦合头**（分类和回归共享一条 conv 链，最后分成两个 1×1 输出），参数量小但梯度会互相干扰。

> **历史小知识**：解耦头（Decoupled Head）最早是 YOLOX（2021）引入并证明有效，YOLOv5 在 v3.0 时短暂用过（`YOLOv5` 的 `Detect` 曾实现解耦），但 v5 最终选择耦合头（省参数）。**v6 和 v8 又都回到了解耦头** —— 这说明解耦头的精度收益是真实的。

### 2.3 v5 的损失函数

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{box}}}_{\text{CIoU}} + \underbrace{\underbrace{\mathcal{L}_{\text{obj}}}_{\text{BCE}} + \underbrace{\mathcal{L}_{\text{cls}}}_{\text{BCE}}}_{\text{乘以各自权重}} \times \text{权重}
$$

Ultralytics 实现（`v8DetectionLoss` / `ComputeLoss`）：

```python
# 伪代码：Ultralytics 的 CIoU 计算
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # 1. 转为 ltrb 形式
    px, py, pw, ph = box1
    gx, gy, gw, gh = box2
    dx, dy = (px - gx), (py - gy)
    # 2. 包含关系
    c1 = min(px - pw/2, gx - gw/2), min(py - ph/2, gy - gh/2)
    c2 = max(px + pw/2, gx + gw/2), max(py + ph/2, gy + gh/2)
    # 3. 中心距离
    rho2 = dx**2 + dy**2
    # 4. IoU
    inter = (min(px+pw/2, gx+gw/2) - max(px-pw/2, gx-gw/2)).clamp(0) * \
            (min(py+ph/2, gy+gh/2) - max(py-ph/2, gy-gh/2)).clamp(0)
    union = pw*ph + gw*gh - inter + eps
    iou = inter / union
    if CIoU:
        v = (4 / math.pi**2) * (torch.atan(w2/h2) - torch.atan(h1/h1)).pow(2)
        alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / (c2**2 + eps) + v * alpha)   # CIoU
```

**权重设置**（Ultralytics 默认）：

| 损失项 | 权重 | 说明 |
|---|---|---|
| Box (CIoU) | 0.05 | |
| Obj (BCE) | 1.0（v5.0）/ 0.7（v5.0 之后按 scale 调整） | |
| Cls (BCE) | 0.5 | |

**归一化技巧**：Ultralytics 的 `ComputeLoss` 里，**回归损失对每个 batch 的目标框数量做了归一化**：

```python
target_bboxes /= target_bboxes.shape[0]   # 除以目标框数
```
这让 box loss 的尺度不随 batch 中目标数变化，避免「一张图 100 个目标 vs 一张图 1 个目标」的量级差异。这是很多复现版本漏掉的关键细节。

### 2.4 v5 的标签分配

v5 的正负样本划分（`build_targets`）：

```
对每个 GT (gi, gj, anchor a):
    计算 anchor 与 GT 的 IoU
    if max_iou > ignore_thr (0.7):  → 忽略（既不当正也不当负，损失不计）
    elif max_iou < anchor_t (0.2):  → 背景（负样本）
    else:                            → 正样本
```

**三个 IoU 阈值**：`anchor_t = 0.2`（正样本阈值）、`ignore_thr = 0.7`（忽略阈值）。

**问题**：固定阈值意味着
- 简单数据集：阈值合适
- 困难数据集 / 小目标：GT 和 anchor IoU 天然就低，**小目标可能全部被判为背景** → 训不出来
- 这正是 YOLOv6/v8 换成 **TAL 动态分配**的动机

### 2.5 v5 的模型规格（COCO val, 640×640）

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 参数量 | GFLOPs |
|---|---|---|---|---|
| YOLOv5n | 56.8 | 37.4 | 1.9M | 4.5 |
| YOLOv5s | 63.0 | 45.7 | 7.6M | 16.5 |
| YOLOv5m | 67.5 | 50.1 | 20.1M | 39.0 |
| YOLOv5l | 69.7 | 53.2 | 46.5M | 82.6 |
| YOLOv5x | 71.8 | 55.0 | 86.7M | 141.8 |

> **口径**：Ultralytics v6.0+ README，640 输入，单卡，未用 TTA。如引用请标注版本。

### 2.6 v5 的工程贡献

v5 的价值有一半在**工程体系**：

```
ultralytics/
├── models/
│   ├── common.py       # 全部模块定义
│   ├── yolo.py         # 检测/分割/分类/姿态模型解析
│   ├── experimental.py # 尝试性模块（Sum, MixConv2d, Ensemble）
│   └── tf.py           # TF 版本
├── utils/
│   ├── datasets.py     # 数据集与增强
│   ├── loss.py         # ComputeLoss
│   ├── metrics.py      # ap_per_class, fitness
│   ├── plots.py        # 绘图
│   └── torch_utils.py  # EMA, smart_DDP, fuse_conv_and_bn
├── cfg/
│   ├── models/         # 模型 YAML
│   ├── datasets/       # 数据集 YAML
│   └── hyp/            # 超参 YAML（hyp.scratch-low.yaml / hyp.no-augmentation.yaml 等）
└── train.py / val.py / detect.py / export.py / benchmark.py
```

**最被低估的两个设计**：

1. **`hyp/*.yaml` 分离超参配置**：低算力 / 中算力 / 高算力 / 无增强，四套配置。
2. **`models/yolo.py` 的模型解析机制**：模型结构写在 YAML 里，Python 代码是通用解释器。加一个新模块只需改 YAML + `common.py` 加一个类。

> **v8 继承了这个架构**（`ultralytics` 包），v6/v7/v9 都没有。

---

## 3. v4 vs v5 对比

| 维度 | YOLOv4 | YOLOv5 |
|---|---|---|
| 论文 | ✅ 有 | ❌ 无 |
| 代码质量 | 研究用，可运行 | 工业级 |
| 训练配方 | 部分 | 完整且系统化 |
| 自动 anchor | ❌ | ✅ AutoAnchor |
| 混合精度 | ❌ | ✅ AMP |
| EMA | ❌ | ✅ |
| 部署导出 | 需手写 | ✅ ONNX/TensorRT/TFLite/FPGA 等 |
| 分布式训练 | 需手写 | ✅ DDP |
| 社区生态 | 小 | 大 |
| 结构更新 | CSPDarknet-53 | Focus + C3 + SPPF |
| 多尺度训练 | ✅ | ✅ |

---

## 4. 工程结论

1. **v5 是 YOLO 生态的「事实基线」**。大量第三方工具（各种 ONNX 导出器、量化脚本、RKNN 适配）都只适配 v5 系列。如果你的产线依赖某个工具链，从 v5 起步最省事。
2. **v5 的训练配方可以直接搬到任何检测器上**（这正是 BoF 思想的体现）。如果你用 Faster R-CNN 或 YOLOX，把 v5 的 Mosaic/MixUp/EMA/Cosine LR 拿过去通常就有 1~3 mAP 提升。
3. **v5 的耦合头不是最优的**，v6/v8 的解耦头精度更高。所以「用 v5 结构」和「用 v5 训练配方」是两件事，可以分开取用。
4. **AutoAnchor 的等价物在 v8 已经不需要了**（Anchor-free），但如果你用 v5，务必让它跑一次，并检查生成的 anchor 是否合理（`model.info()` 会打印）。
5. **v5 的 `rect=True` 推理**是容易被忽略的加速点：它把 letterbox 填充去掉，实测对非正方形图像可省 10~30% 推理时间。

---

## 🔗 相关

- 上一个阶段 → [`02-YOLOv1-v3奠基期.md`](./02-YOLOv1-v3奠基期.md)
- 下一个阶段 → [`04-YOLOv6-v10分化期.md`](./04-YOLOv6-v10分化期.md)
- IoU 损失推导 → [`06-YOLO关键模块与算法.md`](./06-YOLO关键模块与算法.md)
- 训练参数细节 → [`07-训练调参与数据增强.md`](./07-训练调参与数据增强.md)
