# YOLO 关键模块与算法

> **核心问题**：YOLO 里那些反复出现的模块（CSP、C3k2、SPPF、DFL、Decoupled Head、IoU 系列），数学上到底在做什么？

本文是 YOLO 部分的**技术内核**，建议配合代码阅读。

---

## 1. 特征提取模块族

### 1.1 Bottleneck

```
x ─→ [1×1 Conv, c→c/2] ─→ [3×3 Conv, c/2→c] ─→ (+x) ─→ out
```

- 1×1 先降通道（省算力），3×3 再升回来
- **残差连接**：$y = \text{Conv}_{3\times3}(\text{Conv}_{1\times1}(x)) + x$
- 参数量：$3\times3$ 卷积在 c 通道上是 $9c^2$，1×1 降维后是 $9\cdot(c/2)^2 = 2.25c^2$，省 4 倍

### 1.2 C3（CSP Bottleneck ×3）

CSP（Cross Stage Partial）的思想：**把特征分成两路**，一路走轻量路径（shortcut），一路走重路径（bottleneck），最后 concat。

```
          ┌──────────────────────────────┐
x ──┬────→│ 1×1 Conv → c/2              │
    │      │   ↓                         │
    │      │ 3×3 Conv → c/2              │
    │      │   ↓                         │
    │      │ 3×3 Conv → c/2              │  ← 3 个 bottleneck
    │      └──────────────────────────────┘
    │                                       │
    └───────────── Shortcut (identity) ────┴──→ Concat → 1×1 Conv → out
```

**C3k2（v11）**：把上面的 3 个 bottleneck 换成 3 个 **C3k**（每个 C3k 内含两个并行 3×3 分支）：

```
C3k (每个 bottleneck 内):
    ┌─→ [3×3 Conv, c_out=c/3] ─┐
    └─→ [3×3 Conv, c_out=c/3] ─┴─→ Concat
```

**为什么 CSP 有效？**
1. **梯度重复**：原始 ResNet 中，梯度会通过多条路径重复回传，导致 "gradient duplication" 问题。切分后，大通道走轻量路径，减少重复梯度
2. **计算/参数比更优**：同样的精度下参数更少
3. **梯度流更顺畅**

### 1.3 RepVGG / EfficientRep（重参数化）

见 [`04-YOLOv6-v10分化期.md`](./04-YOLOv6-v10分化期.md) §1.2。补充完整折叠公式：

**训练态结构**（RepVGG Block）：
```
    ┌─→ 3×3 Conv (pad=1) ─┐
x ──┼─→ 1×1 Conv (pad=0) ─┼─→ Sum → BN → ReLU
    └─→ Identity ──────────┘
```

**折叠步骤**：
1. **Identity → 3×3 Conv**：构造 $I \in \mathbb{R}^{3\times3\times C\times C}$，仅中心点为 1
2. **1×1 Conv → 3×3 Conv**：在 3×3 核中心放置 1×1 核，其余为 0
3. **BN 折叠**：
$$
W_{\text{folded}} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}, \quad
b_{\text{folded}} = \beta - \frac{\gamma \cdot \mu}{\sqrt{\sigma^2+\epsilon}}
$$
4. **多分支求和**：$W_{\text{sum}} = W_3 + W_1 + W_{\text{id}}, \quad b_{\text{sum}} = b_3 + b_1$

**Ultralytics 实现的片段**（`ultralytics/nn/modules/conv.py`）：

```python
class RepVGGBlock(nn.Module):
    """RepVGG block with re-parameterizable conv branches."""
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True):
        super().__init__()
        self.c1, self.c2 = c1, c2
        # 训练分支
        self.rbr_identity = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None
        self.rbr_dense = ConvBNAct(c1, c2, k, s, None, g, d, act=False)
        self.rbr_1x1 = ConvBNAct(c1, c2, 1, s, 0, g, d, act=False)  # 含 BN

    def forward(self, x):
        if self.training:
            y = self.rbr_identity(x) if self.rbr_identity is not None else 0
            return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + y)
        else:
            # 推理态：使用 fuse 的单一分支
            return self.act(self.rbr_dense(x))

    def get_equivalent_kernel_bias(self):
        # 1. 取三个分支的 (weight, bias)
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # 2. 1x1 pad 到 3x3，identity pad 到 3x3
        kernel1x1 = F.pad(kernel1x1, [1, 1, 1, 1])
        kernelid = F.pad(kernelid, [1, 1, 1, 1])
        # 3. 相加
        return kernel3x3 + kernel1x1 + kernelid, bias3x3 + bias1x1 + biasid
```

> **重要**：导出 ONNX 前必须调用 `model.fuse()`（或 `torch.onnx` 时用 `fuse_conv_and_bn`），否则导出的图里有多分支，NPU 上会走慢路径甚至算子不支持。

---

## 2. 颈部结构：SPP / SPPF / PAN

### 2.1 SPP vs SPPF

**SPP（Spatial Pyramid Pooling，v3/v4）**：
```
x → [Conv 1×1] ──┐
                 ├─→ Concat with [MaxPool(5,1), MaxPool(5,2), MaxPool(5,3)] → Conv 1×1
```
- 原始 SPP 用 4 个不同尺寸的池化核（如 5, 9, 13, 19）
- 计算冗余，因为不同核的输出高度重叠

**SPPF（v5 简化版）**：
```
x → [Conv 1×1] ──┐
                 ├─→ Concat with [x, MaxPool5(x), MaxPool5(MaxPool5(x)), MaxPool5(MaxPool5(MaxPool5(x)))]
                 └─→ Conv 1×1
```
用 **3 个串联的 5×5 MaxPool** 替代不同尺寸的池化，等效覆盖 5/9/13 三种感受野，但计算量只有原来的 ~1/2。

> 严格说，串联 5×5 MaxPool 的感受野是 5、9、13（5 + 4 + 4），与原 SPP 的 5/9/13 对齐。

### 2.2 PAN 与 FPN

**FPN（自顶向下）**：把深层语义信息上采样后与浅层融合
```
C5 --1×1--> P5 --↑--> concat with C4 --> P4 --↑--> concat with C3 --> P3
```

**PAN（自底向上，v4 引入）**：再补一条自底向上的路径，把浅层的定位信息往上带
```
P3 --↓--> N3 --↑--> concat with N4 --> N4 --↑--> concat with N5 --> N5
```

**为什么 YOLO 需要 PAN 而不只是 FPN？**
- FPN 只解决「语义信息缺失」，浅层的高分辨率特征没有向上传递
- 小目标主要依赖浅层的强边缘信息，PAN 让浅层特征直达高层
- 实测 PAN 带来 +1~2 mAP

---

## 3. 检测头

### 3.1 耦合头 vs 解耦头

**耦合头（YOLOv5）**：
```
x ─→ [3×3 Conv] ─→ [3×3 Conv] ─┬─→ [1×1 Conv] → 类别 (nc)
                                 └─→ [1×1 Conv] → 边界框 (4×reg_max)
```
- 分类和回归共享特征提取
- **问题**：分类任务和回归任务的梯度会在共享层互相干扰（分类要语义，回归要几何）

**解耦头（YOLOX / v6 / v8+）**：
```
              ┌─→ [3×3 Conv] ─→ [1×1 Conv] → 类别 (nc)
x ────────────┤
              └─→ [3×3 Conv] ─→ [1×1 Conv] → 边界框 (4×reg_max)
```
- 两条分支完全独立
- **代价**：参数量增加约 2 倍（因为多了一组 3×3 Conv）
- **收益**：mAP +1~2，训练更稳定

### 3.2 Anchor-free 回归 + DFL

**v8+ 的回归头**：每个位置 $(i,j)$ 直接预测到 GT 框四条边的距离：

$$
\hat{d} = (d_{\text{left}}, d_{\text{top}}, d_{\text{right}}, d_{\text{bottom}})
$$

```
       d_l  d_t
    ┌───────────┐
    │    ●      │   ● = 网格点 (i,j)
    │           │   d_l = x - x_min
    └───────────┘   d_t = y - y_min
       d_b  d_r
```

**为什么用 DFL（Distribution Focal Loss）？**

直接回归 4 个距离时，**相邻位置的真值是连续变化的**，而回归是逐位置独立学习的 → 位置之间缺乏一致性约束。

**DFL 的做法**：把每个距离离散成 `reg_max` 个 bin（如 16）：

$$
d \in [0, 15], \quad \text{真值 } d = 3 + 0.4 = 3.4
$$

**监督目标**（label smoothing 式的双邻居分布）：
$$
\text{target}(k) =
\begin{cases}
k - d & \text{若 } k = \lfloor d \rfloor \\
d - k & \text{若 } k = \lceil d \rceil \\
0 & \text{其他}
\end{cases}
$$

网络输出 16 个 logits，softmax 后与 target 做交叉熵：

$$
\mathcal{L}_{\text{DFL}} = \text{CE}(\text{softmax}(\hat{l}), \text{target})
$$

**推理时取期望**：
$$
\hat{d} = \sum_{k=0}^{15} k \cdot \text{softmax}(\hat{l})_k
$$

**DFL 的三个作用**：
1. **建模分布而非点估计** → 表达不确定性
2. **相邻 bin 的监督重叠** → 位置间的一致性
3. **精度可调**：`reg_max` 越大，分辨率越细（但计算量线性增长）

> **量化视角**：DFL 需要 softmax（涉及 exp/归一化）+ 加权求和。在 INT8 量化下这两步误差较大，是量化掉点的主要来源。**YOLO26 移除 DFL 后，回归头退化为 4 通道线性输出，量化友好度大幅提升。**

### 3.3 解码（Decode）

从网络输出到最终框坐标：

**v5/v8 之前的 anchor-based 解码**：
$$
\begin{aligned}
x_c &= (\sigma(t_x) + x_{\text{cell}}) \cdot w_a \\
y_c &= (\sigma(t_y) + y_{\text{cell}}) \cdot h_a \\
w &= w_a \cdot e^{t_w} \\
h &= h_a \cdot e^{t_h}
\end{aligned}
$$
然后转成 ltrb。

**v8+ 的 anchor-free 解码**：
$$
\begin{aligned}
x_{\min} &= (i - 0.5 + d_l) \cdot s \\
y_{\min} &= (j - 0.5 + d_t) \cdot s \\
x_{\max} &= (i + 0.5 + d_r) \cdot s \\
y_{\max} &= (j + 0.5 + d_b) \cdot s
\end{aligned}
$$
其中 $s$ 是该特征层的 stride（8/16/32），$i,j$ 是网格坐标。

> **注意**：v8+ 中 `d_l` 等是 `dist2bbox` 的输出，已经乘过 stride。Ultralytics 里的 `make_anchors` 生成网格点并预先乘上 stride。

---

## 4. 标签分配

### 4.1 三种基本范式

| 范式 | 代表 | 判定规则 |
|---|---|---|
| **固定阈值** | YOLOv5 | IoU > 0.2 为正，> 0.7 忽略，其余为负 |
| **TopK + 任务对齐** | TAL（v6/v8） | 按 $s = c^\alpha q^\beta$ 排序取 top-k |
| **一对一** | Hungarian（DETR）/ v10 / v26 | 每个 GT 唯一匹配一个预测 |

### 4.2 ATSS（Adaptive Training Sample Selection）

**论文**：*Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection*, CVPR 2020
**仓库**：`xingyizhou/ATSS`

**问题**：anchor-free 检测器面临「正样本太少」的问题 —— 一张图可能只有 3 个 GT，但有 8400 个候选位置，选哪些当正样本？

**ATSS 的自适应选点**：

```
Step 1: 候选生成
        对每个 GT 中心附近的 K 个位置（K=9）作为候选
Step 2: 统计交并比
        对每个候选位置，计算其所有预测框与 GT 的 IoU，
        取最大值 q_i，并记录最大值对应的 anchor 索引 j_i
Step 3: 筛选候选
        - 统计每个 anchor 索引 j 被选为最大值候选的次数
        - 取 top-k 的 anchor 索引（k=2），形成候选集合 A
        - 只保留在 A 中、且 IoU > 阈值(0.5) 的候选
Step 4: 中心化
        候选位置减去 GT 中心，得到相对偏移
Step 5: 分配
        候选位置按偏移量排序，与排序后的 GT 一一对应
```

**核心创新**：**正样本的选择依赖预测质量**，而不是依赖 IoU 阈值。预测得准的位置（IoU 最高的 anchor）更容易被选中，预测差的位置被自然淘汰。

### 4.3 TAL（Task-Aligned Assigner）

见 [`04-YOLOv6-v10分化期.md`](./04-YOLOv6-v10分化期.md) §1.3。补充完整流程（对应 Ultralytics `TaskAlignedAssigner`）：

```python
# ultralytics/utils/tal.py (简化)
def assign(scores, gt_bboxes, anc_points, gt_labels, mask_gt, bg_idx, topk=10, alpha=1.0, beta=6.0):
    # 1. 计算所有 anchor 与 GT 的 IoU 矩阵
    mask_pos = scores.new_zeros(scores.shape)          # (num_anchor, num_gt)
    align_metric = scores.new_zeros(scores.shape)
    # overlap = IoU(anchor, gt)  shape: (num_anchor, num_gt)
    for gt_idx in range(num_gt):
        # 任务对齐分数：分类得分^alpha * IoU^beta
        scores_per_gt = scores[:, gt_idx].clone().masked_fill_(~mask_gt[gt_idx], 0)
        align_metric[:, gt_idx] = scores_per_gt.pow(alpha) * overlap[:, gt_idx].pow(beta)
        # topk 候选
        topk_indices = torch.topk(scores_per_gt, min(topk, num_anchor)).indices
        mask_pos[topk_indices, gt_idx] = 1.0

    # 2. 一个 anchor 被多个 GT 争抢 → 保留 align_metric 最大的
    mask_pos_mask = mask_pos.sum(-1)
    if mask_pos_mask.max() > 1:
        # 冲突的 anchor 位置
        for gt_idx in range(num_gt):
            # 把非最优 GT 的 mask 清零（每个 anchor 只归给一个 GT）
            mask_pos[:, gt_idx] *= (align_metric[:, gt_idx] == align_metric.max(dim=1)[0])
        mask_pos = mask_pos.sum(-1)  # 归一化

    # 3. 剩余 anchor：IoU > ignore_thr 的忽略，其余为负
    pos_idx = mask_pos.bool()
    keep = overlap.max(dim=1)[0] > ignore_thr     # ignore_thr = 0.7
    pos_idx[keep] = False
    # 负样本（IoU < bg_thr = 0.2）
    neg_idx = (overlap.max(dim=1)[0] < bg_thr).nonzero().flatten()
    return pos_idx, neg_idx
```

**关键超参**：`alpha=0.5`, `beta=6.0`, `topk=10`。

### 4.4 SimOTA

**论文**：*YOLOX: Exceeding YOLO Series in 2021*, arXiv:2107.08430

与 TAL 类似的动态分配，区别在于**每个 GT 的候选数量自适应**：

$$
k_{gt} = \max\left(1, \left\lfloor \frac{\text{前景候选数}}{\text{GT 数}} \right\rfloor \right)
$$
其中「前景候选数」= IoU > 阈值(0.5) 且分类得分 > 阈值(0.1) 的 anchor 数。

**自适应 top-k** 的意义：GT 密集时每个 GT 分到的候选少，GT 稀疏时每个 GT 分到的候选多。

### 4.5 三种分配器的对比

| | ATSS | TAL | SimOTA |
|---|---|---|---|
| 候选生成 | GT 中心附近 top-k 位置 | 全局按对齐分数 top-k | 全局按对齐分数 top-k |
| 分配数量 | 固定 | 固定（10） | **自适应** |
| 对齐依据 | IoU 最高的 anchor 索引 + 中心距离 | $c^\alpha q^\beta$ | $c^\alpha q^\beta$ |
| 冲突仲裁 | 位置排序一一配对 | 取对齐分最高 | 取对齐分最高 |
| 被谁采用 | FCOS 后续、MMDetection | YOLOv6、**YOLOv8+** | YOLOX |

---

## 5. IoU 损失家族

### 5.1 从 IoU 到 CIoU

设预测框 $B$ 和真值框 $B^g$，交集面积 $I$，并集面积 $U$，中心距离 $d^2$，最小包围框对角线长 $c^2$。

**IoU**：
$$
\text{IoU} = \frac{I}{U}, \quad \mathcal{L}_{\text{IoU}} = 1 - \text{IoU}
$$
**问题**：不相交时 IoU 恒为 0，梯度为 0 → **无法优化「先靠近」的过程**。

**GIoU**：
$$
\text{GIoU} = \text{IoU} - \frac{|C \setminus (B \cup B^g)|}{|C|}, \quad
\mathcal{L}_{\text{GIoU}} = 1 - \text{GIoU}
$$
其中 $C$ 是最小包围框。**解决了不相交时无梯度的问题**。

**DIoU**：
$$
\text{DIoU} = \text{IoU} - \frac{d^2}{c^2}, \quad
\mathcal{L}_{\text{DIoU}} = 1 - \text{DIoU}
$$
**惩罚中心距离**，收敛更快。

**CIoU**：
$$
\text{CIoU} = \text{IoU} - \frac{d^2}{c^2} - \alpha v, \quad
\alpha = \frac{v}{(1-\text{IoU}) + v}, \quad
v = \frac{4}{\pi^2}\left(\arctan\frac{w^g}{h^g} - \arctan\frac{w}{h}\right)^2
$$
$v$ 惩罚**宽高比**不一致。

**为什么 v8+ 仍用 CIoU 而不是 DIoU/CIoU 的后继？**
因为 DFL 已经提供了「宽高比一致性」的隐式约束（相邻 bin 的重叠监督），CIoU 的 v 项边际收益变小。

### 5.2 后续变种

| 名称 | 公式要点 | 特点 |
|---|---|---|
| **EIoU** | 直接惩罚宽高差：$\text{IoU} - \frac{d^2}{c^2} - \frac{\rho^2(w,w^g)+\rho^2(h,h^g)}{c_w^2+c_h^2}$ | 解耦宽高，收敛快 |
| **SIoU** | 引入**角度代价** $A = \sin^2(\Delta\theta)$ | 考虑方向 |
| **Alpha-IoU** | 用幂函数 $\text{IoU}^\alpha$ 重加权 | 超参少 |
| **WIoU v1/v2/v3** | 引入动态的**非单调聚焦机制** | v3 用注意力估计 IoU 质量 |
| **MPDIoU** | 同时惩罚中心距、宽高差、最小包围盒 | 综合性能好 |

### 5.3 IoU 阈值与 mAP 的关系

**关键理解**：mAP@0.5:0.95 是 **10 个 IoU 阈值**（0.50:0.05:0.95）的平均：

$$
\text{AP} = \frac{1}{10}\sum_{t \in \{0.5, 0.55, ..., 0.95\}} \text{AP}(t)
$$

**含义**：一个模型要在**很宽松**（0.5，框稍微偏也能算对）到**很严格**（0.95，框几乎完美）都表现好。

**工程含义**：
- 工业界很多场景只关心 IoU@0.5（框的位置对了就行）
- 但 mAP@0.5:0.95 能反映**定位精度**，在高 IoU 阈值下拉开差距
- 训练时用 CIoU 而不是 IoU，就是为了提升高 IoU 阈值下的表现

---

## 6. 完整的数据流（以 YOLO11 为例）

```
输入 640×640×3
  │
  ├─→ [Conv 6×6 s2]      → 320×320×64
  ├─→ [C3k2]              → 320×320×64
  ├─→ [Conv 3×3 s2]      → 160×160×128
  ├─→ [C3k2]              → 160×160×128
  ├─→ [Conv 3×3 s2]      →  80×80×256
  ├─→ [C3k2]              →  80×80×256
  ├─→ [Conv 3×3 s2]      →  40×40×512
  ├─→ [C3k2]              →  40×40×512
  ├─→ [Conv 3×3 s2]      →  20×20×1024
  ├─→ [C3k2]              →  20×20×1024    ← C5 (P5)
  ├─→ [SPPF]              →  20×20×1024
  │
  ├─→ [C2PSA]            ← 仅最后一层（v11 特有）
  │
  └──────────────────────────── Head
       P3 (80×80)   ← 上采样 + concat C3
       P4 (40×40)   ← 上采样 + concat C4
       P5 (20×20)   ← SPPF 输出
       
       每个尺度: 解耦头
         分类分支 → 1×1 Conv → 80 维 (COCO 80 类)
         回归分支 → 1×1 Conv → 64 维 (4 边 × 16 bins)
```

---

## 7. 工程结论

1. **DFL 是理解 YOLO8-25 检测头的钥匙**。不理解「离散分布 + 期望」，就看不懂 v8 的回归头输出为什么是 64 维而不是 4 维。
2. **量化部署优先选无 DFL 的模型**（YOLO26）。DFL 的 softmax + 加权求和在 INT8 下误差显著。
3. **重参数化模块导出前必须 fuse**。不 fuse 的话 NPU 上多分支会走慢路径。
4. **TAL 的 alpha/beta 值得调**。beta=6.0 意味着 IoU 主导；如果你的场景是「框位置不敏感但类别要准」（如货架盘点），可以调低 beta、调高 alpha。
5. **标签分配的调参收益 > 主干网络的换代收益**。同样的主干换分配器，动辄 +1~2 mAP。

---

## 🔗 相关

- 模块演进背景 → [`04-YOLOv6-v10分化期.md`](./04-YOLOv6-v10分化期.md)
- 损失全景 → [`../02-视觉检测模型/11-损失函数与样本分配.md`](../02-视觉检测模型/11-损失函数与样本分配.md)
- 样本分配理论 → [`../02-视觉检测模型/04-无锚框与样本分配.md`](../02-视觉检测模型/04-无锚框与样本分配.md)
