# LoRA：低秩适配的原理、实现与变体

> **核心问题**：LoRA 为什么成立？rank 和 alpha 怎么定？为什么合并后推理零开销？它和 QLoRA / DoRA / PiSSA 差在哪？

**论文**：*LoRA: Low-Rank Adaptation of Large Language Models* — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. arXiv:2106.09685, ICLR 2022.
**官方实现**：https://github.com/microsoft/LoRA （现已并入 HuggingFace PEFT）

---

## 1. 问题：为什么需要 PEFT

### 1.1 全参微调的三个瓶颈

| 瓶颈 | 具体表现 |
|---|---|
| **显存** | 7B 全参 SFT 需 ~112 GB 优化器/梯度/参数 + 激活，单卡放不下（详见 [`02-微调.md`](./02-微调.md) §2.1） |
| **存储** | 每个任务一份完整权重（7B ≈ 14 GB bf16），10 个任务 = 140 GB |
| **切换** | 多任务无法在同一份权重上并存，只能反复拷贝、覆盖 |

### 1.2 一个关键的观察

微调时，**权重的改变量 $\Delta W$ 通常是低秩的**。

$$
W = W_0 + \Delta W, \qquad \text{rank}(\Delta W) \ll \min(d_{\text{in}}, d_{\text{out}})
$$

依据是 SVD：$\Delta W = U\Sigma V^\top$，其中奇异值衰减很快（少数几个方向承载了绝大部分变化）。这给了两个推论：

1. $\Delta W$ 可以用两个瘦矩阵相乘表示，参数量从 $d_{\text{in}} \cdot d_{\text{out}}$ 降到 $r(d_{\text{in}} + d_{\text{out}})$。
2. **不更新 $W_0$ 本身**，天然缓解灾难性遗忘（改动幅度受限）。

---

## 2. 核心机制

### 2.1 前向传播

LoRA 的做法：冻结预训练权重 $W_0$，只训练两个小矩阵 $A, B$，把 $\Delta W$ 分解为：

$$
\Delta W = \frac{\alpha}{r} B A,
\qquad
A \in \mathbb{R}^{r \times d_{\text{in}}},\quad
B \in \mathbb{R}^{d_{\text{out}} \times r},\quad
r \ll \min(d_{\text{in}}, d_{\text{out}})
$$

完整的前向：

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

```
        x  (d_in)
        │
        ├──[ W0 ] (冻结) ──────────────┐
        │                                │
        │      [A] (训练) → [B] (训练)   │
        │       r×d_in     d_out×r       │
        │         │            │         │
        └───────→(×α/r)──────→(+)────────┴──→ h (d_out)
```

**参数量对比**（单个线性层）：

| 方案 | 可训练参数 | 相对全参 |
|---|---|---|
| 全参 | $d_{\text{in}} d_{\text{out}}$ | 100% |
| LoRA (r=8) | $r(d_{\text{in}} + d_{\text{out}})$ | 例：$4096\times4096$ 时为 $65{,}536 / 16{,}777{,}216 \approx 0.39\%$ |

### 2.2 初始化：保证起点等于预训练模型

$$
A \sim \text{Kaiming} \;(\text{或高斯}), \qquad B = 0
$$

因为 $B = 0$，所以初始时 $BA = 0$，即 $\Delta W = 0$：

$$
h_{\text{init}} = W_0 x
$$

**这意味着训练刚开始时模型的输出与预训练模型完全一致**，不会因为随机初始化而破坏已有能力。梯度方面：

$$
\frac{\partial \mathcal{L}}{\partial B} \neq 0 \quad (\text{因为 } A \neq 0), \qquad
\frac{\partial \mathcal{L}}{\partial A} = 0 \text{ at init} \quad (\text{因为 } B = 0)
$$

所以 $B$ 先动、$A$ 后动——这正是我们想要的（先学「往哪个方向改」，再学「怎么改」）。

> **易错点**：**必须 B=0、A 随机**。反过来（A=0, B 随机）会导致 $\Delta W = 0$ 但 $\partial\mathcal{L}/\partial A = 0$，$A$ 永远不动，训练静默失败。

### 2.3 $\alpha$ 与 $r$ 的关系

**有效缩放系数是 $\alpha / r$，不是 $\alpha$**。

$$
\text{effective\_scale} = \frac{\alpha}{r}
$$

这带来一个重要推论：**改变 $r$ 时，如果不按比例改 $\alpha$，有效学习率就变了。**

| 配置 | $\alpha/r$ | 含义 |
|---|---|---|
| r=8, alpha=8 | 1.0 | 基准 |
| r=8, alpha=16 | 2.0 | 常见配置（alpha = 2r） |
| r=16, alpha=16 | 1.0 | 与上表第一行等效缩放，但容量更大 |
| r=16, alpha=32 | 2.0 | 与上表第二行等效 |
| r=64, alpha=128 | 2.0 | 容量更大，缩放不变 |

**实践约定**：**固定 $\alpha = 2r$**（或固定 $\alpha/r$ 常数），这样调 $r$ 时只改容量、不改有效学习率，消融实验才干净。

> 这也是 **rsLoRA** 要解决的问题：在 $\alpha/r$ 缩放下，rank 增大时性能反而下降（见 §4.3）。

### 2.4 推理：合并回基座，零开销

训练完成后，adapter 可以直接吸收进基座权重：

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

合并后模型结构与原模型**完全相同**，因此：

- ✅ **推理延迟零增加**（这是 LoRA 相对 Adapter / Prefix-Tuning 的决定性优势）
- ✅ 可直接导出 ONNX / TensorRT / GGUF，走原有部署链路
- ✅ 量化、KV cache 优化等全部照常生效

**不合并**也能用（base + adapter 两份权重），适合「一个基座挂多个任务」的场景，但：

- 推理多一次矩阵乘（q/k/v/o 上的额外小矩阵），延迟增加约 10~30%（取决于模型规模和硬件）
- 显存要同时装 base 和 adapter

---

## 3. 论文的核心实证

### 3.1 关键结论（论文报告值）

| 结论 | 数字 |
|---|---|
| 可训练参数减少 | **10,000×**（GPT-3 175B，r=8，可训练 ~37.7M 参数） |
| GPU 显存减少 | **3×** |
| 推理延迟 | **不增加**（合并后） |
| 微调 175B 的硬件门槛 | **单张 80GB GPU** |
| GPU 小时数 | 从全参微调的 **4,000+ 降到 100** |
| MT-Bench（GPT-3 175B） | LoRA **6.31** vs 全参微调 **6.13** vs 未微调 **4.30**（GPT-4 约 7.34） |

> ⚠️ 以上均为**论文报告值**，本仓库未复现。引用前请回原文核对表格（尤其是 MT-Bench 的具体版本与 judge 模型）。

### 3.2 论文的三个消融结论（比主结论更重要）

| 消融 | 结论 |
|---|---|
| **rank 扫描**（r = 4/8/16/32/64/128） | **r = 8 已经饱和**，继续增大收益很小；**r 越大反而在某些设置下略微下降**（引出 rsLoRA） |
| **target modules** | 在**更多**的矩阵上加 LoRA，效果提升**大于**在同样的矩阵上用**更大的 rank** |
| **学习率** | LoRA 需要比全参微调**高 1~2 个数量级**的学习率（论文用 1e-4 ~ 3e-4，全参用 ~1e-5） |

> **工程结论**：**「加更多 target modules」优先于「加大 rank」。** 想要效果，先把 `q,k,v,o` 补全，再考虑加 FFN，最后才考虑提 rank。

---

## 4. 变体与改进

### 4.1 QLoRA（2023，NeurIPS）

**论文**：*QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers, Besnova, Pagnoni. arXiv:2305.14314.

**三个核心技术**：

| 技术 | 作用 |
|---|---|
| **NF4**（4-bit NormalFloat） | 信息论最优的数据类型：把正态分布的量化 bin 按分位数分配，而非均匀分配。量化后的权重噪声接近原值的分布 |
| **Double Quantization** | 对量化常数本身再量化（从 fp32 压到 8bit），再省约 0.4 bit/param |
| **Paged Optimizers** | 用 NVIDIA unified memory 管理优化器状态，避免显存碎片导致的 OOM |

$$
\text{NF4}: \quad \text{bin 边界取自标准正态分布的分位数}
$$

正态分布的绝大部分质量集中在中间区域，所以中间 bin 密、两端 bin 疏——用 4 bit 就能比均匀量化保留更多信息。

**论文报告值**：

- **单张 48GB GPU 微调 65B 模型**
- 在 4bit 量化基座上微调，**质量不低于 16bit 全参微调**
- **单张 RTX 4090 上微调 65B 用时 < 24 小时**

**注意 QLoRA 的两个常被误解的点**：

1. **推理时仍要 4bit 量化**（除非合并——但 4bit 基座 + bf16 adapter 合并后会掉精度，通常不合并）。
2. **NF4 + 双量化是训练时的量化**，不是「把模型压小」——你得到的是一个 4bit 基座 + 少量 bf16 adapter。

### 4.2 DoRA（2024，ICML）

**论文**：*DoRA: Weight-Decomposed Low-Rank Adaptation* — Liu et al. arXiv:2402.09353.

**观察**：LoRA 更新了权重的**方向**（direction），但没更新**模长**（magnitude）。论文发现模型微调时经常需要「单纯放大/缩小某些权重」——这是 LoRA 做不到的。

**做法**：把每个权重列拆成模长 $m$ 和方向 $\hat{W}$，分别学习：

$$
W = m \circ \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

其中 $\|\cdot\|_c$ 是按列的 Frobenius 范数，$\circ$ 是逐元素相乘。

**结论（论文报告值）**：**DoRA 显著优于 LoRA**，在相同或更少的可训练参数下，尤其在小 rank 时优势明显（r=8 时差距最大）。论文也指出 DoRA 与 PiSSA 可以叠加使用。

### 4.3 rsLoRA（2023）

**论文**：*A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* — Kalajdzievski. arXiv:2312.03732.

**问题**：$\alpha/r$ 缩放下，rank 增大时性能反而下降。

**假设的原因**：随着 $r$ 增大，单个分量的幅度 $\propto \frac{1}{r}$ 变小，为了维持总更新幅度，需要更强的正则——等效于对合并权重的 $\ell_2$ 惩罚随 $r$ 增大而增强。

**解法**：把缩放改成

$$
\frac{\alpha}{\sqrt{r}} \quad \text{而不是} \quad \frac{\alpha}{r}
$$

**结论（论文报告值）**：rsLoRA 在**大 rank** 时明显优于 LoRA，在小 rank（r ≤ 16）与 LoRA 相当。

> **实践**：如果你的 rank ≥ 32 且发现效果不升反降，试试 rsLoRA。

### 4.4 PiSSA（2024，NeurIPS）

**论文**：*PiSSA: Principal Singular Values and Singular Vectors Adaptation* — Meng et al. arXiv:2404.02948.

**想法**：既然 $\Delta W$ 低秩，那就**直接用 $W_0$ 的主奇异值/奇异向量来初始化** adapter，让 adapter 从一开始就在「最重要的方向」上调整：

$$
W_0 = U\Sigma V^\top \;\longrightarrow\;
\underbrace{A = V_r^\top}_{\text{初始化}},\quad
\underbrace{B = U_r \Sigma_r}_{\text{初始化}},\quad
W_{\text{residual}} = W_0 - \frac{\alpha}{r}BA
$$

**结论（论文报告值）**：PiSSA 收敛更快、最终 loss 更低，且与 DoRA 兼容。

**代价**：初始化需要做 SVD（对大矩阵有成本，且需要 fp32 计算）。

### 4.5 LoRA+（2024，TMLR）

**论文**：*LoRA Learns Less and Forgets Less* — Hayou et al. arXiv:2402.12354.

**观察**：$A$ 和 $B$ 的梯度幅度差异很大。论文发现**给 $B$ 更大的学习率**收敛更快、效果更好：

$$
\text{lr}_B = \eta \cdot \text{lr}_A, \qquad \eta \approx 16
$$

**附带结论（值得注意）**：LoRA 的「遗忘」比全参微调少，可能是因为其隐式的更新幅度限制起了正则作用。

### 4.6 AdaLoRA（2023，ICLR）

**论文**：*Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning* — Zhang et al. arXiv:2303.10512.

**想法**：不是所有层都需要同样的 rank。让每个 adapter 有一个**重要性分数**（基于奇异值），把全局的 rank 预算动态分配给重要的层/模块，不重要的直接置零。

**适合**：多任务 adapter 训练、层间重要性差异大的场景。

### 4.7 VeRA（2023，ICLR）

**论文**：*Vector-based Random Matrix Adaptation* — Kopiczko et al. arXiv:2310.11454.

**极致压缩**：$A$、$B$ 固定为**共享的随机矩阵**（所有层共享，训练中不更新），只训练中间的 $r \times r$ 矩阵：

$$
\Delta W = \Lambda_b B \Lambda_a A
$$

**可训练参数比 LoRA 再少 100×**，在多任务场景下存储优势明显。

### 4.8 同期的重要邻居：GaLore（2024，ICML）

**论文**：*GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection* — Zhao et al. arXiv:2403.03507.

**注意**：GaLore **不是** LoRA 家族——它是**优化器侧**的方法：把梯度投影到低秩子空间后再做优化器更新。它和 LoRA 是**正交**的，可以叠加使用。

**核心思想**（同样是「更新量低秩」的洞察，但作用在优化过程而非权重表示上）：

$$
g_t \rightarrow P g_t, \qquad P = U U^\top \quad (U \text{ 由 } SVD \text{ 周期性更新})
$$

低秩投影让优化器状态（Adam 的 $m, v$）也只需要低秩存储，从而**把优化器显存从 O(N) 降到 O(Nr)**，可以全参微调 7B~30B。

### 4.9 变体对比速查

| 变体 | 年份 | 一句话 | 什么时候用 |
|---|---|---|---|
| **LoRA** | 2021 | 基准方案 | 默认选择 |
| **QLoRA** | 2023 | 4bit 基座 + LoRA | 显存不够 / 想训更大的模型 |
| **rsLoRA** | 2023 | $\alpha/\sqrt{r}$ 缩放 | rank ≥ 32 且效果不升反降 |
| **AdaLoRA** | 2023 | 秩预算自适应分配 | 多任务 / 层间差异大 |
| **VeRA** | 2023 | 共享随机矩阵，超省参数 | 任务极多、存储敏感 |
| **DoRA** | 2024 | 模长+方向分解 | 想超越 LoRA 且预算允许 |
| **PiSSA** | 2024 | 主奇异值初始化 | 想要更快收敛 |
| **LoRA+** | 2024 | $B$ 用更大 lr | 想榨干收敛速度 |
| **GaLore** | 2024 | 梯度低秩投影（优化器侧） | 想**全参**微调但显存不够 |

---

## 5. 实操

### 5.1 PEFT 标准用法（推荐）

```python
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ---------- 1. 加载基座 ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# decoder-only 模型批量生成时必须左 padding；训练时右 padding 即可
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # QLoRA：4bit 加载
    bnb_4bit_quant_type="nf4",              # NF4 数据类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算用 bf16
    bnb_4bit_use_double_quant=True,         # 双量化
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,          # 纯全参微调时删掉这一行
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ---------- 2. 包装成 LoRA ----------
lora_config = LoraConfig(
    r=16,                                   # 秩
    lora_alpha=32,                          # 缩放系数，建议 = 2*r
    lora_dropout=0.05,
    bias="none",                            # 不训练 bias
    task_type="causal_lm",
    target_modules=[                        # 关键：名字必须和模型里的模块名完全一致
        "q_proj", "k_proj", "v_proj", "o_proj",
        # 想更好可以加上 FFN（显存够时强烈建议）：
        # "gate_proj", "up_proj", "down_proj",
    ],
)

model = prepare_model_for_kbit_training(model)   # 仅 QLoRA 需要
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例：trainable params: 40,894,464 || all params: 7,615,616,512 || trainable%: 0.54
```

**检查 `target_modules` 名字是否匹配**（最常见的失败原因）：

```python
import re
names = set()
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        names.add(n.split(".")[-1])
print(sorted(names))
# LLaMA/Qwen/Mistral: ['down_proj','gate_proj','k_proj','o_proj','q_proj','up_proj','v_proj']
# GPT-2:           ['c_attn','c_proj']  ← 完全不同的命名！
```

> **避坑**：**不同模型家族的模块命名完全不同**（LLaMA 系是 `q_proj`，GPT-2 是 `c_attn`，而 `c_attn` 是 QKV 合并的）。写错名字 → `print_trainable_parameters()` 显示 0 个可训练参数 → 训练「成功」但模型没变。**每次都检查这个输出。**

**训练与保存**：

```python
# 数据集需要提供 input_ids / attention_mask / labels（labels 里已把 prompt 部分置 -100）
args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,      # 有效 batch = 4*8*卡数
    num_train_epochs=2,
    learning_rate=2e-4,                 # LoRA 用 1e-4 ~ 3e-4
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",           # QLoRA 常用
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),  # 因果 LM 不要 mlm
)
trainer.train()

# 只保存 adapter（几十 MB），不保存完整权重
model.save_pretrained("./out/adapter")
tokenizer.save_pretrained("./out/adapter")
```

**加载与合并**：

```python
from peft import PeftModel

# 方式 A：基座 + adapter（可多任务并存）
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "./out/adapter")

# 方式 B：合并（推理推荐，零开销）
merged = model.merge_and_unload()
merged.save_pretrained("./out/merged")
```

### 5.2 自己实现一遍 LoRA（理解用）

```python
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """包装一个 nn.Linear，加装 LoRA adapter。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():      # 冻结基座
            p.requires_grad = False

        self.r, self.alpha = r, alpha
        self.scaling = alpha / r              # 有效缩放

        d_in, d_out = base.in_features, base.out_features
        # A: (r, d_in)  Kaiming 初始化；B: (d_out, r) 全零
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        # BA x 等价于先过 A 再过 B
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self):
        """把 adapter 吸收进基座权重，推理零开销。"""
        delta = (self.scaling * (self.lora_B @ self.lora_A)).to(self.base.weight.dtype)
        self.base.weight += delta
        return self.base


# 快速自检：初始化后应与原层输出完全一致
if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(512, 512)
    x = torch.randn(4, 512)
    ref = base(x)

    lora = LoRALinear(base, r=8, alpha=16)
    assert torch.allclose(ref, lora(x), atol=1e-5), "初始化后输出应与基座一致"

    # 合并后仍应一致（合并的是当前 adapter 的值，此时为 0）
    assert torch.allclose(ref, lora.merge()(x), atol=1e-5)
    print("OK: LoRA 初始化与合并均正确")
```

**这段代码验证了两个关键性质**：初始化输出等于基座；合并是等价的。

### 5.3 显存估算

**LoRA 训练的显存构成**（相比全参少了什么）：

| 项 | 全参 | LoRA |
|---|---|---|
| 参数（bf16） | 14 GB | 14 GB（冻结） |
| 梯度 | 14 GB | **≈ 0.1 GB**（只存 adapter 梯度） |
| AdamW 状态 | 84 GB | **≈ 0.3 GB** |
| 激活 | 10~40 GB | 10~40 GB（可开 gradient checkpointing 降） |

**结论**：**LoRA 训练的显存瓶颈变成了「基座 + 激活」，而不是「优化器状态」。** 这就是为什么 LoRA 能在单张 24GB 卡上跑 7B。

**QLoRA 再进一步**：基座从 14 GB 压到 ~3.5 GB（4bit），**但激活值仍是 bf16 的量级**——所以 QLoRA 训练 7B 时激活往往才是瓶颈，开 gradient checkpointing 收益很大。

### 5.4 各规模的显存参考（7B，LoRA，QLoRA）

| 配置 | 显存需求 | 备注 |
|---|---|---|
| 全参 bf16 + AdamW | ~112 GB + 激活 | 需 2×80GB + ZeRO |
| LoRA bf16（不开 checkpointing） | ~18~25 GB | 24GB 卡可跑 |
| LoRA bf16 + gradient checkpointing | ~12~16 GB | |
| QLoRA 4bit + gradient checkpointing | ~8~12 GB | 12GB 卡的甜点配置 |
| QLoRA 4bit 训 13B | ~16~22 GB | |
| QLoRA 4bit 训 70B | ~40~48 GB | 论文报告单张 48GB 可训 |

> ⚠️ 激活显存随 **序列长度 × batch** 快速上升。2048 token 的 seq_len 相比 512 大约多 4 倍激活。上表按 seq_len ≈ 1024~2048 估算。

---

## 6. 常见坑清单

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | **`target_modules` 名字不匹配** | `trainable params: 0` | 先打印 `named_modules()` 确认 |
| 2 | **训练完模型没变化** | loss 几乎不动 | 同上；或忘了 `model.train()` / `model.enable_input_require_grads()` |
| 3 | **聊天模板不匹配** | 输出格式乱 | 用 `apply_chat_template`，确认 base/chat 模型 |
| 4 | **合并前忘了 `model.eval()`** | 推理结果带 dropout 噪声 | `merge_and_unload()` 前先 `model.eval()` |
| 5 | **合并时 dtype 不一致** | 合并后精度骤降 | 合并前把 base 转成与 adapter 相同的 bf16/fp32 |
| 6 | **alpha 不是 2r** | 调 r 时性能反常 | 固定 $\alpha/r$ 或固定 $\alpha = 2r$ |
| 7 | **QLoRA 后直接合并** | 精度明显下降 | 4bit 基座不要合并，保留 base+adapter 形式推理 |
| 8 | **labels 没有 mask prompt** | 提升有限 | 手工构造 prompt+response，prompt 部分置 -100 |
| 9 | **padding side 错** | 批量推理输出异常 | decoder-only 批量生成用左 padding |
| 10 | **在 base 模型上用 chat 模板** | 模型不按指令说话 | base 模型用 base 模板，先做 SFT 再谈对话 |
| 11 | **忘了 `prepare_model_for_kbit_training`** | QLoRA 训练报错/极慢 | 4bit 路径必须先调用 |
| 12 | **新增的层没训到** | 新加的 head 随机 | `modules_to_save=["lm_head"]` 等 |
| 13 | **多 adapter 串行加载不释放** | 显存爆 | `delete adapter; torch.cuda.empty_cache()` |
| 14 | **拿 merge 后的模型和 merge 前的分数比** | 差异难解释 | 两条路径都测一遍，记录数值差 |

---

## 7. 怎么选：LoRA 还是别的

| 你的约束 / 目标 | 选择 |
|---|---|
| 默认情况 | **LoRA (r=16, alpha=32, q/k/v/o)** |
| 显存非常紧 / 模型更大 | **QLoRA**（NF4 + 双量化 + paged optimizer） |
| LoRA 已到瓶颈、想再涨 | 加 FFN target modules → 提 rank → **DoRA** / **PiSSA** |
| 想**全参**微调但显存不够 | **GaLore**（优化器侧低秩）+ ZeRO-3 |
| 一个基座要挂几十个任务 | **VeRA** / AdaLoRA（存储极省） |
| rank ≥ 32 且效果不升反降 | **rsLoRA** |
| 要极致收敛速度 | **LoRA+**（调大 B 的 lr） |
| 通用能力不能掉 | **LoRA**（低秩天然正则）+ 混入通用数据 |
| 任务与预训练分布差异极大 | 继续预训练（不是 LoRA 的活）→ [`01-预训练.md`](./01-预训练.md) |

---

## 8. 工程结论

1. **LoRA 的核心不是「省显存」，是「不改基座」。** 低秩约束既省显存，也天然抗遗忘，还让多任务共存成为可能。
2. **优先加 target modules，其次提 rank。** 论文的消融结论非常明确。
3. **固定 $\alpha = 2r$**，让调 rank 不会顺带改学习率。
4. **推理前合并**，把延迟开销彻底消掉，保持原有部署链路不变。
5. **每次都检查 `print_trainable_parameters()`。** 「训练成功但模型没变」是这个工具最高频的静默失败。
6. **QLoRA 的瓶颈是激活值不是基座**，开 gradient checkpointing 收益最大。
7. **LoRA 有天花板。** 如果它涨不动，先怀疑数据和目标模块，最后才怀疑方法本身。

---

## 🔗 相关

- 微调全景（全参 vs 高效、对齐方法） → [`02-微调.md`](./02-微调.md)
- 预训练阶段（数据、算力、Scaling Law） → [`01-预训练.md`](./01-预训练.md)
- 本目录索引 → [`README.md`](./README.md)
- 边缘侧量化与部署 → [`../04-边缘部署与端侧AI/README.md`](../04-边缘部署与端侧AI/README.md)
