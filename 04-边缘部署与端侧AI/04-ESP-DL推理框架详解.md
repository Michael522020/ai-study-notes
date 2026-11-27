# ESP-DL 推理框架详解

> **一句话：ESP-DL 不是「能在 ESP32 上跑 AI」这么简单，它是乐鑫把
> 「模型格式 + 算子实现 + 量化工具 + 内存规划」四件事绑成的一套契约。**
> 想自定义模型落地，这套契约的每个条款都会变成你的约束。

> 🔎 **本篇事实核对自本机 ESP-DL 源码副本 `E:\p4work\esp-dl`：
> tag `v3.3.15`，commit `512ed0b`（2026-11-06）。**
> 涉及的性能数字**均为官方仓库中的公开报告值**，本机未逐一复现。
> 与栈级速览见 [`02-ESP32端侧视觉栈-ESP-VISION与ESP-DL.md`](./02-ESP32端侧视觉栈-ESP-VISION与ESP-DL.md)。

---

## 0. 先问：ESP-DL 值不值得用

ESP32 上能跑 AI 的选项不止一个，先把选择摆清楚：

| 方案 | 上手成本 | 模型自由度 | 官方模型生态 | 适合 |
|---|---|---|---|---|
| **ESP-VISION**（MicroPython） | 极低 | 低（模型要转 `.espdl` + sidecar JSON） | 有（但不含农业模型） | 快速验证、原型、零工具链 |
| **ESP-DL**（C++ 组件） | 中 | **高**（算子可自己写、量化可自己调） | 17 个模型 | 自研模型要上板 |
| **TFLite Micro** | 中 | 中 | 通用 | 已有 TFLM 资产 |
| **MicroPython + ulab** | 低 | 低 | 无 | 纯逻辑、无模型 |

> **工程结论**：**验证想法用 ESP-VISION，产品化用 ESP-DL。**
> ESP-VISION 的零工具链（浏览器烧录 + 在线 IDE + 官方 MCP）能把「能不能跑」
> 这个问题的成本压到几分钟；一旦决定上板、要做自定义模型、要在 C++ 里控制
> 内存与算子，就该换 ESP-DL。

---

## 1. 框架总览：五个子系统

```
esp-dl/
├── dl/                    ← 核心：模型 + 算子 + 张量 + 数学
│   ├── model/             模型加载、内存分配与规划
│   ├── module/            算子接口（81 个头文件）
│   ├── base/              算子实现 + ISA 特定加速
│   │   └── isa/           c / riscv / xtensa / esp32p4 / tie728
│   ├── tensor/            张量类型
│   ├── math/              矩阵运算
│   └── tool/              工具函数
├── vision/                ← 应用层：分类 / 检测 / 识别 / 图像处理
├── audio/                 ← 音频：WAV、FBank、MFCC、频谱图
├── fbs_loader/            ← .espdl 模型格式加载器（FlatBuffers）
├── tools/                 ← 算子测试、FFT、Agent skills
└── models/                ← 17 个预训练模型的 IDF 组件
```

### 1.1 四个关键设计

**① `.espdl` 模型格式（FlatBuffers，不是 Protobuf）**

```fbs
namespace espdl;
enum TensorDataType : int32 { UNDEFINED=0, FLOAT=1, UINT8=2, INT8=3, ..., FLOAT16=10, ... };
enum DataLocation : int32 { DEFAULT=0, EXTERNAL=1 };
```

FlatBuffers 相比 ONNX/Protobuf 的两个优势：**体积更小**、**支持零拷贝反序列化**
（映射到内存后不用反序列化过程）。2026-01 起 [netron](https://netron.app/)
已支持可视化 `.espdl`，调试体验显著改善。

**② 三条量化路线，可在模型内任意混用**

| 模式 | 含义 | 典型用途 |
|---|---|---|
| `w8a8` | 权重 8bit / 激活 8bit | 速度优先，backbone |
| `w16a16` | 权重与激活都 16bit | 精度敏感层（检测头） |
| `w8a16` | 权重 8bit / 激活 16bit | v3.3.11 起全面支持 |

「混合量化」是 ESP-DL 的核心卖点：**不是整模型一个精度，而是逐层挑**。
YOLO26 就是典型：backbone 用 INT8，box/class head 用 INT16。

**③ 静态内存规划器**

`dl::Model` 构造时传入 `max_internal_size` 与 `mm_type`，规划器自动决定
每一层放在 RAM / PSRAM / Flash 的哪个位置。

```cpp
dl::Model* model = new dl::Model((const char*)model_espdl,
                                 fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                 0,                         // max_internal_size
                                 dl::MEMORY_MANAGER_GREEDY, // mm_type
                                 nullptr,                   // key
                                 false);                    // param_copy
```

`MEMORY_MANAGER_GREEDY` 表示「有多少可用内存就用多少」。
`param_copy=false` 保留原始参数不复制 —— **省 RAM，但模型缓冲区必须是只读的**。

**④ 8bit LUT 激活 + 双核调度**

除 `Relu` 与 `PRelu(n>1)` 外，所有激活函数默认走 **8bit LUT** 实现。
`Conv2D` 与 `DepthwiseConv2D` 支持双核调度。

---

## 2. 算子支持矩阵：68 个算子与三档芯片

`operator_support_state.md` 明确写着：**68 个算子已实现并测试**，建议导出
**ONNX opset 18**。接口与 ONNX 对齐，但部分算子有额外限制。

### 2.1 芯片分三档，**舍入策略不同**

| 档位 | 芯片 | 指令集 | 舍入策略 | 算子实现 |
|---|---|---|---|---|
| 无 PIE | ESP32 / C2 / C3 / C5 / C6 / S2 | 纯 C | **round half up** | 纯 C，明显慢 |
| PIE V1 | ESP32-S3 | PIE V1 | **round half up** | 汇编加速 |
| PIE V2 | ESP32-P4 / ESP32-S31 | PIE V2 | **round half to even** | 汇编加速 |

> ⚠️ **这是一个容易被忽略的精度陷阱**：同样一个量化模型，
> 在 S3 与 P4 上因为舍入策略不同，**输出不会逐位相同**。
> 跨芯片对比精度时必须说明目标芯片，否则数字没有可比性。

### 2.2 布局约定

大多数算子与 ONNX/PyTorch 布局一致；但 **Conv、GlobalAveragePool、
AveragePool、MaxPool、Resize** 为榨取指令级加速，输入输出采用 **NHWC / NWC**。
自己写后处理时，坐标映射要按这个前提来。

### 2.3 常见算子的限制（摘录）

| 算子 | 限制 |
|---|---|
| **Conv** | 支持 1d/2d；groups 只支持 1 或等于输入通道；**不支持 float32** |
| **Gemm / MatMul** | 最高 4D；**不支持 float32**；w8a16 只作用于静态 input1 |
| **LayerNormalization** | 任意 axis 可归一化，但**只有 `axis == -1` 可部署**（esp_ppq 的 LayerNorm_forward 拒绝其他） |
| **Resize** | 只支持 1d/2d 的 nearest/linear/bilinear；不支持 roi、antialias；**不支持 int16** |
| **MaxPool / AveragePool** | 支持 1d/2d，**不支持 dilation** |
| **ConvTranspose** | 通过 InsertZeros + Conv 实现 |
| **Pad** | 不支持 wrap 模式 |
| **Attention** | 支持 4D 自注意力/交叉注意力；**不支持 KV cache** 与 softcap/window 等高级属性 |
| **RotaryEmbedding** | 仅 4D `[B,H,L,D]`，D 须为偶数，half-split 全旋转，三输入，无 position_ids |
| **Mod** | 支持 4D、多向广播、fmod=1（C 风格） |

> **工程结论**：**先查算子矩阵，再选模型。**
> 很多「导出 ONNX 后 ESP-DL 跑不起来」的问题，根因是某个算子不被支持
> 或超出限制（比如 LayerNorm 的 axis、Resize 的 int16）。
> 这一步应该在量化之前做，而不是量化完之后。

---

## 3. 部署链路：从 ONNX 到板上的四步

```
① 训练 (PyTorch) → ONNX (opset 18)
        ↓  ESP-PPQ 量化（可选 TQT / AutoQuant / skill 自动调参）
② .espdl 模型
        ↓  作为 IDF 组件引入（models/xxx/）
③ 编译固件（模型二进制被 target_add_aligned_binary_data 嵌进固件）
        ↓  烧录
④ 运行时：dl::Model 加载 → preprocess → run() → postprocess
```

### 3.1 模型可以放在三个地方

| 位置 | 常量 | 适用 |
|---|---|---|
| `MODEL_LOCATION_IN_FLASH_RODATA` | Flash 中的只读数据段 | 模型固定，最省事 |
| Flash 分区 | 独立分区 | 模型需 OTA 升级 |
| SDCard | 外挂卡 | 模型大且需频繁更换 |

### 3.2 组件依赖怎么写

```yaml
# main/idf_component.yml
dependencies:
  espressif/esp-dl:
    version: "~3.3.15"
  espressif/yolo26:
    version: "~0.1.1"
```

> ⚠️ **版本钉死的坑**：`models/yolo26/idf_component.yml` 里写的是
> `esp-dl: "~3.3.14"`，而核心组件当前是 `3.3.15`。
> `~3.3.14` 的语义是「>=3.3.14, <3.4.0」，所以 3.3.15 仍满足 —— 但
> **读 yolo26 组件的 README 会以为 esp-dl 只有 3.3.14**，以实际 `idf_component.yml` 为准。

### 3.3 ESP-IDF 版本要求（按芯片不同）

```yaml
idf:
  matches:
    - if: target in ["esp32", "esp32c2", "esp32c3", "esp32c6", "esp32p4", "esp32s2", "esp32s3"]
      version: ">=5.3"
    - if: "target == esp32c5"
      version: ">=5.5"
    - if: "target == esp32s31"
      version: ">=6.0"
```

**S31 需要 IDF ≥ 6.0，S5 需要 ≥ 5.5**，其余 ≥ 5.3。
本机 `E:\esp-idf` 是 **v5.5.4** —— 够 P4/S3/C6，**不够 S31**。

### 3.4 推理四步的标准写法

```cpp
// 1. 解码 JPEG → RGB888
auto img = processor.decode_jpeg(jpg_data, jpg_len);

// 2. 预处理：letterbox + SIMD LUT 量化，直接写入模型输入 RAM
processor.preprocess(img);
heap_caps_free(img.data);          // 立刻释放解码缓冲

// 3. 硬件推理
model->run();

// 4. 后处理（NMS-free，top-K 按分数）
auto results = processor.postprocess(model->get_outputs());
```

第 2 步后的 `heap_caps_free(img.data)` 是**必须的** ——
解码缓冲通常几 MB，ESP32-P4 只有 32MB PSRAM，漏了这行迟早 OOM。

---

## 4. 量化：ESP-DL 的五种打法

这是 ESP-DL 生态里最值得花时间理解的部分。**难点从来不是「跑一遍量化」，
而是「选一组配置」** —— 官方文档原话：真正困难的部分往往并不在于量化流程本身，
而在于如何选择合适的量化配置。

| 打法 | 入口 | 适用 |
|---|---|---|
| **标准 PTQ** | `espdl_quantize_onnx` / `espdl_quantize_torch` | 默认起点 |
| **per-channel 量化** | 同上 + `target="esp32p4"` | P4 的 Conv/Gemm，精度显著更好 |
| **TQT** | `TrainedQuantizedThresholdPass` | 逐块重建损失优化阈值，快、无需真反传 |
| **AutoQuant** | `espdl_auto_quantize_onnx` | 自动搜索量化配置 |
| **espdl-quantize skill** | Agent 驱动的闭环调参 | 把「盯报错猜配置」变成有结构的搜索 |

### 4.1 AutoQuant：把调参变成搜索

```python
from esp_ppq.api import AutoQuantSearchSetting, espdl_auto_quantize_onnx

def evaluate_fn(graph):
    top1, top5 = ...
    return top1, {"top1": top1, "top5": top5}

setting = AutoQuantSearchSetting(
    search_mode="fast",
    num_of_candidates=5,
    score_direction="maximize",
    run_dir="outputs/auto_quant_mobilenetv2",
)

espdl_auto_quantize_onnx(
    onnx_import_file="model.onnx",
    espdl_export_file="outputs/model.espdl",
    calib_dataloader=calib_loader,
    calib_steps=32,
    input_shape=[3, 224, 224],
    evaluate_fn=evaluate_fn,          # ← 用户只需提供这一段
    target="esp32p4",
    setting=setting,
    device="cuda",
)
```

与 `espdl_quantize_onnx` 的区别只有一句话：
**前者执行一次固定配置，后者自动搜索、评测、筛选**。
用户侧的契约极小 —— 只需要一个校准 dataloader 和一个评估函数。

### 4.2 espdl-quantize skill：把调参经验结构化

`tools/agents/skills/espdl-quantize/` 是一个给 AI Agent 用的 skill，
它驱动的闭环是：

```
baseline → 校准 × TQT(default) 笛卡尔积 → 分布感知的残差修复
        → Agent 驱动的开放探索 → 重新评估
```

几条硬规则值得抄进自己的流程里：

- **方法顺序是「精度优先」**，并对会拖慢端侧推理的 pass 施加软惩罚；
- **LSQ 在 POWER_OF_2 目标上会被自动禁用**（会静默退化），改用 TQT；
- **esp32p4 的 layer-wise equalization 只警告不阻断** ——
  esp-ppq 官方标注「不推荐」用于 per-channel 权重，但实测在部分模型上仍有帮助。

> **工程结论**：**量化调参本质上是「分布问题」不是「参数问题」。**
> 先看逐层量化误差分布，再决定用哪个 pass —— 而不是先挑一个 pass 再看结果。

### 4.3 YOLO26 的七步流水线（完整案例）

```
① ONNX 导出（打 RegMax=1 补丁，移除 DFL 层）
② PTQ 校准（喂校准图，确定逐层动态范围）
③ TQT（逐块重建损失优化阈值，快、无需真反传）
④ Passive + Alignment Pass（推导 bias/passive scale，把 Add/Concat 对齐到同一量化 scale）
⑤ INT16 Step-Interpolated LUT 融合（把 INT16 Swish 激活转成紧凑 LUT，step 默认 32）
⑥ 图手术（把 Concat 输出节点拆成 6 个离散张量：one2one_p3_box / one2one_p3_cls / …）
⑦ .espdl 导出（内嵌 LUT 表）
```

四个关键设计：

| 设计 | 作用 |
|---|---|
| **RegMax=1** | 彻底去掉 DFL 层，**计算量减少约 30%** |
| **NMS-Free** | One2One 头，按置信度取 top-K，**MCU 上不需要 NMS** |
| **INT8 backbone + INT16 head** | 敏感层（box/class head、neck 出口）用 INT16 保住精度 |
| **LUT 融合** | Swish 用带步长插值的 INT16 LUT 精确模拟硬件行为 |

> ⚠️ **图手术这一步有硬约束**：ESP-DL 的图切分按**节点名**匹配
> （`one2one_p3_box` / `one2one_p3_cls` 这类名字）。
> **自定义模型必须复现这套命名**，否则导出后无法正确切图。
> 这是自研模型上板最常踩的坑之一。

---

## 5. YOLO26 端到端案例

### 5.1 官方跑分（转述自 `models/yolo26/README.md`）

| 型号 | 输入 | Flash/PSRAM (MB) | 预处理 | 推理 | 后处理 | mAP50-95 |
|---|---|---|---|---|---|---|
| yolo26n_512_s8_p4 | 512×512×3 | 16 / 32 | 10.4 | **929.8** | 8.5 | 0.365 |
| yolo26n_640_s8_p4 | 640×640×3 | 16 / 32 | 15.2 | **1516.0** | 13.3 | 0.387 |
| yolo26n_512_s8_s31 | 512×512×3 | 16 / 16 | 16.7 | 1557.6 | 12.2 | 0.365 |
| yolo26n_640_s8_s31 | 640×640×3 | 16 / 16 | 24.4 | 2428.3 | 18.9 | 0.387 |
| yolo26n_512_s8_s3 | 512×512×3 | 16 / 16 | 34.6 | 2348.2 | 23.5 | 0.363 |
| yolo26n_640_s8_s3 | 640×640×3 | 16 / 16 | 51.2 | 3792.9 | 36.6 | 0.384 |

> 单位 ms。口径：int8，`CONFIG_COMPILER_OPTIMIZATION_PERF`。
> 芯片参数：S3 CPU 240MHz / data cache 64KB / Octal PSRAM 80MHz；
> S31 CPU 320MHz / PSRAM 250MHz；
> P4 CPU 400MHz / L2 256KB / PSRAM 32MB@250MHz。
> **性能受内存配置（Flash vs PSRAM）影响。**

**换算成帧率**（推理时间口径，不含预处理）：

| 芯片 | 512 输入 | 640 输入 |
|---|---|---|
| ESP32-P4 | 1.08 FPS | 0.66 FPS |
| ESP32-S31 | 0.64 FPS | 0.41 FPS |
| ESP32-S3 | 0.43 FPS | 0.26 FPS |

**P4 相对 S3 是 2.5×，相对 S31 是 1.67×。**

### 5.2 精度代价

YOLO26n 原版 FP16 是 **40.9 mAP**，ESP-DL 的 INT8 是 **36.5~38.7 mAP**（视分辨率）。
**掉约 2~4 个点**，这个代价比很多人预期的小 —— 混合量化（head 用 INT16）贡献很大。

### 5.3 模块层的六个工程细节

| 特性 | 说明 |
|---|---|
| **NMS-Free 后处理** | One2One 头只按置信度选 top-K，不需要 IoU 抑制 |
| **硬件 letterbox** | 灰边填充（value=114），与 Python 模拟**逐像素一致** |
| **SIMD LUT 量化** | 256 项 INT8 LUT 硬件加速，直接把像素量化进模型输入 RAM，**零拷贝** |
| **模板化 decode** | `decode_grid<T>` 运行时按张量 dtype 分派，INT8/INT16/混合都能透明处理 |
| **整数阈值优化** | 置信度阈值预先转成整数空间，低分格子直接跳过 sigmoid，**热循环里没有浮点运算** |
| **类别数自动探测** | `num_classes` 从输出张量形状运行时读出，不用硬编码 |

### 5.4 模型文件与构建

实际 `.espdl` 文件大小（本地 v3.3.15 仓库）：

```
models/yolo26/models/p4/yolo26n_512_s8_p4.espdl   3,072,864 B ≈ 2.93 MB
models/yolo26/models/p4/yolo26n_640_s8_p4.espdl   3,072,256 B ≈ 2.93 MB
models/yolo26/models/s3/yolo26n_512_s8_s3.espdl   3,073,040 B ≈ 2.93 MB
models/yolo26/models/s3/yolo26n_640_s8_s3.espdl   3,072,208 B ≈ 2.93 MB
```

> 口径区分：benchmark 表里的 **Flash 16 MB / PSRAM 32 MB** 是「模型 + 运行时」的
> 存储需求口径；**2.93 MB** 是单个 `.espdl` 文件本身的大小。
> 两个数不是一回事，别混用。

构建时模型解析顺序（`examples/yolo26_detect/main/CMakeLists.txt`）：

```
1. esp-dl/models/yolo26/models/[p4|s3]/     ← 组件市场优先
2. main/models/[p4|s3]/                      ← 本地自定义优先于 zoo？反过来
```

实际逻辑是 **Zoo 优先，本地其次**（`if(EXISTS ZOO_PATH) ... elseif(EXISTS LOCAL_PATH)`），
且 CMake 会**按芯片自动选模型**：P4/S31 用 `yolo26n_512_s8_p4.espdl`，
S3 用 `yolo26n_512_s8_s3.espdl`。文件名里带 `lego` 时自动定义 `USE_LEGO_MODEL`。

---

## 6. AI Agent 工具链：ESP-DL 最容易被忽略的变化

`tools/agents/skills/` 下有 **四个给 coding agent 用的 skill**（2026 年新增）：

| Skill | 干什么 |
|---|---|
| `espdl-quantize` | 量化调参的闭环搜索（见 §4.2） |
| `espdl-operator` | **新增算子的完整生命周期**：C++ module → C 参考实现 → SIMD 汇编优化 → esp-ppq 量化支持 → Docker 构建测试 → 与 esp-ppq 的推理结果对齐 |
| `esp32p4-pie-simd` | 把代码重写为 **PIE V2** 指令加速版本 |
| `esp32s3-pie-simd` | 把代码重写为 **PIE V1** 指令加速版本 |

`espdl-operator` 的工作流纪律写得非常明确，值得原样抄进任何「改底层算子」的项目：

> 完成**任何**代码修改后，立刻进入测试阶段，不要停下来问用户「现在要跑测试吗」。
> 唯一该停下来的理由是：缺少用户没提供的信息、编译错误需要用户输入、
> 或用户明确要求停下。多文件修改时**先全部改完，再统一跑一次完整测试**，
> 不要每改一个文件就测一次。

`espdl-operator` 还强调了一个关键概念：**推理结果对齐** ——
新算子在 esp-dl 上的输出，必须与 esp-ppq 的参考实现**逐值对齐**，
否则量化误差会被算子实现误差掩盖。

> **工程结论**：**这套 skill 本身就是一份「怎么写底层算子」的方法论。**
> 即便你不用 AI agent，PIE SIMD 的两条 skill（V1/V2）也是现成的
> 「怎么把 C 代码改成汇编加速」的参考。

---

## 7. 算子加速比基准

`benchmark_report.md`（esp-dl 3.3.10，2026-09-01 生成，955 个用例，
以 **esp32 为 1.00× 基线**，每算子取几何平均）：

| 算子 | esp32 | esp32s3 | esp32p4 |
|---|---|---|---|
| **Conv** | 1.00× | **22.9×** (3.40–171) | **38.8×** (3.54–283) |
| **ConvTranspose** | 1.00× | 31.8× | **64.3×** |
| **MaxPool** | 1.00× | 37.1× | **77.2×** |
| **Gemm** | 1.00× | 2.36× | 5.52× |
| **Mul** | 1.00× | 3.20× | 10.7× |
| **MatMul** | 1.00× | 3.04× | 6.98× |
| **GlobalAveragePool** | 1.00× | 3.07× | 5.02× |
| **LayerNormalization** | 1.00× | 1.32× | 3.00× |
| **Flatten** | 1.00× | 0.90× | 1.15× |

> 括号内是各算子用例的 min–max 区间，**波动很大**（Conv 在 P4 上最高能到 283×）。
> 用几何平均做对比，不要用单点数字下结论。

**注意 Flatten 在 S3 上是 0.90×** —— 比基线还慢。算子加速不是均匀的，
取决于数据布局与形状。

2026-10-15 的更新记录：Conv、Gemm、MatMul、Resize、Reduce、LUT、Transpose、Pooling
等大量算子优化后，**MobileNetV2、YOLO11 在 ESP32-S3 上获得 3 倍以上加速，
在 ESP32-P4 上获得 1.9 倍以上加速**。

---

## 8. 工程避坑清单

1. **量化前先查算子矩阵。** LayerNorm 的 axis、Resize 的 int16、Conv 不支持 fp32 —— 这些是根因级问题。
2. **必须用 RegMax=1 变体**（去掉 DFL），否则图节点名对不上，图切分直接失败。
3. **自定义模型要复现节点命名**（`one2one_p3_box` / `one2one_p3_cls`），这是图切分的匹配键。
4. **S31 需要 ESP-IDF ≥ 6.0，S5 需要 ≥ 5.5。** 用错 IDF 版本会在编译期才暴露。
5. **`heap_caps_free(img.data)` 不能漏**，否则几 MB 的解码缓冲会拖垮 PSRAM。
6. **per-channel 量化依赖 esp-ppq ≥ 1.2.10 且 ESP-DL ≥ 3.3.1。**
7. **LSQ 在 POWER_OF_2 目标上静默退化**，用 TQT 替代。
8. **跨芯片对比精度要声明目标芯片**：S3（half-up）与 P4/S31（half-to-even）舍入不同。
9. **模型二进制被编进固件**，改模型要重新编译；想 OTA 换模型要用 Flash 分区或 SDCard。
10. **不要把 Zoo 里的模型当自己的模型用**：CMake 解析时 **Zoo 优先**，本地 `main/models/` 只在 Zoo 里没有同名文件时才生效。
11. **ESP-IDF 磁盘占用大**：只装 riscv32 目标约 8–10GB，带上 Xtensa 15–20GB，每工程 build 缓存约 2GB。
12. **卷积/池化等算子采用 NHWC**，自己写后处理时坐标映射别按 NCHW 想。

---

## 9. 一条落地路线

如果要把自研模型搬上 ESP32，按这个顺序走：

| 阶段 | 动作 | 通过标准 |
|---|---|---|
| ① 可行性 | 查算子矩阵，列出**不支持**的算子 | 全部有替代方案 |
| ② 导出 | 导出 ONNX（opset 18），确认能加载 | `netron` 能打开 |
| ③ 粗量化 | `espdl_quantize_onnx` 出第一版 `.espdl` | 能跑通、精度可接受 |
| ④ 调参 | TQT → AutoQuant → espdl-quantize skill | 精度达标 |
| ⑤ 图手术 | 按 ESP-DL 命名规则切图 | 节点名匹配成功 |
| ⑥ 上板 | 组件化、嵌入、跑 example | 延迟与内存达标 |
| ⑦ 优化 | PIE SIMD skill 重写热点算子 | 算子级加速 |

> **工程结论**：**把「算子支持」和「图节点命名」这两个检查放到最前面。**
> 它们是唯一会让前面所有工作作废的环节，而检查成本只有几十分钟。

---

## 10. 工程结论

1. ESP-DL 是一套**契约**：模型格式、算子集合、量化方式、内存模型都是绑定的。
2. **验证用 ESP-VISION，产品化用 ESP-DL。**
3. 混合量化（w8a8 / w16a16 / w8a16 逐层混用）是 ESP-DL 的真正优势，YOLO26 的 2 点精度损失就是这么来的。
4. **68 个算子、三档芯片、三种舍入策略** —— 跨芯片比较必须声明目标。
5. 量化调参的难点是**选配置**不是**跑流程**：AutoQuant 与 espdl-quantize skill 把这件事自动化了。
6. **TQT 是当前最值得掌握的量化技术**（无真反传、快、效果好）。
7. RegMax=1 与节点命名是自研模型上板的**两个硬门槛**。
8. `tools/agents/skills/` 里的四个 skill 本身就是一份可复用的底层工程方法论。
9. **算子加速不均匀**（Flatten 在 S3 上是 0.90×），别用单点数字下结论。
10. 先查算子矩阵与命名规则，再动手 —— 这两步作废前面所有工作。

---

## 附：源码地图（本机 `E:\p4work\esp-dl`）

| 关注点 | 路径 |
|---|---|
| 框架总览（官方） | `README_cn.md` |
| 算子支持矩阵 | `operator_support_state.md` |
| 算子加速比基准 | `benchmark_report.md` |
| 项目组织说明 | `docs/zh_CN/introduction/esp_dl_project.rst` |
| YOLO26 量化流水线 | `examples/tutorial/how_to_quantize_model/quantize_yolo26/README.md` |
| AutoQuant 教程 | `docs/zh_CN/tutorials/auto_quantization/how_to_use_AutoQuant.rst` |
| 量化 skill 教程 | `docs/zh_CN/tutorials/auto_quantization/how_to_use_espdl_quantize_skill.rst` |
| 算子实现 skill | `tools/agents/skills/espdl-operator/SKILL.md` |
| PIE SIMD skill | `tools/agents/skills/esp32p4-pie-simd/`、`esp32s3-pie-simd/` |
| YOLO26 模型组件 | `models/yolo26/`（`yolo26.cpp` / `yolo26.hpp` / `coco_classes.cpp`） |
| 推理例程 | `examples/yolo26_detect/main/app_main.cpp` |
| 模型格式定义 | `esp-dl/fbs_loader/espdl.fbs` |
| 算子实现 | `esp-dl/dl/module/include/`、`esp-dl/dl/base/isa/` |

---

## 🔗 相关

- 栈级速览（ESP-VISION / 官方 MCP / 模型上架流程）→ [`02-ESP32端侧视觉栈-ESP-VISION与ESP-DL.md`](./02-ESP32端侧视觉栈-ESP-VISION与ESP-DL.md)
- 本机 ESP-IDF 环境与踩坑 → 记忆 `esp-idf-local-setup`
- 农业场景选型 → [`03-农业病虫害监测方案选型.md`](./03-农业病虫害监测方案选型.md)
- 本章索引 → [`README.md`](./README.md)
