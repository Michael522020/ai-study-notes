# ESP32 端侧视觉栈：ESP-VISION 与 ESP-DL

> **核心问题**：把 YOLO26 塞进 ESP32，硬件到底够不够？有没有不装十几 GB 工具链的捷径？

本章梳理乐鑫（Espressif）**ESP32-S3 / ESP32-S31 / ESP32-P4** 上的端侧视觉 AI 技术栈：**ESP-VISION**（低代码 MicroPython 框架 + 官方 MCP）与 **ESP-DL**（推理框架 + YOLO26 模型库），并给出自研模型的量化落地方案。

**数据口径**：本章除明确标注「本机实测」外，其余数字均为**转述自 Espressif 官方开源仓库的公开报告值**（`espressif/esp-vision`、`espressif/esp-dl`），引用前请回原始出处核对。

---

## 1. 先认清三款芯片的定位

| 芯片 | CPU | 缓存 | PSRAM | 射频 | AI 加速 |
|---|---|---|---|---|---|
| **ESP32-P4** | 双核 RISC-V，400 MHz（+ 低功耗核） | 256 KB L2 | 32 MB Octal @250 MHz | **无** | 定制 SIMD 向量扩展（VLEN 256-bit），约 8 INT8 MAC/cycle |
| **ESP32-S31** | Xtensa LX7，320 MHz | — | 16 MB @250 MHz | 有 | 同 PIE 内核路线 |
| **ESP32-S3** | Xtensa LX7，240 MHz | 64 KB | 16 MB Octal @80 MHz | 有 | 同 PIE 内核路线 |

### 1.1 最容易踩空的一条：P4 没有射频

**ESP32-P4 不带 Wi-Fi 也不带蓝牙。** 它是主控/协处理器，不是无线节点。要联网必须外挂 **ESP32-C6 / C5**。做"田间设备上报告警"这类需求时，这条决定了板级设计——不是软件能补的。

### 1.2 原版 ESP32 不用考虑

原版 ESP32（Xtensa LX6，240 MHz，520 KB SRAM，多数模组无 PSRAM）**跑不了 ESP-DL 的 YOLO 模型**——ESP-DL 的 YOLO26 支持列表只有 S3 / S31 / P4，且要求 16 MB Flash + 16 MB 以上 PSRAM。ESP-DL 里给原版 ESP32 准备的是 `esp-nn`（早期 legacy 路线）和 `motion_detect` / `color_detect` 这种极简模型。

### 1.3 硬件加速的定位

P4 的 AI 加速是**固定点、面向量化模型的协处理器**，不是通用 NPU。它通过 ESP-DL 的静态图编译器工作，接受的是 ESP-PPQ 量化后的定点模型，不是标准 ONNX/PyTorch 模型。

> 常见误解需要澄清：「P4 有 NPU 所以能跑 YOLOv5s」——不成立。P4 的加速单元只吃 ESP-DL 编译出的定点模型，且算力规模只够 n 档。

---

## 2. ESP-VISION：把工具链整个绕开

**ESP-VISION**（乐鑫 2026-07-02 发布）是面向 ESP32-P4 / S31 / S3 的低代码边缘 AI 视觉框架，核心卖点是**零本地工具链**。

### 2.1 架构：五个 Python API

| API | 职责 |
|---|---|
| `sensor` | 摄像头初始化、像素格式、分辨率、抓帧 |
| `image` | 绘图、滤波、颜色追踪、帧差、Canny、模板匹配、QR/Barcode/AprilTag |
| `display` | 显示输出 |
| `espdl` | ESP-DL 推理（检测 / 姿态 / 分类），另带 `tflite` 走 TFLM |

官方给的完整示例是 **17 行**：

```python
import espdl, sensor, time

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=1000)

det = espdl.ESPDet("/sdcard/hand_det.espdl", score=0.5, nms=0.7)

while True:
    img = sensor.snapshot()
    for x, y, w, h, score, category in det.detect(img):
        img.draw_rectangle(x, y, w, h, color=(255, 0, 0), thickness=2)
        img.draw_string(x, max(0, y - 12), "%.2f : %d" % (score, category))
    img.flush()
    time.sleep_ms(20)
```

### 2.2 浏览器里烧录和写代码

- 官网 / 在线 Web IDE：<https://vision.espressif.com>
- **Web Serial 一键烧录**：Chrome/Edge 连数据线 → 选固件 → 烧录 → 自动进 Web IDE，**不需要装任何本地工具链**
- 也有 VS Code 扩展；固件按板子分版本发布（截至 2026-10-23 为 ESP32-P4X-EYE 最新）
- 硬件编解码：H.264 / MJPEG / RTSP / USB CDC 实时预览，能把视频流推到局域网

### 2.3 官方 MCP：Claude Code 可以直接驱动板子

ESP-VISION 提供官方 MCP server，**明确点名支持 Codex / Claude / Cursor / VS Code / OpenAI Codex**：

```bash
npx -y @esp-vision/plugin install claude     # 或 install codex
```

- MCP endpoint：`https://mcp.vision.espressif.com/mcp`
- npm 包：`@esp-vision/plugin`（插件）、`@esp-vision/device-cli`（设备命令）、`@esp-vision/device-core`（SDK）、`@esp-vision/knowledge-mcp`（知识服务）
- 设备命令入口：`esp-vision-device --help --json`
- 架构上分两层：**知识 MCP**（在线知识查询，走 Streamable HTTP）与**本地 Device CLI**（通过 USB/TCP 操设备，输出 JSON / NDJSON 事件 / 二进制文件）

> 这一层的设计值得单独记住：**Agent 拥有 AI 循环，设备只暴露能力**。设备源文件、照片、凭据不会上报到知识服务，只上报按天聚合的匿名调用统计（保留 90 天）。

### 2.4 自定义模型怎么接进来

**模型不烧进固件。** 把 `.espdl` 拷到板载 `/sdcard` 或 `/flash`，代码里一行加载。

每个模型文件要配一个**同名 `.json` sidecar**，网站模型列表是靠扫 `models/` 目录自动生成的，放进去就上架：

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | ✅ | 显示名，如 `ESPDet Pico Cat` |
| `architecture` | ✅ | 如 `ESPDet Pico`、`YOLO11n` |
| `api` | ✅ | 运行时 API 类，如 `espdl.ESPDet` |
| `task` | ✅ | `detection` / `pose` / … |
| `input` | ✅ | 如 `224x224` |
| `inputFormat` | ✅ | 如 `RGB565` |
| `dataset` | ✅ | 如 `COCO` |
| `labels` | ✅ | 标签列表（COCO 这类大类别集可留空） |
| `sizeBytes` | ✅ | 文件大小，用于与 `.espdl` 交叉校验 |
| `description` / `datasetUrl` | ❌ | 可选 |

`file` / `path` / `slug` / `downloadUrl` 由文件位置推导，**不能手写**。

### 2.5 内置模型（注意：不包含农业模型）

内置：ESPDet Pico（猫/狗/人脸/手）、YOLO11n（COCO 检测）、YOLO11n-Pose、安全帽检测。

仓库 `models/` 目录实际包含 `espdet` / `nanotrack_v2` / `pedestrian_detect` / `pp_ocr_v6` / `tflite` 五个子目录——**没有 YOLO26，也没有农业模型**。更多模型可从 ESP-DL 模型库取。

---

## 3. ESP-DL 与 YOLO26 官方模型

仓库：<https://github.com/espressif/esp-dl>，最新发布 **v3.3.14（2026-10-19）**。

### 3.1 官方跑分表（YOLO26n INT8）

| 芯片 | 输入 | Flash/PSRAM | 预处理 | **推理** | 后处理 | mAP50-95 |
|---|---|---|---|---|---|---|
| **ESP32-P4** | 512×512 | 16/32 MB | 10.4 ms | **929.8 ms** | 8.5 ms | 0.365 |
| **ESP32-P4** | 640×640 | 16/32 MB | 15.2 ms | **1516.0 ms** | 13.3 ms | 0.387 |
| ESP32-S31 | 512×512 | 16/16 MB | 16.7 ms | 1557.6 ms | 12.2 ms | 0.365 |
| ESP32-S31 | 640×640 | 16/16 MB | 24.4 ms | 2428.3 ms | 18.9 ms | 0.387 |
| ESP32-S3 | 512×512 | 16/16 MB | 34.6 ms | 2348.2 ms | 23.5 ms | 0.363 |
| ESP32-S3 | 640×640 | 16/16 MB | 51.2 ms | 3792.9 ms | 36.6 ms | 0.384 |

**读这张表的关键：输入分辨率才是决定性变量。**

- P4 跑 640×640 只有 **0.66 FPS**
- ESP-VISION 官方实测 **P4X-EYE @ QVGA(320×240) 目标检测 ≈ 11 FPS**——像素量少了一个量级（76,800 vs 262,144），速度差 3 倍多

**结论：不要用 640 输入去评估 ESP32 的可行性，先确定你的应用能接受多少像素。**

### 3.2 精度代价意外地小

YOLO26n 原版在 COCO 上官方给出的五档整体区间是 **40.9–57.5 mAP**（下界对应 n 档），而 ESP-DL 的 INT8 版本 640 输入是 **mAP50-95 = 0.387**。

**量化只掉约 2 个点。** 做到这一点靠的是三件事：

| 手段 | 作用 |
|---|---|
| **RegMax=1** | 直接砍掉 DFL 层，回归不再需要分布打分，计算量降约 30% |
| **INT8 + INT16 混合量化** | 骨干走 INT8，检测头 / 分类头 / 颈部出口等敏感层走 INT16 |
| **INT16 步进插值 LUT** | Swish 激活换成紧凑 LUT（默认步长 32），在 MCU 上做硬件级精确逼近 |

### 3.3 例程实测日志与 model zoo 数字的差异

`examples/yolo26_detect` 在 P4 开发板上的实际输出：

```text
I (4354) yolo26_detect: Pre: 12 ms | Inf: 2071 ms | Post: 13 ms
```

比 model zoo 表里的 929.8 ms 高一倍多。**同一模型同一芯片，两个数字差 2.2 倍。** 原因是构建配置（`CONFIG_COMPILER_OPTIMIZATION_PERF`、PSRAM 频率、模型版本）不同。

> 引用任何跑分都要带口径。**model zoo 的表和例程的日志都别单独用。**

### 3.4 YOLO11n 老基线（供对照）

ESP-DL 早期 YOLO11n INT8 数据：

| 目标 | 预处理 | 推理 | 后处理 | mAP50-95 |
|---|---|---|---|---|
| ESP32-S3 @ 640 | 207.9 ms | 26,919 ms | 59.0 ms | 0.307 |
| ESP32-S3 @ 640（v2） | 207.9 ms | 16,796 ms | 59.3 ms | 0.332 |
| ESP32-P4 @ 640 | 105.8 ms | 3,109 ms | 16.6 ms | 0.307 |
| ESP32-P4 @ 320（QAT） | 27.4 ms | 665 ms | 6.8 ms | 0.269 |

**P4 比 S3 快约 8.6 倍**（同精度、同输入）。

---

## 4. 自研模型的量化落地：ESP-PPQ

### 4.1 为什么不能直接量化

ESP32 只有 320 KB SRAM，INT8 是强制的。但**朴素量化会直接毁掉模型**（公开实测值）：

```
float32                92.3%
朴素 int8（不校准）     61.7%   ← 直接废了
PTQ + 约 100 张校准图   89.1%
```

**校准集质量比量化算法选择更重要。** 另一条经验：**剪枝之后不要直接量化**，要先微调，否则输出会塌成 0。

### 4.2 ESP-DL 的七步流水线

官方教程：`examples/tutorial/how_to_quantize_model/quantize_yolo26/README.md`

| 步 | 名称 | 作用 |
|---|---|---|
| 1 | **ONNX 导出** | 从 PyTorch 导出，**带 RegMax=1 patch**（移除 DFL 层） |
| 2 | **PTQ 校准** | 校准图过一遍图，确定逐层动态范围 |
| 3 | **TQT（Trained Quantization Threshold）** | 逐块优化量化阈值，重建损失驱动，不做反传 |
| 4 | **Passive + Alignment** | 推导 bias / passive scale；把逐元素算子（Add / Concat）对齐到统一量化 scale |
| 5 | **INT16 步进插值 LUT 融合** | Swish 激活转成带 LUT 的 INT16 算子 |
| 6 | **图切分** | 把 Concat 输出节点拆成 6 个离散张量（`one2one_p3_box`、`one2one_p3_cls` …） |
| 7 | **`.espdl` 导出** | 写出含 LUT 的部署模型 |

输出物：

```
output/
├── coco_512_s8_p4/
│   ├── yolo26n_512_s8_p4.espdl     ← 固件部署模型
│   ├── yolo26n_512_s8_p4.info      ← 逐层调试信息（约 15 MB）
│   ├── yolo26n_512_s8_p4.json      ← 量化 scale / 配置
│   └── yolo26_export.onnx          ← 中间 ONNX（量化前）
```

命名规范：`<model>_<img_sz>_s8_<platform>.*`。教程还支持 **160–640 任意输入分辨率**、任意数据集（COCO / Roboflow / 自定义）。

### 4.3 两个现成 notebook

| notebook | 用途 |
|---|---|
| `quantize_yolo26_roboflow.ipynb` | 自定义数据集：贴 Roboflow API key → 微调（默认 4 epoch）→ 全流水线 → `.espdl` |
| `quantize_yolo26_coco.ipynb` | 复现官方 mAP 基线，或做通用 80 类检测器 |

### 4.4 图切分这一步的约束

第 6 步把 Concat 输出拆成 `one2one_p3_box` / `one2one_p3_cls` 这类**固定节点名**。这意味着：

> **你自己训练的模型必须与官方 example 的图结构兼容**（One2One 头 + RegMax=1 直回归），否则第 6 步找不到节点，`.espdl` 导不出来。
>
> 换句话说：**不能拿标准 YOLO26n 的 ONNX 直接喂给 ESP-PPQ**，必须先打上 RegMax=1 patch。

---

## 5. 工具链磁盘：这是最容易被低估的成本

| 路线 | 磁盘开销 |
|---|---|
| ESP-IDF 完整安装（含所有 target） | 15–20 GB |
| ESP-IDF 只装 `riscv32-esp-elf`（P4 所需） | 约 8–10 GB |
| 单个工程的 build 缓存 | 约 2 GB |
| Windows 图形安装器离线包 | 约 7.4 GB |
| **ESP-VISION（浏览器方案）** | **≈ 0** |

> 📌 **本机实测约束**：这台机器 C 盘只剩 **17 GB**。按完整 ESP-IDF 路线装完基本就满了。
>
> 两个对策：① 装 ESP-VISION 路线，零工具链；② 若必须走 ESP-IDF，**把 `IDF_TOOLS_PATH` 指到 E 盘**，并**只勾选 riscv32 target**（Xtensa 工具链是给 ESP32/S2/S3 的，P4 用不上，能省 1–3.5 GB）。

国内下载用 `dl.espressif.cn` 镜像，路径不要带空格或非 ASCII 字符。

---

## 6. 可参考的开源项目

| 项目 | 内容 |
|---|---|
| **PlantGuard** | ESP32-S3 N16R8 + ESP-DL 3.3.9 + **YOLO26n** 番茄叶病检测，320×320，INT8；precision 0.814 / recall 0.665 / mAP@50 0.764；约 6.2 s/帧；ESP32-CAM(OV2640) 经 HTTP 送帧 |
| **`espressif/esp-dl/examples/yolo26_detect`** | 泛型框架：任意数据集 / 任意输入尺寸 / 任意类别数，模型加载系统可配置 |
| **ESP-DL 模型库** | <https://github.com/espressif/esp-dl/tree/master/models> |
| **组件市场** | <https://components.espressif.com/components/espressif/yolo26>（v0.1.1，预编译模型，开箱即用） |

> PlantGuard 的 6.2 s/帧 是个重要参照：**YOLO 级别的模型在 ESP32-S3 上就是"秒级"，不是"帧级"**。
> 想做实时，必须把输入分辨率和模型规模一起压。

---

## 7. 工程结论

1. **先算像素预算，再选芯片。** P4 跑 640×640 只有 0.66 FPS，跑 320×240 可以到 11 FPS——**分辨率比模型名更能决定可行性**。
2. **P4 没有射频是硬件事实，不是软件问题。** 任何需要联网的方案都要外挂 C6/C5。
3. **优先用 ESP-VISION 而不是 ESP-IDF。** 零工具链直接省下 8–20 GB 和一整天的环境折腾；还附带官方 MCP，Agent 可以直接驱动板子。有 C 盘紧张问题的机器尤其该走这条。
4. **要跑 YOLO26，必须用 RegMax=1 直回归变体。** 标准 ONNX 喂不进 ESP-PPQ 的图切分步骤。
5. **校准集决定量化生死。** 朴素 int8 能把 92.3% 打到 61.7%，加约 100 张校准图能救回 89.1%。剪枝后先微调再量化。
6. **引用跑分必须带口径。** model zoo 表与例程日志在同一芯片同一模型上差 2.2 倍；报告值、构建配置、PSRAM 频率、是否含后处理，四项都要交代。
7. **"P4 有 NPU 所以能跑 YOLOv5s"是误解。** P4 的加速单元只吃 ESP-DL 定点模型，算力只够 n 档，且需要约 16 MB Flash + 16/32 MB PSRAM。

---

## 🔗 相关

- 本机（x86 + 核显）部署实测 → [`./01-本机部署实战-YOLO26多后端实测.md`](./01-本机部署实战-YOLO26多后端实测.md)
- 农业场景的选型 → [`./03-农业病虫害监测方案选型.md`](./03-农业病虫害监测方案选型.md)
- RKNN 侧部署 → [`../01-YOLO/08-导出部署与推理加速.md`](../01-YOLO/08-导出部署与推理加速.md)
- 量化理论 → [`../02-视觉检测模型/13-训练工程与性能优化.md`](../02-视觉检测模型/13-训练工程与性能优化.md)
- 端到端无 NMS → [`../01-YOLO/05-YOLO11-26新周期.md`](../01-YOLO/05-YOLO11-26新周期.md)
