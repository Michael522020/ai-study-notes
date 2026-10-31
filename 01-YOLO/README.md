# 01 · YOLO 系列研究笔记

> **从 YOLOv1 到 YOLO26：一条「把检测从两阶段拉回单阶段，再把单阶段做成一阶段到底」的连续演化史。**

YOLO 的故事不是一串孤立模型的堆叠，而是一条**问题驱动的收敛曲线**。理解这条曲线，比记住每个版本改了哪个模块更重要。

---

## 📖 本目录文档

| # | 文档 | 核心问题 |
|---|---|---|
| 01 | [YOLO 总览与演进史](./01-YOLO总览与演进史.md) | 12 代演进的「继承—分化—收敛」主线，每代到底解决了上一代什么病 |
| 02 | [YOLOv1–v3：奠基期](./02-YOLOv1-v3奠基期.md) | 网格回归范式怎么建立的？Darknet 和多尺度预测解决了什么？ |
| 03 | [YOLOv4–v5：工程化期](./03-YOLOv4-v5工程化期.md) | Bag-of-Freebies / Bag-of-Specials 的思想拆分，CSP 与解耦头 |
| 04 | [YOLOv6–v10：分化期](./04-YOLOv6-v10分化期.md) | 解耦头回潮、Anchor-free、PGI/GELAN、NMS-free |
| 05 | [YOLO11–26：新周期](./05-YOLO11-26新周期.md) | C3k2/PSA、Attention 改造、DFL 退场、YOLOE 开放词汇 |
| 06 | [关键模块与算法](./06-YOLO关键模块与算法.md) | CSP、C3k2、SPPF、DFL、解耦头、标签分配、IoU 系列完整推导 |
| 07 | [训练调参与数据增强](./07-训练调参与数据增强.md) | 超参全解、闭集自训练、Mosaic/MixUp/Copy-Paste 实操配方 |
| 08 | [导出部署与推理加速](./08-导出部署与推理加速.md) | ONNX/TensorRT/RKNN 全链路，RK3588S NPU 实测与 INT8 校准 |
| 09 | [复现手册与避坑](./09-复现手册与避坑.md) | 从零跑通的完整命令链、60+ 条实战踩坑清单 |
| 10 | [论文与资源索引](./10-论文与资源索引.md) | 逐篇读法建议与官方仓库/权重地图 |

---

## 🧭 一句话理解 YOLO 的三个时代

```
v1 ─ v2 ─ v3 ─ v4 ─ v5 ─ v6 ─ v7 ─ v8 ─ v9 ─ v10 ─ v11 ─ v12 ─ v26
│ 奠基期      │ 工程化期      │  分化期       │   新周期
│ 2016-2018   │ 2020-2021     │ 2022-2024     │ 2024-2026
│ 建范式      │ 堆模块        │ 各路创新      │ 收敛 + 端到端
```

**奠基期（v1–v3）**：确立「一次前向出所有框」的范式。v1 解决了「两阶段太慢」，v2 解决了「v1 定位不准、泛化差」，v3 解决了「小目标漏检」。

**工程化期（v4–v5）**：不再追求「发明新范式」，而是把一堆已知有效的技巧系统化。v4 首次提出 **Bag-of-Freebies / Bag-of-Specials** 的分类法，v5 把这套方法论做成了工业级代码库。**这是 YOLO 从学术玩具变成工业标配的转折点。**

**分化期（v6–v10）**：思想解禁，多个团队并行探索。YOLOv6 把解耦头带回来（v5 之前被用烂了），YOLOv7 做重参数化，YOLOv9 做梯度信息流，YOLOv10 干掉 NMS。同期还出现了 YOLOX（旷视）、YOLO-World（阿里）等平行分支。

**新周期（v11–v26）**：从「加模块」转向「改范式」。v11 引入 C3k2 与 PSA，v12 把注意力引入主干，v26 彻底移除 DFL 并默认端到端无 NMS。

---

## ⚡ 一张表看懂各代「关键突破」

> **口径提醒**：mAP 均为 COCO `val`（除非特别标注），mAP@0.5:0.95，输入 640×640（v1–v3 为原论文设定）。不同仓库口径差异大，**横向对比时务必核对同一份 README**。

| 版本 | 年份 | 关键突破 | 典型 mAP@0.5:0.95 | 代表产物 |
|---|---|---|---|---|
| v1 | 2016 | 网格回归范式，一次前向出所有框 | — (VOC 63.4 mAP@0.5) | darknet |
| v2 | 2017 | 锚框 + 批归一化 + 暗网多尺度 | — (VOC 78.6 mAP@0.5) | darknet |
| v3 | 2018 | Darknet-53 + 三尺度预测 + 独立 Logistic | ~33.0 (test-dev, 608) | darknet / YOLOv3-tf |
| v4 | 2020 | CSP + PANet + BoF/BoS 分类法 | — (43.0 mAP@0.5 test-dev) | Bochkovskiy/YOLOv4 |
| v5 | 2020–22 | 训练配方工程化、AutoAnchor、EMA、AMP | 37.4 (n) → 55.0 (x) | ultralytics/yolov5 |
| YOLOX | 2021 | 解耦头 + SimOTA 标签分配 | 40.5 (S) → 51.5 (X) | MegviiEngine/YOLOX |
| v6 | 2021–22 | EfficientRep + TAL 分配器 + 解耦头 | ~36.0 (N) → ~53.1 (L) | meituan/YOLOv6 |
| v7 | 2022 | 可训练 Bag-of-Freebies + 重参数化 | ~51.3 (L) | WongKinYiu/yolov7 |
| v8 | 2024 | C2f + 解耦头 + Anchor-free 回归 + mosaic 收尾 | 37.3 (n) → 53.9 (x) | ultralytics/ultralytics |
| v9 | 2024 | PGI 梯度信息流 + GELAN | ~40.5 (t) → ~50.6 (E) | WongKinYiu/yolov9 |
| v10 | 2024 | 一致双分支标签分配，**无 NMS 端到端** | ~38.5 (S) → ~40.4 (L) | THU-MIG/yolov10 |
| v11 | 2024 | C3k2 + PSA 注意力块 | 39.5 (n) → 54.7 (x) | ultralytics |
| YOLOE | 2025 | 实时开放词汇检测 + 提示切换 | — | Alibaba/YOLOE |
| v12 | 2025 | 注意力中心架构，A2C 注意力→卷积 | 40.1 (n) → 55.2 (x) | ultralytics |
| v26 | 2025 | 端到端默认无 NMS、**移除 DFL**、MuSGD | 官方以最新 README 为准 | ultralytics |

---

## 🧭 三条主线读懂 YOLO

如果只记三件事，记这三条：

### 主线一：标签分配（Label Assignment）的持续博弈

这是 YOLO 历史上**最重要也最被低估**的技术主线，几乎每一代都在改它：

```
v1/v2   网格中心点硬分配（一个格子只能负责一个目标）
v3      每个格子固定负责 3 个 anchor（中心落在格子里即正样本）
v5      AutoAnchor + 固定 IoU 阈值正负样本划分
v6      TAL（Task-Aligned Assigner）——按分类得分与 IoU 加权动态选正样本
v7      保留 v5 方案
v8      TAL（TopK）——从 v6 借来
v9      保留 v8
v10     一致双分支（Consistent Dual Assignments）——训练用一对多，推理用一对一
v26     STAL（Stable Loss Assignment）——稳定版一对一分配
```

**结论**：标签分配的质量对 mAP 的影响，往往超过主干网络的换代。

### 主线二：从「Anchor 依赖」到「Anchor-free」的往复

```
v2/v3/v4/v5   强 Anchor 依赖（v5 的 AutoAnchor 甚至需要先跑一遍聚类）
v6/v8/v9      Anchor-free（v8 的 TaskAlignedAssigner 输出的是网格点）
v10           Anchor-free + 一对一分配（真正的 NMS-free 前提）
v26           Anchor-free + 端到端 + 无 DFL
```

中间 YOLOX/v6 还做过「解耦头」回潮。理解这条线能解释：**为什么 YOLOv8 比 YOLOv5 小很多但精度不降** —— 去掉 Anchor 意味着不需要为每个位置存 3 个先验框，也不需要 Anchor 的宽高比先验。

### 主线三：从「加模块」到「改信息流」

```
v4  加模块（CSP/PANet/Mish/cmBN/...）  —— 堆料
v5  加模块 + 工程化（EMA/AMP/损失改进）—— 堆料 + 秩序
v7  改训练结构（辅助头、重参数化）      —— 改信息流
v9  改梯度信息流（PGI/GELAN）          —— 改信息流
v12 改算子结构（注意力替代卷积、A2C）    —— 改算子
v26 改损失与输出表示（ProgLoss + 去 DFL）—— 改输出表示
```

---

## 📌 工程结论

1. **新项目默认从 YOLO11 或 v12 起步**，不要从 v5 起步（除非你的产线已经锁死 v5 生态）。v5 的价值在于**生态成熟度**（大量第三方插件、社区支持），而不是精度。
2. **追求极致精度且不介意速度** → 看 `02-视觉检测模型/`，DINO / DEIM / D-FINE 这类 DETR 系在 COCO 上已超过 YOLO 顶级型号。
3. **追求极致实时** → YOLOv10 / YOLOv8n / YOLO26n 端到端版本，注意后处理耗时常常才是真正的瓶颈（NMS 是）。
4. **做开放词汇 / 零样本** → YOLOE、YOLO-World、Grounding DINO，不要指望闭集 YOLO 微调出新类别。
5. **边缘部署** → 见 `08-导出部署与推理加速.md`。重点：**DFL 是 INT8 量化的主要精度杀手**，v26 移除 DFL 对边缘端是重大利好。

---

## 🔗 关联文档

- 理论全景 → [`../02-视觉检测模型/`](../02-视觉检测模型)
- 应用侧 → [`../03-RAG/`](../03-RAG)
- 术语速查 → [`../appendix/术语表.md`](../appendix/术语表.md)
