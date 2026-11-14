# 05 · 逆向分析工具链

> **前四条主线讲的是「怎么把模型跑起来」。这一条讲「怎么把别人做好的东西拆开看」。**

前四条主线里，训练和部署的对象都是**你自己写的代码**——源码在你手上，出问题直接看日志、调参、再训一轮。但工程上有一大类问题不是这样的：

- 现场设备跑不起来，手上只有一个 `.bin` / `.so` / `.apk`
- 客户给了一份 APK，要你查它到底访问了哪些域名
- 板子上固件升级失败，需要知道分区里到底装了什么
- 线上抓的 `.pcap` 打开一看几百个包，不知道该看哪个字段

这类问题不在「读源码」的范畴里，而在**逆向工程**的范畴里。工具链的选择、平台限制、版本对齐，构成了一条和前四条完全不同的工程线。

本目录的写法：**按「一次真实配置 / 一次真实实现」组织**。所有版本号和实测结果要么标「本机实测」，要么标「转述上游文档」，不混用。

---

## 📖 本目录文档

| # | 文档 | 核心问题 |
|---|---|---|
| 01 | [REA 逆向工程工具链配置实录](./01-REA-逆向工程工具链配置实录.md) | 把 REA 的 207 个工具全部点亮要装什么？为什么 Windows 上怎么折腾都少一半能力？ |
| 02 | [静态代码分析工具链：自建一个污点追踪嗅探器](./02-静态代码分析工具链-自建污点追踪嗅探器.md) | 静态扫描为什么必然误报？「有污点追踪」和「没污点追踪」是两种工具；自己写一个够用的版本，难点到底在哪 |

---

## ⚡ 一张表看懂「REA 的工具面怎么被操作系统切开」

> **口径**：REA `rea-agents` 6.7.0，共 207 个 MCP 工具。平台判定读自 `dist/server/createServer.js:100-123`。

| 能力域 | 引擎 | Windows | Linux | 代表工具 |
|---|---|---|---|---|
| 原生反编译 | Ghidra 12.1.4 | ✅ | — | `procedure_pseudo_code`、`analyze_function` |
| Android 静态分析 | jadx-headless-mcp 0.7.1 | ✅ | — | `inspect_android_method`、`search_android_classes` |
| Android 资源解码 | apktool 3.0.3 | ✅ | — | `decode_android_resources` |
| 设备交互 | adb 37.0.1 | ✅ | — | `read_adb_logcat`、`pull_adb_package` |
| WebAssembly | WABT 1.0.42 | ✅ | — | `inspect_wasm_artifact` |
| 浏览器场景 | Chrome 156 | ✅ | — | `capture_browser_scenario` |
| **ELF 布局** | pwntools 4.15.0 | ❌ | ✅ | `inspect_binary_layout` |
| **core dump** | pwntools 4.15.0 | ❌ | ✅ | `inspect_recorded_crash` |
| **固件拆解** | Unblob 26.6.4 / Binwalk 3.1.0 | ❌ | ✅ | `extract_firmware`、`inspect_firmware_regions` |
| **抓包分析** | TShark 4.6.6 | ❌ | ✅ | `inspect_packet_capture` |
| **信号解码** | sigrok-cli 0.7.2 | ❌ | ✅ | `decode_signal_recording` |
| **JS 混淆还原** | Wakaru 1.14.0 | ❌ | ✅ | `recover_javascript_sources` |
| **EVM 接口推断** | evmole 0.9.3 | ❌ | ✅ | `inspect_evm_interface` |

**Windows 侧拿到的是「反编译与分析」，Linux 侧拿到的是「格式拆解与取证」。** 两边互补。

---

## 🧭 三条不可绕过的结论

### 1. 平台门是代码写死的，不是「没装」

```js
// dist/server/createServer.js:100-123
const linux = process.platform === "linux";
const pwntools = linuxX64 && isAbsolute(environment.REA_PWNTOOLS_PYTHON ?? "");
const tshark    = linux     && isAbsolute(environment.REA_TSHARK_COMMAND ?? "");
```

**在 Windows 上把 pwntools 装进任何一个 Python 环境，它依然不可用。** 这类判断在每个 provider 内部还会再判一次（`PwntoolsDecoder.js:123`、`EvmoleInterfaceProvider.js:125`、`WakaruCommand.js:34`、`MitmproxyCaptureAdapter.js:19`）。

> 多 Python 版本帮不上忙——本机同时装着 3.11 / 3.12 / 3.13 / 3.14，一个都没用上，因为门在 `process.platform` 上。

### 2. 版本对齐比安装本身难

REA 在源码里写死了审计过的版本画像，并且**会主动校验**：

| 引擎 | REA 要求的画像 | 来源 |
|---|---|---|
| pwntools | `pwntools@4.15.0;pyelftools@0.33;unicorn@2.1.2` | `PwntoolsRelease.js` |
| Binwalk | `3.1.x`（审计于 3.1.0） | `FirmwareRelease.js` / `FirmwareVersion.js` |
| Unblob | `26.6.x`（审计于 26.6.4） | 同上 |
| sigrok-cli | `0.7.2` | — |
| mitmdump | `12.2.3` | `CaptureRelease.js` |
| jadx-headless-mcp | `0.7.1`（附 SHA-256） | `JadxRelease.js` |

版本差一点就是 `capability_unavailable`。装之前先 grep `dist/*/*Release.js`，能省几轮试错。

### 3. `doctor` 报绿 ≠ 引擎可用

`doctor` 只做**注册期**检查（node / host / hopper / ghidra / ida / jadx / kuna / ilspycmd），**不检查** tshark / sigrok / binwalk / unblob / wakaru / pwntools——那些是**执行期**检查。判断一个引擎到底能不能用，唯一可靠的办法是**拿真实输入调一次**。

### 4. 工具不报错 ≠ 工具做对了

`[`02`](./02-静态代码分析工具链-自建污点追踪嗅探器.md) 里有一个实测出来的例子：`--render` 扫本地文件时**静默空操作**——不报错、不警告，报告里只是「没有发现」。原因在 `cli.py:140`：渲染路径只对 `http://` / `https://` 目标生效。

> 这与 04 主线里「旧 sdkconfig 会盖掉 defaults」是同一类问题：**配置或环境不匹配时，工具链继续跑、继续给出「看起来正常」的结果。**
> **唯一的验证方式始终是拿真实输入调一次，并检查它是不是真的做了那件事。**

---

## 🧭 与其他主线的关系

| 主题 | 去哪看 |
|---|---|
| 导出产物（`.onnx` / `.rknn` / `.tflite`）的静态分析 | [`01-REA-逆向工程工具链配置实录`](./01-REA-逆向工程工具链配置实录.md) §5、§6 |
| RK3588S 板端部署与 NPU 推理 | [`../04-边缘部署与端侧AI/01-本机部署实战-YOLO26多后端实测.md`](../04-边缘部署与端侧AI/01-本机部署实战-YOLO26多后端实测.md) |
| ESP32 端侧固件与 OTA | [`../04-边缘部署与端侧AI/02-ESP32端侧视觉栈-ESP-VISION与ESP-DL.md`](../04-边缘部署与端侧AI/02-ESP32端侧视觉栈-ESP-VISION与ESP-DL.md) |
| 模型结构与权重文件本身的分析 | [`../01-YOLO/06-YOLO关键模块与算法.md`](../01-YOLO/06-YOLO关键模块与算法.md) |

---

## ⚠️ 关于本章数据的性质

- 本章的**全部版本号、路径、下载速度、内存阈值均为作者本机实测**（2026-11-10），会随 REA 与各上游版本升级失效，引用前请用 `rea doctor --json` 与 `dist/*/*Release.js` 复核。
- 本章的实测环境为 Windows 11 x64 + VMware 虚拟机的 Kali GNU/Linux Rolling 2026.2（x86-64）。**镜像速度、Python 预编译 wheel 覆盖情况、apt 源表现与网络位置强相关**，换网络环境需重新实测。
- **`[`02`](./02-静态代码分析工具链-自建污点追踪嗅探器.md) 的规则数量、严重度分布、各靶页面发现数与字段结构为作者本机实测**（2026-11-14），来自自建的零依赖静态分析工具。**该章只收录设计与取舍，不收录工具源码**；工具本身只做静态分析，不发送写请求、不执行目标脚本、不尝试利用任何漏洞。
- 本章**不含任何第三方源码与二进制文件**，只收录配置方法与实测结论。涉及的第三方工具版权归各自作者，遵循各自许可证。
