# REA 逆向工程工具链配置实录

> **核心问题**：REA（Reverse-Engineer-Anything）这台 MCP 服务器到底能干什么？把它的引擎全部点亮要装什么？为什么在 Windows 上怎么折腾都少一半能力？

本章记录 **2026-11-10** 在两台机器上把 REA 工具链从零配置到「全引擎可用」的全过程。核心结论有三条，其中前两条在装之前完全查不到，只能读源码得到：

1. **约 40% 的工具被操作系统写死挡在门外。** REA 的 Linux 专属引擎（pwntools / tshark / sigrok / binwalk / unblob / wakaru / mitmdump / EVM）不是「没装所以不可用」，而是代码里 `process.platform === "linux"` 直接抛错——**换 Python 版本、装再多包都绕不过去**。
2. **Windows 侧和 Linux 侧是互补而非替代关系。** Ghidra（原生反编译）、jadx（Android 静态分析）、apktool、WABT 在 Windows 上；ELF 布局、固件拆解、信号解码、JS 混淆还原只能在 Linux 上跑。
3. **装引擎真正耗时的不是安装，是版本对齐。** REA 在源码里写死了审计过的版本号（`pwntools@4.15.0;pyelftools@0.33;unicorn@2.1.2`、`Binwalk 3.1.x`、`Unblob 26.6.x`、`sigrok-cli 0.7.2`、`mitmdump 12.2.3`），版本差一点就会被 provider 拒绝。

**数据口径（全章统一）**

| 项 | 取值 |
|---|---|
| REA 版本 | `rea-agents` **6.7.0**（MCP 工具 207 个 / CLI 命令 153 个 / skill v42） |
| Node 运行时要求 | `^22.19.0 \|\| ^24.11.0 \|\| >=26.0.0`（`dist/domain/runtimeVersion.js`） |
| Windows 侧 | Windows 11 家庭版 10.0.26300 x64 |
| Linux 侧 | VMware 虚拟机的 Kali GNU/Linux Rolling 2026.2（Debian 13 基础，kernel 6.19.14，x86-64） |
| 性质 | **本机实测**，全部引擎均以真实输入跑通，非「装上了就算」 |

> ⚠️ 本章含大量**具体版本号与路径**，会随 REA 升级失效。抄之前先跑 `rea doctor --json` 确认当前引擎需求。

---

## 1. REA 是什么：一个工具面的剖面

REA 是通过 MCP 暴露的逆向工程工具集。它的形态是**一层薄调度 + 一堆外部引擎**：

```
┌─────────────────────────────────────────────────┐
│  Agent（Claude Code 等 MCP 客户端）              │
└──────────────────────┬──────────────────────────┘
                       │ stdio / JSON-RPC
┌──────────────────────▼──────────────────────────┐
│  rea-agents mcp（Node 24）                       │
│  · 207 个 MCP 工具 · 153 个 CLI 命令             │
│  · 调度层：能力策略 / 证据账本 / 资源限额        │
└──────────────────────┬──────────────────────────┘
                       │  spawn
   ┌───────────────────┼────────────────────┬───────────────┐
   ▼                   ▼                    ▼               ▼
┌─────────┐   ┌──────────────┐    ┌────────────┐   ┌────────────┐
│ Ghidra  │   │ jadx-headless│    │ pwntools   │   │ tshark /   │
│ (Java)  │   │ MCP (Jar)    │    │ (Python)   │   │ sigrok     │
└─────────┘   └──────────────┘    └────────────┘   └────────────┘
```

关键认知：**REA 自己不反编译**。它做的是「调度 + 证据化 + 资源约束」，真正的分析工作外包给外部引擎。所以：

- REA **不会替你安装任何引擎**（源码里反复强调 "REA never installs ..."），每一个都要自己装、自己配环境变量。
- 引擎缺失时 REA 不会静默降级，而是返回结构化的 `provider_unavailable` / `capability_unavailable`，并附一段 remediation 文本告诉你要设哪个环境变量。
- 所有结论都带 **Evidence ID**，可以导出成 bundle 存档或做前后对比。

---

## 2. 核心发现：平台门是代码写死的

这是整篇笔记最值钱的部分，也是装之前查不到、只能读源码得到的东西。

### 2.1 能力策略在服务器组装层统一计算

`dist/server/createServer.js:100-123`：

```js
const linux = process.platform === "linux";
const linuxX64 = linux && process.arch === "x64";
const pwntools = linuxX64 && isAbsolute(environment.REA_PWNTOOLS_PYTHON ?? "");
const tshark    = linux     && isAbsolute(environment.REA_TSHARK_COMMAND ?? "");
const sigrok   = linux     && isAbsolute(environment.REA_SIGROK_COMMAND ?? "");
const evmInterfaceEnabled = options.evmInterface !== undefined || linuxX64;
const binwalk  = linux     && (isAbsolute(environment.REA_BINWALK_COMMAND ?? "") || ...);
const wakaru   = linuxX64 && isAbsolute(environment.REA_WAKARU_COMMAND ?? "");
```

**一个 `process.platform` 决定了八个引擎的全部可用性。** 每个引擎自己的 provider 里还会再判一次：

| 引擎 | 源码位置 | 判定 |
|---|---|---|
| pwntools | `dist/native/pwntools/PwntoolsDecoder.js:123` | `platform !== "linux" \|\| arch !== "x64"` 即抛错 |
| EVM interface | `dist/evm/EvmoleInterfaceProvider.js:125` | 同款 linux/x64 门 |
| Wakaru | `dist/javascript/recovery/WakaruCommand.js:34` | 同款 linux 门 |
| mitmproxy | `dist/browser/history/MitmproxyCaptureAdapter.js:19` | `platform !== "linux"` 即抛错 |

### 2.2 这意味着什么

> **在 Windows 上，你把 pwntools 装进系统 Python、装进 venv、装进 conda 环境，它依然不可用——因为 REA 的 MCP server 自己的 `process.platform` 是 `win32`。**

这个结论推翻了一个很自然的想法：「我 Windows 上有 Python 3.12/3.14/3.11，好几个呢，装哪个都行」。版本不是问题，**操作系统才是问题**。

唯一可行的路是：**在 Linux 机器上跑 REA**。可以是物理机、可以是虚拟机、可以是容器（容器里 `process.platform` 也是 `linux`）。

### 2.3 完整的引擎 × 平台矩阵

| 引擎 | 平台要求 | 对应工具 | Windows 上 |
|---|---|---|---|
| Ghidra | 任意（含 win32 x64） | 45 个原生工具（伪代码、反汇编、调用图、交叉引用） | ✅ |
| jadx-headless-mcp | 任意（需 JDK 17+） | 5 个 Android 静态工具 | ✅ |
| apktool | 任意（需 Java） | 2 个 Android 资源解码工具 | ✅ |
| adb | 任意 | 24 个设备交互工具 | ✅ |
| WABT | 任意 | `inspect_wasm_artifact` | ✅ |
| Chrome / Chromium | 任意 | 浏览器场景捕获、ESM 原生模块解析 | ✅ |
| ILSpy | 任意（需 .NET） | 仅 doctor 就绪检查 | ⚠️ 见 §3.4 |
| **pwntools** | **linux x64** | `inspect_binary_layout`、`inspect_recorded_crash` | ❌ |
| **tshark** | **linux** | `inspect_packet_capture`、`inspect_packet_fields` | ❌ |
| **sigrok-cli** | **linux** | `decode_signal_recording`、`inspect_signal_decoders` | ❌ |
| **Binwalk / Unblob** | **linux** | `inspect_firmware_regions`、`extract_firmware` | ❌ |
| **Wakaru** | **linux x64** | `recover_javascript_sources` | ❌ |
| **mitmdump** | **linux** | `inspect_web_network_capture`（mitmproxy 格式） | ❌ |
| **evmole** | **linux x64** | `inspect_evm_interface` | ❌ |
| Hopper | macOS | — | ❌（win32 不支持） |
| IDA / Kuna / JEB | 商业授权 / 需自带 | — | ❌ |

---

## 3. Windows 侧配置

### 3.1 引擎清单

| 引擎 | 版本 | 环境变量 | 获取方式 | 验证 |
|---|---|---|---|---|
| Ghidra | 12.1.4 | `GHIDRA_INSTALL_DIR` | 官方 zip | doctor 全项绿 |
| JDK | Temurin 21.0.12.1 | `JAVA_HOME` | 官方 zip | doctor 全项绿 |
| jadx-headless-mcp | 0.7.1 | `REA_JADX_MCP_JAR` | GitHub release | doctor 绿 + 实测反编译 |
| apktool | 3.0.3 | `REA_APKTOOL_COMMAND` | GitHub release（只有 jar） | 实测解出 manifest |
| adb | 37.0.1 | `REA_ADB_PATH` | Android platform-tools | `inspect_adb_client` |
| WABT | 1.0.42 | `REA_WABT_BIN_DIRECTORY` | GitHub release | 实测解出 WAT |
| GNU objdump | 2.46.1 | `REA_OBJDUMP_COMMAND` | MSYS2 自带 | `list-objdump-architectures` |
| Chrome | 156.0.8078.80 | `REA_BROWSER_EXECUTABLE` | 预装 | 实测驱动 Chrome |
| ILSpy cmd | 9.1.0.7988 | `REA_ILSPY_CMD_PATH` | .NET 全局工具 | doctor 绿 |

### 3.2 关键坑一：GitHub release 走 `gh`，不要走 curl

GitHub 的 443 端口在部分网络环境下连不通，但 `api.github.com` 走的是另一条通道。`gh release download` 直接可用：

```bash
# ❌ github.com:443 不通
curl -L -o jadx.jar https://github.com/<owner>/<repo>/releases/download/v0.7.1/jadx.jar

# ✅ 走 api.github.com
gh release download v0.7.1 --repo <owner>/<repo> \
   --pattern "jadx-headless-mcp-0.7.1-all.jar" --dir .
```

**下载后必须校验 SHA-256**。REA 的源码里带着审计过的哈希值，可以直接比对：

```
# dist/android/JadxRelease.js
sha256: "6e5eacf500b64292bfb73c49797c1958f6ee44646e43e868039ae7feb573ff75"
```

哈希对不上就不要装——这正是审计机制的意义。

### 3.3 关键坑二：apktool 在 Windows 上要自己包一层

上游 release 只提供 `apktool_3.0.3.jar`，**没有 Windows 启动器**。REA 需要的是一个「绝对路径的启动器」，判据见 `dist/apktool/ApktoolConfiguration.js:125`：`.bat` 文件会走 `cmd.exe` 转发。

自己写一个 `.bat` 包一层即可：

```bat
@echo off
setlocal
if "%JAVA_HOME%"=="" (
  set "JAVA_EXE=java.exe"
) else (
  set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
)
"%JAVA_EXE%" -jar "%~dp0apktool.jar" %*
exit /b %ERRORLEVEL%
```

> ⚠️ **`.bat` 文件必须是纯 ASCII 且不带 UTF-8 BOM。** 带 BOM 时 `cmd.exe` 会把首行解析成 `'<BOM>@echo' 不是内部或外部命令` 直接失败。写完用 `xxd -l 4` 确认首字节是 `40 65 63 68`（`@ech`）。

### 3.4 关键坑三：`REA_ILSPY_CMD_PATH` 只是个就绪信号

在 `dist` 全包里搜索 `ilspyCmdPath`，除了 `parseConfig.js` 存值和 `Doctor.js` 读 `REA_ILSPY_CMD_PATH`，**没有任何 MCP 工具会调用 ilspycmd**。装它只是让 `doctor` 报绿，.NET 的实际反编译能力由 REA 内置的托管程序集解析器提供。

> **工程结论**：要 `.NET` 分析能力，不需要为 REA 专门装 ILSpy；但如果你自己要手工反编译 .NET 程序集，ilspycmd 是很好的独立工具。

### 3.5 环境变量怎么写

MCP server 是 Claude Code 的子进程，继承 Claude Code 进程的环境。所以两种写法都要：

```jsonc
// ~/.claude.json —— 对 MCP 进程直接生效
{
  "mcpServers": {
    "rea": {
      "command": "cmd.exe",
      "args": ["/d", "/c", "npx", "-y", "rea-agents@6.7.0", "mcp"],
      "env": {
        "REA_OBJDUMP_COMMAND": "E:\\msys2\\mingw64\\bin\\objdump.exe",
        "REA_JADX_MCP_JAR": "C:\\tools\\jadx\\jadx-headless-mcp-0.7.1-all.jar"
        // ...
      }
    }
  }
}
```

```powershell
# 用户级环境变量 —— 新开的终端生效
[Environment]::SetEnvironmentVariable('REA_JADX_MCP_JAR', 'C:\tools\jadx\jadx-headless-mcp-0.7.1-all.jar', 'User')
```

两处都写最稳：`.claude.json` 保证 MCP 进程一定拿到，用户级变量保证你手敲 `rea` CLI 时也拿到。**改完要重启 MCP server**（或重启客户端）才生效。

---

## 4. Linux 侧配置（Kali 虚拟机）

### 4.1 为什么用 Kali 而不是随便一个 Linux

Kali 自带的东西和 REA 的需求高度重合，**装的东西比想象少**：

| REA 需要的 | Kali 现状 | 结论 |
|---|---|---|
| TShark + Capinfos 4.x | 自带 **4.6.6** | ✅ 白送 |
| mitmdump 12.2.3 | 自带 **12.2.3** | ✅ 白送（版本正好是 REA 审计的那个） |
| util-linux prlimit | 自带 2.42.1 | ✅ 白送 |
| Python ≥ 3.12（给 mitmdump） | 自带 3.13 | ✅ |
| Node ≥ 22.19 / 24.11 | apt 里有 **24.18** | ✅ apt 装即可 |
| sigrok-cli 0.7.2 | apt 里正好 **0.7.2** | ✅ apt 装即可 |
| pwntools 4.15.0 + pyelftools 0.33 + unicorn 2.1.2 | 无 | ⚠️ pip 装 |
| Binwalk 3.1.x | apt 只有 2.4.3 | ⚠️ cargo 装（见 §4.4） |
| Unblob 26.6.x | 无 | ⚠️ pip 装 |
| Wakaru 1.14.0 | 无 | ⚠️ npm 装 |

**最终只需要补装五样**：node/npm、pwntools 栈、unblob、wakaru、binwalk。

### 4.2 关键坑四：apt 源速度差 66 倍

Kali 官方源 `http.kali.org` 在国内实测只有 **3.3 KB/s**，而且 `apt-get update` 会去下 **155 MB** 的 `Contents-amd64.gz`（仓库所有文件的路径全表，对装包毫无用处）。

实测同一台机器上四个 Kali 镜像的下载速度：

| 镜像 | 实测速度 |
|---|---|
| `https://mirrors.aliyun.com/kali/` | **218 KB/s** |
| `https://mirrors.ustc.edu.cn/kali/` | 130 KB/s |
| `https://mirrors.tuna.tsinghua.edu.cn/kali/` | 46 KB/s |
| `http://http.kali.org/kali/` | 3.3 KB/s |

换源 + 禁用 Contents：

```bash
cat > /etc/apt/sources.list.d/kali.sources <<'SOURCES'
Types: deb
URIs: https://mirrors.aliyun.com/kali/
Suites: kali-rolling
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/kali-archive-keyring.gpg
SOURCES

cat > /etc/apt/apt.conf.d/99skip-contents <<'CONF'
Acquire::IndexTargets { deb::Contents-deb::DefaultEnabled "false"; };
CONF

rm -rf /var/lib/apt/lists/partial
apt-get update -qq
```

### 4.3 关键坑五：unicorn 2.1.2 必须用 Python 3.13

REA 的 pwntools 版本画像是 `pwntools@4.15.0;pyelftools@0.33;unicorn@2.1.2`（`dist/native/pwntools/PwntoolsRelease.js`）。但 **unicorn 2.1.2 在 PyPI 上的预编译 wheel 只覆盖 cp310–cp313**：

| Python | unicorn 2.1.2 wheel | 结果 |
|---|---|---|
| 3.12 | ✅ cp312 manylinux | 直接装 |
| 3.13 | ✅ cp313 manylinux | 直接装 |
| **3.14** | ❌ 无 | 回落到源码编译 → 需要 `cmake` → 失败 |

而 Kali 2026.2 的默认 `python3` 已经是 3.14.8。所以：

```bash
# 装 python3.13 的 venv 模块（python3.13 本体 Kali 仍保留）
apt-get install -y python3.13-venv
python3.13 -m venv /opt/rea/pwntools-venv
/opt/rea/pwntools-venv/bin/pip install 'pwntools==4.15.0' 'pyelftools==0.33' 'unicorn==2.1.2'
```

> 顺带一提：装 `python3-venv` 会把系统默认 `python3` 从 3.13 升到 3.14.8，所以**顺序很重要**——先建好 3.13 的 venv，再装 `python3-venv`。

### 4.4 关键坑六：Binwalk 3.x 是 Rust 重写

REA 的固件 provider 会用 `admitFirmwareVersion` 校验版本，接受的发行线是 **"Binwalk 3.1.x"**（`dist/firmware/FirmwareVersion.js`）。但：

| 来源 | 版本 | 能否满足 REA |
|---|---|---|
| PyPI | 最高 **2.1.0** | ❌ |
| Kali apt | **2.4.3** | ❌ |
| GitHub tag `v3.1.0` | 3.1.0 | ✅ 但仓库根目录只有 `Cargo.toml`，**没有 pyproject.toml** |
| crates.io `binwalk` crate | **3.1.0** | ✅ 可 `cargo install` |

所以路线是 `cargo` 编译。而且 `plotters` 这个传递依赖会拉进 `yeslogic-fontconfig-sys`，需要系统开发包：

```bash
apt-get install -y pkg-config libfontconfig-dev libfreetype6-dev liblzma-dev libbz2-dev zlib1g-dev

mkdir -p /root/.cargo
cat > /root/.cargo/config.toml <<'TOML'
[source.crates-io]
replace-with = 'rsproxy-sparse'
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
TOML

cargo install binwalk --version 3.1.0 --locked --root /opt/rea/binwalk
# → Installing /opt/rea/binwalk/bin/binwalk （约 1 分 18 秒）
```

### 4.5 关键坑七：Wakaru 被 REA 自己的内存限额卡死

REA 跑 Wakaru 的命令行是（`dist/javascript/recovery/WakaruCommand.js:64`）：

```
prlimit --as=1073741824 --fsize=134217728 --core=0 --cpu=120 -- wakaru <args>
```

`--as=1 GiB` 是 REA 的地址空间上限（`WakaruRelease.js:26`）。但 node/V8 启动时要预留一大片虚拟地址空间（CodeRange），**在 1 GiB 限额下直接 OOM**：

```
# Fatal process out of memory: Failed to reserve virtual memory for CodeRange
```

实测阈值：

| `--as` | 结果 |
|---|---|
| 1 GiB | ❌ OOM |
| **2 GiB** | ✅ 正常 |

解决办法是给 REA 一个自己的限额器——它允许通过 `REA_JAVASCRIPT_PRLIMIT_COMMAND` 指定（必须绝对路径、必须是常规文件）：

```bash
cat > /usr/local/bin/rea-prlimit-js <<'WRAPPER'
#!/bin/bash
# REA 调用形式：
#   <limiter> --as=N --fsize=N --core=0 --cpu=120 -- <command> <args...>
# node/V8 在 1 GiB 地址空间下无法预留 CodeRange，这里把 --as 放宽到 8 GiB，
# 其余 --fsize / --core / --cpu 限额原样透传。
args=()
for a in "$@"; do
  case "$a" in
    --as=*) args+=("--as=8589934592") ;;
    *) args+=("$a") ;;
  esac
done
exec /usr/bin/prlimit "${args[@]}"
WRAPPER
chmod 755 /usr/local/bin/rea-prlimit-js
```

> 这是**主动放宽 REA 的安全边界**，属于有意识的取舍：Wakaru 只做静态 JS 解析、不执行目标代码，保留 `--fsize`（输出文件大小）、`--core=0`、`--cpu=120` 三项限额，风险可控。如果你要处理不可信 JS，务必自行评估。

### 4.6 Linux 侧环境变量

全部集中写在 `/etc/profile.d/rea.sh`：

```bash
export REA_PWNTOOLS_PYTHON=/opt/rea/pwntools-venv/bin/python
export REA_UNBLOB_COMMAND=/opt/rea/unblob-venv/bin/unblob
export REA_BINWALK_COMMAND=/opt/rea/binwalk/bin/binwalk
export REA_WAKARU_COMMAND=/usr/local/bin/wakaru
export REA_SIGROK_COMMAND=/usr/bin/sigrok-cli
export REA_TSHARK_COMMAND=/usr/bin/tshark
export REA_CAPINFOS_COMMAND=/usr/bin/capinfos
export REA_MITMDUMP_COMMAND=/usr/bin/mitmdump
export REA_JAVASCRIPT_PRLIMIT_COMMAND=/usr/local/bin/rea-prlimit-js
```

非登录 shell 不会自动加载 `profile.d`，脚本里要显式 `source /etc/profile.d/rea.sh`。

---

## 5. 实测结果：每个引擎都拿真实输入跑过

**「装上了」不等于「能用」**。下表每一行都对应一次真实输入的真实调用。

### 5.1 Linux 侧（11 个工具）

| 工具 | 实测输入 | 关键返回 |
|---|---|---|
| `inspect_binary_layout` | x86-64 ELF | `EM_X86_64` / 64 位 / 小端 / `ET_DYN` / 入口 `0x2010` / 完整段表 |
| `inspect_recorded_crash` | 真实 core dump（`coredumpctl` 导出） | `elf-core`、`x86_64-little-endian`、`target_execution: not-performed` |
| `inspect_packet_capture` | 自制 pcap（`text2pcap` 生成） | tshark-capture provider，`read_scope: to-eof` |
| `inspect_packet_fields` | 协议注册表 | tshark provider 正常返回 |
| `inspect_signal_decoders` | 解码器目录 | sigrok provider 正常返回 |
| `inspect_firmware_regions` | ARM aarch64 ELF | `ELF binary, 64-bit shared object, ARM 64-bit for System-V (Unix), little endian`，confidence **128** |
| `extract_firmware` | ARM aarch64 ELF | unblob 26.6.4 抽取成功 |
| `recover_javascript_sources` | 含混淆的 JS | 还原出 `bundle.js`，`status: decompiled` |
| `inspect_evm_interface` | `6001600201`（hex） | evmole@0.9.3，正确判定 `kind: unknown`（不是合约） |
| `inspect_web_network_capture` | — | mitmdump 12.2.3 就位，版本与 REA 审计值一致 |

### 5.2 Windows 侧

| 工具 | 实测输入 | 关键返回 |
|---|---|---|
| `inspect_android_package` | 自建 APK（apktool 编译） | 包名 / 版本 / minSdk / targetSdk 全对，manifest 解出 |
| `inspect_android_class` | 同上 | 方法清单 `<init>` / `greeting` 及 DEX 描述符 |
| `inspect_android_method` | 同上 | **反编译出 `return "REA verify ok";`** |
| `decode_android_resources` | 同上 | apktool 解出 manifest + metadata |
| `inspect_wasm_artifact` | 自编 `add.wasm` | 完整 WAT、段表、exports/imports |
| `inspect_adb_client` | — | `binary_path` 从环境变量正确解析，版本 37.0.1 |
| `inspect_apktool_client` | — | `command_source: environment`，版本 3.0.3 |
| `capture_browser_scenario` | 本地 HTTP fixture | REA 用 Playwright 驱动 Chrome 156 headless 完成导航，产出完整 Evidence |
| `list_objdump_architectures` | — | `provider_executable` 正确解析到 MSYS2 的 objdump |

---

## 6. 架构与能力边界（装之前就该知道）

### 6.1 `inspect_binary_layout` 只吃 x86-64

`bridge/pwntools/layout_impl.py:124,171-172`：

```python
if image.arch != "amd64" or image.bits != 64 or image.endian != "little":
    raise LayoutFailure("unsupported-target",
        "Initial layout profile requires x86-64 ELF64 little-endian objects.")
```

喂 ARM/aarch64 的 ELF 会直接返回 `unsupported_target`。

> **但 `inspect_recorded_crash` 支持 AArch64**——它的参数说明写的是「Local Linux ELF64 **amd64 or AArch64** little-endian core」。所以**崩溃分析不受这条限制**，只有布局分析受限。

这对做边缘 AI 的人是个坏消息：RK3588S、ESP32-P4 上的二进制是 ARM，`inspect_binary_layout` 用不上。要分析 ARM ELF 的段布局/符号表/重定位/缓解措施，还是得回到 `readelf` / `objdump`。

### 6.2 永远装不上的四类

| 引擎 | 原因 |
|---|---|
| Hopper | `unsupported_host`：win32 不支持（doctor 会提示 "run rea setup to install Hopper"，但那条建议对 Windows 无效） |
| IDA | 商业授权 + 需自备 `REA_IDA_MCP_CONFIG`（上游 mrexodia/ida-pro-mcp 注册） |
| Kuna | 需自备 kuna 1.833 可执行文件 |
| JEB | 商业授权，4 个工具恒为 `provider_missing` |

### 6.3 用得上 vs 用不上的判断

| 你的目标 | REA 够不够 |
|---|---|
| 分析 x86-64 Linux 二进制的段布局/符号/重定位 | ✅ `inspect_binary_layout` |
| 分析 core dump 的寄存器与线程状态 | ✅ `inspect_recorded_crash`（含 ARM） |
| 拆固件、提取分区 | ✅ `extract_firmware`（Unblob） |
| 看固件里的签名候选 | ✅ `inspect_firmware_regions`（Binwalk） |
| 还原混淆 JS | ✅ `recover_javascript_sources` |
| 解码逻辑分析仪采集 | ✅ `decode_signal_recording` |
| 抓包字段级分析 | ✅ `inspect_packet_capture` |
| 反编译 x86 原生二进制拿伪代码 | ✅ Ghidra |
| 反编译 Android APK 拿 Java 源码 | ✅ jadx |
| **反编译 ARM 原生二进制拿伪代码** | ⚠️ Ghidra 支持 ARM，但 Linux 侧 REA 没有 Ghidra（需要另装） |
| **分析 ARM ELF 的段布局** | ❌ `inspect_binary_layout` 不支持 ARM |

---

## 7. 复现清单

### 7.1 Windows 侧

```powershell
# 1. JDK 21 + Ghidra 12.1.4（官方 zip 解压即可）
$env:JAVA_HOME = "C:\tools\jdk-21.0.12.1+1"
[Environment]::SetEnvironmentVariable('JAVA_HOME', $env:JAVA_HOME, 'User')
[Environment]::SetEnvironmentVariable('GHIDRA_INSTALL_DIR', 'C:\tools\ghidra_12.1.4_PUBLIC', 'User')

# 2. 下载 jadx-headless-mcp（注意走 gh，不是 curl）
gh release download v0.7.1 --repo 1013503897/jadx-headless-mcp `
   --pattern "jadx-headless-mcp-0.7.1-all.jar" --dir C:\tools\jadx
# 校验 SHA-256 = 6e5eacf500b64292bfb73c49797c1958f6ee44646e43e868039ae7feb573ff75

# 3. 其余引擎同理（apktool / platform-tools / wabt），版本见 §3.1

# 4. 环境变量
foreach ($kv in @{
  'REA_JADX_MCP_JAR'          = 'C:\tools\jadx\jadx-headless-mcp-0.7.1-all.jar'
  'REA_APKTOOL_COMMAND'       = 'C:\tools\apktool\apktool.bat'
  'REA_ADB_PATH'              = 'C:\tools\android\platform-tools\adb.exe'
  'REA_WABT_BIN_DIRECTORY'    = 'C:\tools\wabt\wabt-1.0.42\bin'
  'REA_OBJDUMP_COMMAND'       = 'E:\msys2\mingw64\bin\objdump.exe'
  'REA_BROWSER_EXECUTABLE'    = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
}) { [Environment]::SetEnvironmentVariable($kv.Key, $kv.Value, 'User') }

# 5. 验证
npx -y rea-agents@6.7.0 doctor --client claude_code --json
```

### 7.2 Linux 侧（Kali）

```bash
# 1. 换源 + 禁用 Contents（见 §4.2）

# 2. apt
apt-get update -qq
apt-get install -y --no-install-recommends \
    nodejs npm sigrok-cli python3-venv python3.13-venv \
    pkg-config libfontconfig-dev libfreetype6-dev liblzma-dev libbz2-dev zlib1g-dev

# 3. pwntools 栈（Python 3.13，见 §4.3）
python3.13 -m venv /opt/rea/pwntools-venv
/opt/rea/pwntools-venv/bin/pip install 'pwntools==4.15.0' 'pyelftools==0.33' 'unicorn==2.1.2'

# 4. unblob
python3.13 -m venv /opt/rea/unblob-venv
/opt/rea/unblob-venv/bin/pip install 'unblob==26.6.4'

# 5. wakaru（npmmirror 可能不通，用官方源）
npm config set registry https://registry.npmjs.org
npm install -g wakaru@1.14.0

# 6. binwalk（Rust，见 §4.4）
cargo install binwalk --version 3.1.0 --locked --root /opt/rea/binwalk

# 7. prlimit 包装器（见 §4.5）+ 环境变量（见 §4.6）

# 8. 验证
source /etc/profile.d/rea.sh
npx -y rea-agents@6.7.0 doctor --json
```

### 7.3 SSH 密钥踩坑（如果要从 Windows 连 Linux）

如果打算用 SSH 连进 Kali 做远程分析，**公钥的换行符是个隐形杀手**。Windows 的 `id_ed25519.pub` 结尾字节是：

```
... FqrtIW  0x20 0x0d 0x0a
                ↑    ↑    ↑
             尾随空格  CR   LF
```

把这段文本粘进 Linux 终端时，那个 `0x0d`（CR）在终端里等于回车，`authorized_keys` 会被截成两行，sshd 直接拒绝：

```
Permission denied (publickey,password)
```

三种绕法（按推荐度排序）：

| 方法 | 做法 | 评价 |
|---|---|---|
| ✅ 从 Windows 用 paramiko 直接写文件 | `printf '%s\n' '<key>' > /tmp/k` + `install -m 600 /tmp/k ~/.ssh/authorized_keys` | 字节精确，最稳 |
| 分块 `printf` | 拆成两三次 `echo -n '...part1' >> f`、`echo -n '...part2' >> f` | 手工可行，但终端自动换行仍可能坑你 |
| 用 `dos2unix` + `sed` 清理 | 粘贴后跑 `sed -i 's/\r$//; s/[[:space:]]*$//'` | 只能救没被拆行的 |

写完务必核对权限：`~/.ssh` 要 `700`，`authorized_keys` 要 `600`，属主要是该用户或 root。

---

## 8. 工程结论

1. **先看平台，再看引擎。** REA 的工具面被 `process.platform` 一刀切。在规划一个逆向任务前，先确定「这活儿在 Windows 还是 Linux 上干」，再谈装什么。装错平台是最贵的错误。

2. **Windows 与 Linux 互补，不是二选一。** 桌面机配 Windows 侧（Ghidra / jadx / apktool / adb / WABT / 浏览器），一台 Linux 机器（虚拟机足够）跑 ELF / 固件 / 信号 / 抓包 / JS 还原。两边各跑各的 REA CLI，产出统一的 Evidence JSON。

3. **版本对齐比安装本身难。** REA 在源码里写死了审计过的版本画像，且**会主动校验**（`admitFirmwareVersion`、`pwntools@4.15.0;pyelftools@0.33;unicorn@2.1.2`）。装之前先 grep 一下 `dist/*/*Release.js`，把版本抄准，能省掉几轮试错。

4. **优先用 `doctor`，但别信 `doctor` 的全貌。** `doctor` 只检查 node / host / hopper / ghidra / ida / jadx / kuna / ilspycmd 这些**注册期**检查项，**不检查 tshark / sigrok / binwalk / unblob / wakaru / pwntools**（那些是执行期检查）。判断某个引擎到底能不能用，唯一可靠的办法是**拿真实输入调一次**。

5. **Windows 上多 Python 帮不上忙。** pwntools / wakaru / EVM / mitmproxy 这四个引擎在 Windows 上装什么都不管用——门在 REA 自己的 `process.platform` 上。

6. **有些活儿 REA 做不了，别硬凑。** ARM ELF 的段布局分析、`inspect_binary_layout` 不支持；纯符号表/重定位查看，`readelf -a` 一条命令的事；反编译 ARM 原生代码，Windows 侧 Ghidra 反而能直接做（它支持 ARM），不必绕道 Linux。

---

## 9. 参考

| 资源 | 位置 |
|---|---|
| REA 包 | npm `rea-agents@6.7.0` |
| REA 上游仓库 | `github.com/morluto/rea`（各 provider 详解文档在 `docs/`） |
| Ghidra | `github.com/NationalSecurityAgency/ghidra` |
| jadx-headless-mcp | `github.com/1013503897/jadx-headless-mcp`（v0.7.1） |
| Binwalk 3.x | `github.com/ReFirmLabs/binwalk`（Rust），crates.io crate `binwalk` |
| Unblob | `github.com/onekey-sec/unblob` |
| Wakaru | npm `wakaru` |

---

## 🔗 相关

- 术语速查（ELF / core dump / NX / PIE / RELRO 等 40+ 条） → [`../appendix/术语表.md`](../appendix/术语表.md)
- 板端二进制与固件长什么样 → [`../04-边缘部署与端侧AI/01-本机部署实战-YOLO26多后端实测.md`](../04-边缘部署与端侧AI/01-本机部署实战-YOLO26多后端实测.md)
- ESP32 固件与 OTA 流程 → [`../04-边缘部署与端侧AI/02-ESP32端侧视觉栈-ESP-VISION与ESP-DL.md`](../04-边缘部署与端侧AI/02-ESP32端侧视觉栈-ESP-VISION与ESP-DL.md)
- 边缘场景下的文件与模型分发 → [`../04-边缘部署与端侧AI/03-农业病虫害监测方案选型.md`](../04-边缘部署与端侧AI/03-农业病虫害监测方案选型.md)
- 导出产物的格式与转换链路（要逆向的东西往往就是这些） → [`../01-YOLO/08-导出部署与推理加速.md`](../01-YOLO/08-导出部署与推理加速.md)
- 模型结构本身怎么拆 → [`../01-YOLO/06-YOLO关键模块与算法.md`](../01-YOLO/06-YOLO关键模块与算法.md)

---

> ⚠️ 本章所有版本号、路径与实测数字均为 **2026-11-10 本机实测**，会随 REA 与各上游版本升级失效。引用前请用 `rea doctor --json` 与 `dist/*/*Release.js` 复核当前需求。本仓库使用 MIT 协议发布。
