# 01 · RH 的 Lean 形式化工程解剖

> **本章要回答的问题**：一个「据说证明了黎曼猜想」的 Lean 项目，社区接手之后到底把它变成了什么？验证装置是怎么搭的？1 GB 的证明长什么样？

**本章口径**：所有本地实测数字取自上游仓库 `leanprover-community/riemann-hypothesis` 的 `main` 分支 @ `26c37b400bc9cfd43c81d798aa5e798af5c55b94`（提交时间 2026-11-24T12:39:27Z）。原始归档的数字转述自上游 `verification/NOTES.md`。

> ℹ️ **本节不复制上游源码。** 上游 README 明确写着这份提交「published without a license」——版权属于原作者。因此本章用**数学记号 + 散文**描述五个定理的陈述，源码请直接看上游 `src/Statements.lean` 与 `src/Main.lean`（合计不到 60 行）。

---

## 1. 起点：一份公告、一个归档、一个仓库

**2026-11-20**，数学家 Levent Alpöge 公开宣布他用 Lean 4 完成了黎曼猜想的形式化证明，并给出了两样东西：

| 东西 | 地址 | 说明 |
|---|---|---|
| 论文 | `alpo.ge/remark.pdf` | 公告正文，sha256 `3fc97784b71ddd44ff8aa0a177e99622a1c61b13cfa843a0989ee4a7a0cfeb8d` |
| 源码归档 | `alpo.ge/lean.tar.gz.part01` … `part14` | 切成 14 段分发，拼接后 239,275,554 字节，sha256 `9631b4f020199e580ba86178cb56c6e9b09f690c2369a80dc4334aa93f49c7fc` |

同一天，Lean 社区的 `leanprover-community` 组织建了 `riemann-hypothesis` 仓库，**第一个提交 `40e56bab`（2026-11-20T14:36:09Z，提交信息「Mirror the submission exactly as published」）就是那个归档的逐字节副本**。

这个仓库的自我定位写得非常克制：

> **Independent Comparator setup** for the Lean formalization of the Riemann Hypothesis

注意它**没有**说「这是 RH 的证明」。它说的是「这是**检验**装置」。这个措辞差别是整个项目的立身之本：**社区不担保证明正确，社区担保「任何人可以独立复现这次检查」**。

上游 README 有一句必须原样记住的话：

> The submission belongs to its author and was published without a license.

即：**这份提交没有许可证**。社区镜像它是为了能被检验，不是因为获得了再分发授权。这一点在 [`02` §6](./02-压缩分发与供应链校验.md#6-许可证分发一份无许可证的证明) 里会变成一个实际约束。

### 1.1 四天之内长出来的东西

| 指标 | 2026-11-24 实测 |
|---|---:|
| Star / Fork | 961 / 357 |
| Open issues | 221 |
| Open PR | ≥ 100 |
| 已合并 PR | 583 |
| 提交总数 | ≈ 2,000 |
| 仓库体积（GitHub 计） | 976,257 KB ≈ 953 MiB |
| CI workflow | 9 个 |

583 个已合并 PR、4 天。**这个吞吐量本身就是本章要讲的现象**：一个需要数小时编译、且每个 PR 都要跑完整构建检查的项目，是怎么做到 4 天 583 个 PR 的？答案在 §7。

---

## 2. 目录导览

```
riemann-hypothesis/
├─ src/                       ← 证明源码本体（60,493 个 .lean）
│  ├─ Main.lean              ← 五个定理的最终陈述（2.3 KB）
│  ├─ Statements.lean        ← 五个命题的定义（0.9 KB）
│  ├─ Challenge.lean         ← 同样五个命题，但用 sorry（挑战文件，禁止修改）
│  ├─ Development/           ← 41,838 个模块，11.9M 行（占 71%）
│  ├─ Formalism/             ←  9,918 个模块
│  ├─ GroundTheory/          ←  5,225 个模块
│  ├─ ReciprocityTheory/     ←  2,423 个模块
│  ├─ CompanionTheory/       ←    706 个模块
│  ├─ PositivityTheory/      ←    227 个模块
│  └─ Lemmata/               ←    153 个模块
├─ comparator.json            ← 验证装置的配置（禁止修改）
├─ comparator/                ← 第二套比较器（对标 formal-conjectures）
├─ export-targets.txt         ← 导出时要带上的名字清单
├─ lakefile.toml / lake-manifest.json / lean-toolchain
├─ verification/              ← 验证记录：NOTES.md / CACHE.md / PORTABLE-CACHE.md
├─ scripts/                   ← CI 与缓存流水线（约 40 个脚本，60 万字节）
├─ tools/ci_native/           ← 自建 runner 侧的原生实现
├─ docs/references.bib        ← 21 条参考文献
└─ AGENTS.md                  ← 给 AI agent 的行为规范
```

**`src/` 与「基础设施」是分开的**：`lakefile.toml` 顶部三行是社区加的缓存配置，末尾有一行分隔注释，说明之后是社区加的 `Challenge` 库和 `cache` 可执行文件。这样任何时候都能用 `git diff 40e56bab HEAD -- src` 精确列出「社区对原作者提交动过什么」。

---

## 3. 五个定理：怎么陈述的

上游 `src/Statements.lean` 用四条 `Prop` 定义加一条 Mathlib 自带的 `RiemannHypothesis`，`src/Main.lean` 逐个给出证明。用数学记号重述如下。

### 3.1 四条定义

**(a) ζ 函数的零点在临界线上** —— 记作 `RiemannZetaZerosOnLine`：

> 对任意复数 $s$：若 $\zeta(s) = 0$，且 $s$ 不是平凡零点（即 $s \ne -2, -4, -6, \dots$），且 $s \ne 1$，则 $\operatorname{Re}(s) = \tfrac12$。

**(b) L 函数在右半平面不消失** —— 记作 `DirichletLNonvanishingRightOfHalf`：

> 对任意模 $N$、任意模 $N$ 的狄利克雷特征 $\chi$（复值）、任意复数 $s$：若 $\operatorname{Re}(s) > \tfrac12$，且 $(\chi, s) \ne (\mathbf{1}, 1)$，则 $L(\chi, s) \ne 0$。

**(c) L 函数的零点在临界线上** —— 记作 `DirichletLZerosOnLine`：

> 对任意模 $N$、任意特征 $\chi$、任意复数 $s$：若 $0 < \operatorname{Re}(s) < 1$ 且 $L(\chi, s) = 0$，则 $\operatorname{Re}(s) = \tfrac12$。

**(d) 本原特征的特例** —— 记作 `PrimitiveDirichletLZerosOnLine`：把 (c) 里的「任意特征」换成「任意**本原**特征」，结论不变。

外加 Mathlib 自带的 `RiemannHypothesis`（数学家熟悉的那条）。

### 3.2 五个定理名

| 定理名 | 对应 | 证明难度（上游实现） |
|---|---|---|
| `riemannHypothesis` | Mathlib 的 RH | 一次函数复合：`Claim.zeta` 把 (b) 变成 ζ 的版本 |
| `riemannZeta_eq_zero_re` | (a) | 直接取 `riemannHypothesis` |
| `LFunction_ne_zero_of_half_lt_re` | (b) | **最核心的一条**，来自 `UpperTheory.FinalTheorems.everyCharacter` |
| `LFunction_eq_zero_re` | (c) | `Claim.strip` 从 (b) 推出 |
| `LFunction_eq_zero_re_of_isPrimitive` | (d) | 引入本原性后逐项转交 (c) |

**注意这份证明的形状**：真正承重的只有一条（(b)），其余四条是它的直接推论。`riemannZeta_eq_zero_re` 甚至就是 `riemannHypothesis` 的别名。

### 3.3 四个技术点，逐个看

**① 为什么要五个定理，而不是一个？**
因为 RH 在数学上有多个精度不同的版本：黎曼 ζ 函数一个、狄利克雷 L 函数全体一个、本原特征一个。**如果只验 ζ 函数，其他所有基于 L 函数的推论都没有形式化依据**。社区的做法是：五个都验，且把本原特征那条单独列出来——哪怕它在 Lean 里只是一次转交，单独列出来是为了让「哪些结论被覆盖」在报告里一目了然。

**② 排除平凡零点和 $s = 1$**
- $s \ne 1$：Mathlib 的 `riemannZeta` 在 $s = 1$ 处是「垃圾值」（junk），不是真的零点。必须排除。
- 排除 $-2, -4, -6, \dots$：ζ 函数的**平凡零点**显然不在临界线上。必须排除，否则陈述为假。

上游 `verification/NOTES.md` 明确写了这两个排除是**逐字读出来的**，不是猜的。

**③ $(\chi, s) \ne (\mathbf{1}, 1)$ 这一支**
平凡特征（principal character）模 1 的 L 函数就是 ζ 函数，它在 $s = 1$ 处是极点。所以「$\operatorname{Re}(s) > \tfrac12$ 时 $L(\chi,s) \ne 0$」这条陈述必须把这个点单独挖掉。

**④ $0 < \operatorname{Re}(s) < 1$ 的射程问题**
(c) 只讨论临界带内的零点；(b) 说的是 $\operatorname{Re}(s) > \tfrac12$ 恒不为零。两条合起来能推出「临界带内的零点实部都是 $\tfrac12$」，但 **$\operatorname{Re}(s) \le 0$ 的情形不在射程内**。

上游 NOTES.md 专门有一段讲这件事，并说明 `comparator/` 里那套 formal-conjectures 比较器补上了这个缺口（见 §4.3）—— 那套陈述用的是「对所有 $s$ 且 $s$ 不在平凡零点集里」的写法，覆盖面更宽。

### 3.4 `src/Challenge.lean`：把证明换成 `sorry`

`src/Challenge.lean`（1.2 KB）是同样五个定理，但证明体是 `sorry`。它的作用是**给比较器一个「标准答案」**：

```
Comparator 流程：
   编译 Challenge（带 sorry）  ──┐
                                  ├── 比对：陈述是否一致 / 用了几条公理 / 内核是否接受
   编译 Main（真实证明）      ──┘
```

如果 `Main.lean` 证明的东西和 `Challenge.lean` 不是一回事，Comparator 立刻报错。**这解决了「你证明的是不是 RH」这个问题**——不是靠读代码，是靠机器比对常量的依赖闭包。

---

## 4. Comparator：独立验证装置的核心

`comparator.json` 的五个字段（社区新增，禁止修改）：

| 字段 | 值 |
|---|---|
| `challenge_module` | `Challenge` |
| `solution_module` | `Main` |
| `theorem_names` | 上述五个定理名 |
| `permitted_axioms` | `propext`、`Quot.sound`、`Classical.choice` |
| `external_kernels` | `leanchecker-paranoid`（Lean paranoid）与 `nanoda_bin`（nanoda） |

### 4.1 公理白名单：这条检查抓什么

Lean 的 `theorem` 背后可能藏着 `Classical.choice`（经典选择公理）、`propext`（命题外延）、`Quot.sound`（商同余）这三条「标准但不算逻辑」。除了这三条，**任何**公理都不许用——包括 `sorryAx`（`sorry` 会引入它）。

| 公理 | 是什么 | 为什么允许 |
|---|---|---|
| `propext` | 命题外延：$(a \leftrightarrow b) \to a = b$ | Lean 里 `Prop` 是类型论意义上的命题，逻辑等价必然可替换 |
| `Quot.sound` | 商同余 | Mathlib 里 `Quot` / `Quotient` 的定义基础 |
| `Classical.choice` | 经典选择 | Lean 的可判定性、连续统假设类论证的标准基础 |

**这道检查的真正威力在于排除 `sorryAx`。** 一个 17 万模块的项目，你没法靠 code review 找出藏在第 12,000 个文件里的一个 `sorry`。但 `sorryAx` 一旦出现在 axiom 列表里，Comparator 直接失败。

上游 CI 还会做一遍**纯文本静态扫描**（`verification/static-scan.txt`）：全文搜索 `axiom`、`sorry`、`admit`、`native_decide`、`implemented_by`、`extern`、`unsafe`、`opaque`、`debug`、`skipKernelTC`。扫描结论：原始源码里**一处都没有**。

> ⚠️ **静态扫描不是证明。** 上游在 NOTES.md 里自己写了这句话。文本搜索只能证明「没有用这些关键字」，不能证明「没有用别的手段绕过检查」——所以才需要 §4.2。

### 4.2 两个外部内核：为什么 Lean 自己验自己不够

`external_kernels` 里配了两个**不是 Lean 官方工具链**的独立内核：

| 内核 | 是什么 | 为什么要它 |
|---|---|---|
| `leanchecker-paranoid` | Lean 社区的 paranoid checker，把内核里所有「优化过的路径」都关掉，走最保守的检查 | 排除 Lean 内核里**任何**可能过于宽松的实现 |
| `nanoda_bin` | 第三方从零实现的 Rust 内核 | 排除「Lean 内核自己有 bug」这一类风险 |

**这是整个项目最重要的设计。** 逻辑上，形式化证明的可信度最终落在「内核是对的」这个假设上。如果只有一个内核，那么「内核有 bug」和「证明有 bug」是无法区分的。**两个互相独立实现的内核都接受，才能说「这条证明链路上没有单点故障」。**

导出机制是 `leanexport`：把编译产物里指定的一组名字导出成可被外部内核读取的二进制格式。`export-targets.txt` 里除了五个定理名，还显式列出了三条允许的公理，以及一批 `Nat` / `String` / `List` 的标准库构造子——**这些是判定「两个导出是否在做同一件事」所需的锚点**。缺一个，导出就可能因为共享前提不同而比较失真。

### 4.3 第二套比较器：对标 formal-conjectures

`comparator/` 目录下还有一套独立配置，判的是**别人早就写好的定理陈述**——Google DeepMind 的 `formal-conjectures` 仓库里的 `RiemannHypothesis.lean`（commit `d8d031d4`，2026-11-19）。这是社区在做的事里最值得学的一点：

> **自己写的挑战文件只能证明「我对上了自己的口径」；别人早先写好的定理陈述才能证明「这个证明说的是大家以为的那个东西」。**

工程上有个坑：formal-conjectures 的原文编译不过本项目锁定的 Lean / Mathlib 版本（`v4.33.1` → `v4.36.0` 升级后失效），所以 `comparator/FormalConjecturesChallenge.lean` 是它的**手工副本**，只允许三处差异：

1. 实例名不同（原版里 Lean 自动生成的名字带一串后缀，副本把它写出来，因为 Comparator **按常量名匹配**）；
2. `trivialZeros` 标了 `noncomputable`（`v4.36.0` 强制要求）；
3. 删掉了原版里的 `category` 和 `AMS` 属性。

CI 每次都会去拉取 formal-conjectures 在那个 commit 的两个文件，**逐字比对副本**，确保只有这三处差异、且副本里除了注释和 import 之外没有自己的声明。这个副本还做过一次实跑验证：在 formal-conjectures 自己的 pin 上编译副本，与 formal-conjectures 自己的文件做比较（两者都是 `sorry`），结果一致，且只在允许 `sorryAx` 时通过、不允许时只报 `sorryAx` 一个问题。

### 4.4 完整的独立检查流程

上游 README 的「自己验一遍」是三步：

1. **只取 Mathlib 的缓存**（`lake exe mathlib/cache get`），**不取本项目的缓存**；
2. 设 `LEAN_ABORT_ON_PANIC=1` 与 `LAKE_ARTIFACT_CACHE=false`（后者强制全部现编、不读任何 Lake 缓存），编译 `Challenge` 并导出成 `challenge.export`；
3. 同样编译 `Main` 导出成 `solution.export`，然后跑 `lake comparator --config comparator.json --challenge-from-export … --solution-from-export …`。

> ⚠️ **这一步禁止用 `lake exe cache get`。** 它会把本项目的预编译产物也拉下来——那就等于「拿别人编译好的东西证明它编译得对」。上游 README 用加粗写了这一点。

整个流程必须在一台**装了 bubblewrap 的干净 Linux 机器**上跑，因为「编译待检代码」等于「执行待检代码」。上游原话：

> Compiling the submission runs its code, so you may want to do it in a VM or your own sandbox.

---

## 5. 规模解剖

### 5.1 库级分布（2026-11-24 实测）

| 库 | 文件数（原始） | 文件数（当前 main） | 行数（原始） | 行数（当前 main） | 字节数（当前 main） |
|---|---:|---:|---:|---:|---:|
| Development | 141,928 | 41,838 | 12,232,663 | 11,873,516 | 834,788,725 |
| Formalism | 18,254 | 9,918 | 3,200,748 | 2,896,997 | 158,701,514 |
| GroundTheory | 8,507 | 5,225 | 1,119,138 | 1,058,074 | 54,555,292 |
| ReciprocityTheory | 3,845 | 2,423 | 563,016 | 591,468 | 32,421,546 |
| CompanionTheory | 2,027 | 706 | 194,218 | 213,979 | 13,511,222 |
| PositivityTheory | 449 | 227 | 55,130 | 51,842 | 3,084,930 |
| Lemmata | 167 | 153 | 9,232 | 8,220 | 366,943 |
| 其余（Main/Statements/Challenge 等） | 10 | 3 | 70 | 103 | 4,437 |
| **合计** | **175,187** | **60,493** | **17,374,215** | **16,694,199** | **1,097,434,609** |

**三个观察**：

1. **`Development` 一个库占了 71% 的文件和 71% 的行数。** 它是展开最深的那个部分（文件数从 8,507 涨到 41,838，是唯一逆势增长的库，说明这里的模块被拆得更细）。
2. **文件数减半的行数基本没变，说明社区在「合并模块」而不是「删证明」。** `ReciprocityTheory` 和 `CompanionTheory` 的行数甚至**涨了**（+5.1% / +10.2%），同时文件数分别掉了 37% 和 65%。
3. **字节数比行数涨得多（+6.5% vs −3.9%），说明平均行变长了。** 原始源码「一行 59.3 字节」，现在「一行 65.7 字节」——更长的一行通常意味着**更长的标识符**。Lean 的推导式证明里，长变量名（而不是 `a b c`）是可读性的关键，这也解释了为什么字节数反而上升。

### 5.2 声明统计（正则近似，2026-11-24 实测）

| 声明 | 数量（当前 main） |
|---|---:|
| `theorem` | 533,516 |
| `def` | 134,359 |
| `lemma` | 20,189 |
| `abbrev` | 9,673 |
| `structure` | 6,245 |
| `instance` | 5,221 |
| `inductive` | 82 |
| `class` | 35 |

原始归档的官方统计是 **818,755 个 `theorem`/`lemma`** + 242,067 个其他声明。当前 main 的 `theorem` + `lemma` 合计约 **553,705**（−32%），`def` 等其他声明约 **155,615**（−36%）。

> ⚠️ 这里是**正则近似**（`^\s*(theorem|lemma|...)\b`），与 Lean 语义不完全等价——例如 `theorem` 写在行内的情形、或被注释掉的情形都算不准。只用于看量级。

**总量减少 32% 的声明同时，行数只减少 3.9%**，意味着**保留下来的声明平均变长了 40%**——这和上游 README 把「readable」列为三大目标之一（lemma 组织、格式、惯用风格）是吻合的。

### 5.3 静态扫描结论（转述自上游 `verification/static-scan.txt`）

原始源码中**未发现**：`axiom`、`sorry`、`admit`、`native_decide`、`implemented_by`、`extern`、`unsafe`、`opaque`、`debug.*` 选项、`skipKernelTC`。

另外：所有记号都是 `local` 或 `scoped` 的；源码中没有任何读写文件、启动进程、读环境变量的操作。

**「原始源码没有注释」**这一点值得单独说。17,374,215 行、零注释——这意味着这份证明**没有任何作者意图的说明**，读者只能从类型签名和数学结构反推意图。社区接手后正在补的「readable」目标里，注释与 docstring 是重要一项。这也是为什么 [`04`](./04-向上游提PR的规则与禁区.md) 里上游专门有一条 PR 标题示例是「补上某个表示的文档说明」。

---

## 6. 元编程的暗角：`kexactB`、`kcopy`、`run_tac`

这是本章技术含量最高、也最容易被忽略的一节。

### 6.1 三个「跳过类型检查」的手法

上游 NOTES.md 专门列了一节，说明源码用到了 Lean 的元编程 API：

| 手法 | 位置 | 做什么 |
|---|---|---|
| `kexactB` | `Formalism/LocalToGlobal/IndefiniteReal/HeckeNonvanishing/Imports.lean` | 定义一个 tactic：**先 elaboration 出一个项，然后直接把它赋给目标，elaborator 不检查它的类型是否匹配**。通过小宏在 **67 个文件**里使用 |
| `kcopy` / `#kcopy_probe` | `Formalism/LocalToGlobal/NumberFieldFamilies/OwnStruct.lean` | 逐字段重建一个项并直接赋值，同样不检查类型 |
| `run_tac` | 33 处 | 31 处是「revert 掉所有假设后 `exact`」，2 处直接把假设赋给目标 |
| `addDecl` 复制 | `Formalism/HermitianCurvature/.../CutCptCopy.lean` | 定义一个命令，把声明复制到新名字下，**这样内核会检查每一份拷贝** |
| `@[positivity]` 扩展 | `…/TotientGaloisClasses/EvalFrameOrd.lean` | 一个 positivity 扩展 |

### 6.2 为什么这件事很重要

用 `kexactB` / `kcopy` 做出来的证明，**elaborator 没有检查过项与目标的类型是否匹配**。所以：

> 对于用 `kexactB`、`kcopy` 或直接赋值做出来的证明，**唯一可能的检查者就是内核**。

上游自己的结论是：这是可靠的，因为内核会检查每一个声明；**而这正是本仓库要用独立内核重新检查导出证明的原因**。

**这就把 §4.2 的两个外部内核从「锦上添花」升级为「必需品」。** 一个 67 个文件用 `kexactB` 的证明，如果只有 Lean 自带内核在验，那么「证明是否成立」的置信度就等于「Lean 内核是否正确」这一个假设。加入 nanoda 这个独立实现，才把这个假设拆成两个。

**这是本主线最重要的一条可迁移经验**：

> **当证明里存在「绕过 elaborator 的捷径」时，独立内核不是加分项，是必要条件。**
> 因为 elaborator 是第一道防线，它在这条路径上已经被绕过了。

---

## 7. 从「一个归档」到「一个可维护仓库」

4 天 583 个已合并 PR、约 2,000 次提交——在「每个 PR 都要编译数小时」的前提下。三个设计让这件事成为可能。

### 7.1 共享缓存：先有缓存，再谈贡献

上游的核心命题写在 README 第一段：

> **accessible**: we want to improve build performance and modularity to enable researchers to feasibly engage with the proof even without access to massive amounts of compute.

实现方式是**公共构建缓存**，配置写在 `lakefile.toml` 末尾：

| 配置项 | 值 |
|---|---|
| `defaultService` | `riemann-hypothesis` |
| `kind` | `s3` |
| `artifactEndpoint` | `https://rh-cache.taucetiproject.org/artifacts` |
| `revisionEndpoint` | `https://rh-cache.taucetiproject.org/revisions` |

用法是两行：

```bash
lake exe cache get    # 拉取预编译产物
lake build --no-build # 校验：没有任何模块需要重新编译
```

缓存里有 **175,187 个压缩产物（LTAR）**，加上一张「输入哈希 → 产物」的映射表和一份模块清单。完整机制在 [`03`](./03-向官方共享缓存贡献构建产物.md) 展开。

> **顺序很重要**：上游是先建好缓存、再开放 PR 协作。如果反过来（先开 PR 再建缓存），每个贡献者都要先编译几小时才能开始干活，三天就能劝退所有人。

### 7.2 模块粒度合并：把 175,187 压到 60,493

上游 README 说合并 PR 主要做两件事：**删掉证明用不到的文件**、**缩短证明**。但 §5.1 的数据显示，字节数是上升的。综合两个事实，实际发生的是：

- **模块合并**：多个小模块合成一个，文件数下降、行数基本不变、字节数上升（平均行长增加）；
- **声明合并**：818,755 → 553,705，−32%，但总行数只降 3.9%，即**留下来的声明平均长了 40%**。

从 PR 标题也能看出这个方向。例如已合并的 `#1099`「Shorten 336 proofs in GroundTheory, ReciprocityTheory, CompanionTheory and PositivityTheory（−1,019 lines）」，以及 `#1103`「prune 125.7k redundant private Mathlib imports across ReciprocityTheory, CompanionTheory, PositivityTheory & Lemmata（−60.7 CPU-min）」。

**最后这条特别值得注意**：删掉 12.57 万个**私有的** Mathlib import。这不是删证明，是**切断依赖**——一个模块 import 的 Mathlib 越少，它能被复用和重编译的粒度就越细，整个项目的增量构建成本越低。**在一个 60,493 模块的项目里，import 图的边数比顶点数更决定构建性能。**

### 7.3 CI 流水线

9 个 workflow，按职责分：

| Workflow | 职责 |
|---|---|
| `checks.yml` | 基础 CI（README 顶部的 badge 就是它） |
| `build-pr.yml` | PR 构建，支持 `quick-check` / `fragment` / 全量三种模式 |
| `publish-ci-cache.yml` | **发布缓存**（在 GitHub 托管机上跑，不碰构建机） |
| `challenge-files.yml` | 校验 Comparator 的挑战文件没被改 |
| `inbox.yml` / `inbox-review.yml` | PR 分诊机器人（自动贴标签、催作者） |
| `label-pr.yml` | 自动打标签 |
| `batch-queue.yml` | 批量队列调度（24 KB 的 workflow，配 48 KB 的 Python） |

**缓存发布的凭证模型是这套流水线里最值得学的部分**：

> - 构建 job **不持有任何 R2 凭证**，它把新产物作为 GitHub Actions artifact 上传；
> - 一个全新的 `ubuntu-latest` job 在 `ci-cache-publish` 环境里用 `R2_ACCOUNT_TOKEN` 派生密钥，**验证完再上传**；
> - `R2_ACCOUNT_TOKEN` 是 `ci-cache-publish` 的**环境密钥**，**不是仓库密钥**；
> - 「上传凭证永远不能到跑 PR 代码的机器上」。

详见 [`03` §4](./03-向官方共享缓存贡献构建产物.md#4-凭证模型上传凭证永远不在构建机上)。

---

## 8. 治理：README 与 AGENTS.md

### 8.1 README 的「不收什么」

上游 README 有一节专门列出**不欢迎的贡献**，翻译成中文是三条：

1. 沿着**实质不同的证明路径**做的大改；
2. **关于你自己机器上的构建 / Comparator 运行结果的声明**；
3. **加入证明本身用不到的文件**。

第 2 条尤其关键：**上游不接受任何形式的「我在我机器上跑通了」**。这直接决定了 [`03`](./03-向官方共享缓存贡献构建产物.md) 里说的——你不能靠提交一份「本机构建报告」来参与，你只能**真的把压缩产物喂进缓存**。

### 8.2 AGENTS.md：写给 agent 的规范

`AGENTS.md` 开头就是：

> Read [README.md](README.md) first. It says what this repository is for, which changes are welcome and how the proof is checked. The rules below are the ones an agent is most likely to break.

**「规则是 agent 最容易违反的那些」**——这是一个很有意思的设计：它假设读者可能是 LLM，所以把「最容易被违反的规则」单独拎出来。

内容分四块：

1. **构建**——先 `lake exe cache get` 再 `lake build --no-build`；
2. **PR**——改完并确认能编译之前不要开 PR；先开就开 draft；维护者打回后贴 `ready` 恢复排队；
3. **不要**——不改 Comparator 挑战文件、不加公理或 `sorry`、不重写证明路径、不加证明用不到的文件；
4. **常见错误**——7 条。

第 4 块是整份 `AGENTS.md` 最有价值的部分——**6 条**。展开在 [`04`](./04-向上游提PR的规则与禁区.md)。

---

## 9. 可核对的事实清单

> 全部为 2026-11-24 实测或上游文档直引。想复核的话，命令在括号里。

| 事实 | 值 | 复核方式 |
|---|---|---|
| 固定 commit | `26c37b400bc9cfd43c81d798aa5e798af5c55b94` | `gh api repos/leanprover-community/riemann-hypothesis/commits/main --jq .sha` |
| 固定 commit 的时间 | 2026-11-24T12:39:27Z | 同上，`--jq .commit.committer.date` |
| 证明源码文件数 | 60,493 | `find src -name '*.lean' \| wc -l` |
| 证明源码字节数 | 1,097,434,609 | `find src -name '*.lean' -printf '%s\n' \| awk '{s+=$1}END{print s}'` |
| 证明源码行数 | 16,694,199 | `find src -name '*.lean' -print0 \| xargs -0 cat \| wc -l` |
| Lean 工具链 | `leanprover/lean4:v4.37.0-rc1` | `cat lean-toolchain` |
| Mathlib 版本 | `31ce06eaadf4a200c0e1407c5f10953cf606b8f8` | `lakefile.toml` 的 `[[require]]` |
| 允许的公理 | `propext` / `Quot.sound` / `Classical.choice` | `comparator.json` |
| 外部内核 | `leanchecker-paranoid` / `nanoda_bin` | `comparator.json` |
| 缓存地址 | `https://rh-cache.taucetiproject.org` | `lakefile.toml` 的 `[[cache.service]]` |
| R2 bucket | `riemann-hypothesis-cache` | `scripts/publish-lake-cache.py` |
| 参考文献条数 | 21 | `grep -c '^@' docs/references.bib` |
| 上游 release / tag | **0 个 / 0 个** | `gh release list --repo leanprover-community/riemann-hypothesis` |

---

## 10. 怎么读这个项目

| 你的目的 | 建议入口 |
|---|---|
| 想搞懂 RH 到底证了什么 | 本章 §3 + 上游 `src/Statements.lean`（只有 15 行） |
| 想自己复现一次检查 | 本章 §4.4 的三步 + [`03` §2](./03-向官方共享缓存贡献构建产物.md#2-先决条件) 的环境要求 |
| 想给上游提 PR | [`04`](./04-向上游提PR的规则与禁区.md) 全章，**先读 `AGENTS.md` 原文** |
| 想参与贡献构建产物 | [`03`](./03-向官方共享缓存贡献构建产物.md) 全章 |
| 想拿一份源码做本地阅读 / 检索 / 喂给 LLM | [`02`](./02-压缩分发与供应链校验.md) |
| 想读数学部分 | `docs/references.bib`（21 条）+ `Formalism/GammaFactorNonvanishing/CritLine.lean`（从 ζ 的临界线论证看起） |

---

## 🔗 相关

- 本主线 → [`README.md`](./README.md)
- 下一章 → [`02 · 压缩分发与供应链校验`](./02-压缩分发与供应链校验.md)
- 术语表 → [`../appendix/术语表.md`](../appendix/术语表.md)
