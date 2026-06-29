# CV-CUDA Resize 算子 SYCL 移植 — 总结与思考

> 本文件说明：做了什么、文件夹里是什么、怎么编译运行、性能如何、以及过程中的一些思考。
> 更详尽的过程上下文见同目录 `claude_task_resize`（273 行，逐 phase 记录）。

## 1. 这是什么

把 NVIDIA CV-CUDA 0.16.0 的 Resize 算子（CUDA）移植到 Intel SYCL，在 Intel Arc Graphics [0xb080] GPU 上跑通、验证、优化。这是 cvcuda 算子移植系列的**第 1 个算子**（前一个 BEVFusion 体素化算子已完成）。

- **平台**：Intel Arc Graphics [0xb080] (B390, ~29GB)，oneAPI 2025.3.3，OpenCL 后端。
- **源算子**：CV-CUDA 0.16.0 `priv/OpResize.cu`（4 种插值）+ `priv/legacy/resize_var_shape.cu`（变尺寸）。
- **工作目录**：本机 `cvcuda-ops-sycl/OpResize/`；Intel 机 `/work/bevfusion_migration/cvcuda_ops/OpResize/`。

## 2. 做了什么

1. **移植**：CUDA → SYCL，忠实翻译 4 种插值 kernel（NEAREST / LINEAR / CUBIC / AREA）。剥掉 nvcv/cvcuda 框架（IOperator / TensorDataStridedCuda / cvcudaResizeSubmit 全不要），只留 kernel 算法 + 自研最小 tensor wrapper。接口自设计 `run_resize<T,C>` / `run_resize_varshape<T,C>`，非 drop-in 替换。
2. **多数据类型 + 多通道**：模板化 `<T,C>`，实现 **F32 / U8 × C1/C3/C4**（规格上 CV-CUDA 还支持 U16/S16，本次未做，加 SaturateCast 特化即可）。
3. **变尺寸（var-shape）**：每图自带 W/H 的 pitched batch，NN/LINEAR/CUBIC 三种插值。
4. **正确性验证**：合成数据逐用例对比「同算法 CPU 参考」，36 用例全 PASS（25 固定尺寸 + 11 var-shape）。
5. **性能 profiling**：SYCL event profiling，best-of-20，大图（1920×1080）基线。
6. **LINEAR 读复用优化**：移植 CUDA 的 `LinearInterpolatePack<INTERSECT=true>`（NIX + INTERSECT），U8 C3 上采样 4K LINEAR **3.27→1.79ms（1.83×）**，且 F32 无回退。
7. **7 个 phase 快照**齐全，每步可回退复现。

## 3. 目录结构

```
OpResize/
├── README.md                       # 本文件（总结与思考）
├── claude_task_resize              # 详尽过程上下文文档（逐 phase）
├── build.sh                        # 编译脚本（编 3 个二进制）
├── resize.hpp                      # header-only 模板 kernel：4 插值 + run_resize 分发
│                                   #   （LINEAR 含 NIX+INTERSECT 读复用，kNIX 门控）
├── resize_varshape.hpp             # header-only var-shape kernel：NN/LIN/CUB + 分发
├── OpResize.cpp                    # 原始 NVIDIA op 包装源（参考，不参与编译）
├── common/
│   ├── tensorwrap.hpp              # TensorView<T,C> 最小 NHWC wrapper（模板）
│   ├── varshape.hpp                # VarShapeBatch<T,C> pitched 变尺寸 batch
│   ├── saturate_cast.hpp           # SaturateCast<float>(恒等) + <uint8_t>(半偶舍入+clamp)
│   └── math_ops.hpp                # min/max/clamp 小工具
└── test/
    ├── test_resize.cpp / test_resize                 # 固定尺寸正确性（25 用例）
    ├── test_resize_profile.cpp / test_resize_profile # 性能 profiling（best-of-20）
    └── test_resize_varshape.cpp / test_resize_varshape # var-shape 正确性（11 用例）
```

**组织原则**：kernel 全部 header-only 模板（`resize.hpp` / `resize_varshape.hpp`），test 直接 include，build.sh 只编 3 个 test 二进制；共享工具放 `common/`；原始 NVIDIA 源留作参考不参与编译。

## 4. 怎么编译

宿主机有 `icpx`（容器里没有，**必须宿主机编译**）：

```bash
cd /work/bevfusion_migration/cvcuda_ops/OpResize
bash build.sh
```

编出 3 个二进制（`test_resize` / `test_resize_profile` / `test_resize_varshape`），放到 `test/` 下。编译选项 `-fsycl -std=c++17 -O2 -ffp-model=precise -fno-fast-math -fimf-precision=high`（精确 FP 标志同体素化移植，减少加权求和的非结合漂移）。

## 5. 怎么运行

> **必须**带 `ONEAPI_DEVICE_SELECTOR=opencl:gpu` —— Level Zero 后端在这台机器有 bug 会段错误。
> 建议每个 SSH session 先 export 一次：
> ```bash
> export ONEAPI_DEVICE_SELECTOR=opencl:gpu
> ```

下面命令假设在 `.../OpResize` 目录。

### 5.1 固定尺寸正确性（25 用例）
```bash
./test/test_resize
```
预期最后一行：`=== ALL TESTS PASSED ===`。覆盖 F32/U8 × C1/C3/C4 × NEAR/LIN/CUB/AREA，上/下采样、非整数比例、batch N=2。

### 5.2 var-shape 正确性（11 用例）
```bash
./test/test_resize_varshape
```
覆盖 F32/U8 × C1/C3 × NN/LIN/CUB，N=2~3 张不同尺寸图。

### 5.3 性能 profiling（best-of-20）
```bash
./test/test_resize_profile
```
输出大图（1920×1080 F32 C1 + U8 C3 + U8 C1）上/下采样各插值的 best device ms + dst Gpix/s + GB/s。

## 6. 四种插值算法要点（移植依据）

| 插值 | 采样 | 坐标映射 | 边缘处理 |
|---|---|---|---|
| NEAREST | 1 像素 | `(dst+0.5)*scale` | floor + clamp [0,size-1] |
| LINEAR | 2×2 | `(dst+0.5)*scale - 0.5` | floor + 边缘 wx=0/1 + clamp [0,size-2] |
| CUBIC | 4×4 | `(dst+0.5)*scale - 0.5` | fx 在 x 边缘置零(**非对称**, fy 不置零), isx/isy clamp [1,size-3], abs + SaturateCast |
| AREA | 盒均值 | 足迹 `[x·sx,(x+1)·sx)×[y·sy,(y+1)·sy)`（**无 +0.5**） | xmin=ceil / xmax=floor, 整数 scale 简单均值, 非整数 9 项分数边缘 |

- `scale = srcSize / dstSize`（per-axis）。CUBIC 用 Keys 核 A=-0.75。
- **AREA 所有 scale 在 host 算好传入，kernel 内无除法** → 不受 Intel GPU FP 除法非正确舍入问题影响（maxerr 极小）。

## 7. 性能报告

> 测量方法：SYCL queue `enable_profiling`，逐次取 device 时间，warmup + best-of-20。Arc GPU 跨进程有 ±10%+ 噪声，重要 A/B 在同一 session 交替跑取 best。

### 7.1 LINEAR 优化前后对比（优化目标）

| 场景 | phase5 基线 | phase5b 优化后 | 变化 |
|---|---|---|---|
| F32 C1 上采样 4K LINEAR | 0.350ms / 23.7 Gpix/s | 0.358ms / 23.2 | 持平（门控保护）|
| **U8 C3 上采样 4K LINEAR** | 3.267ms / 2.5 Gpix/s | **1.790ms / 4.6** | **1.83×** ⬆️ |
| U8 C3 下采样 LINEAR | 0.151ms | 0.103ms | 1.47× ⬆️（反超 NEAREST）|
| U8 C1 上采样 4K LINEAR | — | 0.346ms / 24.0 Gpix/s | NIX=16 表现好 |

### 7.2 完整基线表（phase5b 后，best ms）

```
[F32 C1 downscale 1920x1080->960x540]   NEAR 0.114 / LIN 0.125 / CUB 0.122 / AREA 0.123 ms
[F32 C1 upscale  ->3840x2160]           NEAR 0.323 / LIN 0.358 / CUB 0.502 ms
[U8 C3 downscale ->960x540]             NEAR 0.115 / LIN 0.103 / AREA 0.143 ms
[U8 C3 upscale   ->3840x2160]           NEAR 1.279 / LIN 1.790 ms
[U8 C1 upscale   ->3840x2160]           NEAR 0.293 / LIN 0.346 ms
```

### 7.3 NIX+INTERSECT 优化说明

移植 CUDA `LinearInterpolatePack<INTERSECT=true>`：每线程处理 NIX 个连续 dst 像素，相邻 dst 映射同一 src 时复用 2×2 邻域，src x+1 时只增量读 2 个像素。

| 项 | 改动 | 收益 |
|---|---|---|
| NIX 多像素/线程 | 每线程处理 NIX 个连续 dst，减 launch 量 | U8 C3 上采样主力收益 |
| INTERSECT 读复用 | 相邻 dst 同 src 时复用邻域，增量读 | 减重复访存 |
| **kNIX 门控** | `kNIX = (sizeof(T)==1) ? 公式 : 1` | **F32 走 NIX=1 简单路径，避免 8% 回退** |

### 7.4 性能极限分析

1. **小图 launch-bound**：960×540 输出 ~0.1ms，4 种插值差异极小，固定 launch 开销主导。要更快需 batch 融合 / 算子链。
2. **U8 C3 多字节像素访存未对齐**曾是主要瓶颈（phase5b 已对 LINEAR 处理，1.83×）。NEAREST/CUBIC/AREA 未做 NIX 优化（非瓶颈；CUBIC 用得少，NEAREST 已快，AREA host 算 scale 已无除法）。
3. **F32 C1 大图已较快**（~24-25 Gpix/s），优化空间小；NIX 反而因减线程损并行度，故门控掉。
4. **结论**：LINEAR（最常用）的 U8 多通道场景已优化到位；进一步提速需算子链融合或向量化 WritePack，属另一阶段工作。

### 7.5 正确性

- 固定尺寸 25 用例全 PASS。maxerr：NEAREST=0，LINEAR/CUBIC F32~1e-5、U8=0，AREA F32~1e-6、U8=0。
- var-shape 11 用例全 PASS（F32/U8 × C1/C3 × NN/LIN/CUB，N=2~3 不同尺寸图）。
- LINEAR 优化后 25 用例 bit-一致（U8 LINEAR 在 ±1 容差内，实测这批 maxerr=0）。
- AREA 不受 Intel GPU FP 除法问题影响（scale 全 host 算好，kernel 无除法）。

## 8. 注意事项与思考

- **必须用 OpenCL 后端**：`ONEAPI_DEVICE_SELECTOR=opencl:gpu`。默认 Level Zero 后端在这台机器跑 kernel 会段错误（同体素化任务）。
- **测量噪声**：Arc GPU 跨进程 ±10%+ 时钟/热噪声。重要 A/B 要在同一 SSH session 交替跑多轮取 best，不要先跑完 A 退出再跑 B。
- **U8 LINEAR 容差 ±1**：Intel GPU 浮点边界舍入可致 U8 在 x.5 处差 1（同体素化 ±1 邻居分析的设备固有特性，非 port bug），测试用 ±1 容差作安全网。实测这批用例 maxerr=0。
- **C++ 模板特化顺序**：`SaturateCast` 的 `<float>`/`<uint8_t>` 全特化**必须写在主模板声明之后**，否则报 "no matching function"（phase2 编译错一次，调换顺序修复）。
- **CUBIC 非对称边缘**：原版只对 `fx`（x 轴）在边缘置零，`fy`（y 轴）不置零只 clamp。必须照抄，否则输出不一致。
- **负数左移是 UB**：phase5b 初版写过 `-1 << 30`，改 `int iPrevX=0` + `bool have` 守卫。

### ⚠️ 验证局限（诚实声明，重要）

1. **自洽验证，非对齐验证**：正确性测试用的是我自己写的「同算法 CPU 参考」，证明「SYCL kernel == 我的 CPU 实现」，**不直接证明 == NVIDIA CV-CUDA 实际输出**。若我对原算法某处理解错，CPU 和 GPU 会一起错，测试仍 PASS。缓解：关键逻辑（坐标公式、`get_cubic_coeffs` A=-0.75、AREA 的 ceil/floor 边缘、CUBIC 非对称 fx 置零、SaturateCast 半偶舍入）**逐字照抄原版**，非重写，理解错风险集中在边缘细节。
2. **未与 NVIDIA CV-CUDA 做 bit 级 diff**：真正的金标准是同图同参数喂 NVIDIA `cvcudaResize` 和本 SYCL kernel 比输出。这一步没做（需 NVIDIA GPU + 编译 CV-CUDA，Intel Arc 跑不了 CUDA 版；P2200 跳板机 / thor / orion 等 NVIDIA 机器理论可行但未做）。
3. **合成数据 ≠ 真实图像极端情况**：没用真实图片做端到端测试，可能存在未测到的边界（极大图整数溢出、特殊宽高比等）。
4. **AREA var-shape 未移植**：原版用另一套 OpenCV-aligned 实现 `resize_area_ocv_align` + BorderVarShapeWrap，复杂少用，标注为限制。
5. **U16/S16 dtype 未做**：加 SaturateCast 特化即可，规格上 CV-CUDA 支持。
6. **未集成进 nvcv/cvcuda 框架**：独立 SYCL 算子，非 drop-in 替换。

## 9. 一点思考

- **「忠实翻译」优于「重写」**：AREA 的 ceil/floor 边缘、CUBIC 的非对称 fx 置零这类细节，如果按「直觉」重写很容易写错且测试测不出来（CPU 参照会跟着错）。逐字照抄原版把理解错的风险压到边缘细节，是这次移植和体素化任务一致的策略。
- **优化要带门控，别一刀切**：NIX+INTERSECT 对 U8 多通道是收益，对 F32 反而损并行度（-8%）。`kNIX = sizeof(T)==1 ? 公式 : 1` 这个门控保住了 F32 不回退。移植 CUDA 优化时不能盲目照搬，要按本设备特性（Arc 对齐访问成本、并行度敏感度）重新判断适用边界。
- **「自洽 ≠ 对齐」要诚实标注**：没有 NVIDIA GPU 做金标准 diff，测试只能证明自洽。这个局限必须写进文档，不能让「36 用例全 PASS」读起来像「与 NVIDIA 一致」。下一步真正要补的就是对齐验证。
- **复用体素化任务的环境与纪律**：icpx / opencl 后端 / in-order queue / USM device 内存 / 每步存快照 / 同 session 交替 A/B 取 best —— 这些在体素化任务里建立的纪律直接复用，省了大量环境踩坑时间。
