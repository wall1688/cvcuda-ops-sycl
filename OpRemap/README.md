# CV-CUDA Remap 算子 SYCL 移植

> CV-CUDA 0.16.0 Remap (`priv/OpRemap.cu`) CUDA→Intel SYCL 移植。系列第 2 个算子(第 1 个 Resize 已完成)。目标平台 Intel Arc Graphics [0xb080],oneAPI 2025.3.3,OpenCL 后端。
> 进度记忆见本机 Claude 记忆 `cvcuda-remap-port`。

## 0. 当前状态(2026-06-30)

**Phase 1-7 全部完成,共 45 用例全 PASS(38 固定形状 + 7 var-shape),maxerr 全 0。** Remap 移植完成。

| Phase | 范围 | 状态 |
|---|---|---|
| 1 | F32 C1 / ABSOLUTE / NEAREST×NEAREST / REPLICATE | ✅ |
| 2 | + U8 C1/C3/C4(规格:F32 仅 C1,U8 支持 C1/C3/C4) | ✅ |
| 3 | + src LINEAR/CUBIC + 全 5 border | ✅ |
| 4 | + map LINEAR/CUBIC(map float2,恒 REPLICATE;mapW≠dstW 触发分数 mapCoord) | ✅ |
| 5 | + 3 mapValueType×alignCorners 完整矩阵(Absolute 不受 ac 影响) | ✅ |
| 6 | + profiling + 内部邻域快速路径优化(bit-identical,38 用例无回归) | ✅ |
| 7 | + var-shape(每图独立 src/dst/map 尺寸;sampler 重构 shape-agnostic) | ✅ |

## 1. Remap 在做什么

对每个输出像素 `(dx,dy,z)`:
```
1. mapCoord = ((dx+dstOffset)*mapScale, (dy+dstOffset)*mapScale, mz)
2. mapValue = sample_map(mapCoord, mapInterp)        // map border 恒 REPLICATE, 返回 float2
3. srcCoord = (dx*srcScale + mapValue.x*valScale + srcOffset,
              dy*srcScale + mapValue.y*valScale + srcOffset)
4. dst      = sample_src(srcCoord, srcInterp, 用户 border)
```
所有 scale/offset 由 `getRemapParams`(按 mapValueType×alignCorners)在 **host** 算好传入;**kernel 内只乘加无除法** → 避开 Intel GPU FP 除法非 IEEE 问题(同 Resize AREA)。

## 2. 目录结构

```
OpRemap/
├── remap.hpp              # getRemapParams(3 mvt×alignCorners 逐字照抄)+ DoRemap kernel + run_remap<T,C> 分发
├── remap_varshape.hpp     # Phase 7 var-shape:每图 host 算 RemapParams(无 kernel 除法),kernel 走 max 维度+边界检查,复用 sample_impl
├── common/
│   ├── tensorwrap.hpp / saturate_cast.hpp / math_ops.hpp   # 复制自 OpResize
│   ├── border.hpp         # getIndexWithBorder 5 border 逐字照抄 BorderWrap.hpp
│   ├── sampler.hpp        # sample_impl/fetch_impl(shape-agnostic,W/H 显式)+ 薄 sample<T,C> wrapper;CUBIC A=-0.5;内部快速路径
│   └── varshape.hpp       # VarShapeBatch<T,C> pitched(batch 内每图独立 imgW/imgH,复制自 OpResize)
├── test/test_remap.cpp         # 固定形状 38 用例:模板化 run_case + CPU 参照(复用同一 sample)
├── test/test_remap_varshape.cpp# var-shape 7 用例:每图独立尺寸 + map 广播
├── test/perf_remap.cpp         # Phase 6 perf harness(大图,逐核 profiling,best-of-N)
├── build.sh               # icpx 编 3 个二进制
└── OpRemap.cpp            # 原始 NVIDIA 包装源(参考,不编译)
```

## 3. 关键设计

- **interp/border 走 runtime switch 非模板**:Remap 组合空间 1620 种,只对 T×C(6 种)做模板。warp uniform 无发散。
- **CPU 参照复用同一 `sample()`**:test 里的 `cpu_remap` 调用与 kernel 完全相同的 `sample<T,C>()`,保证 host/device 数学一致。
- **CUBIC A=-0.5**(OpenCV `GetCubicCoeffs`),非 Resize 的 A=-0.75。
- **NEAREST = floor(c+0.5)**;map border 恒 REPLICATE;mapNumSamples==1 广播。
- **Phase 6 内部快速路径**:`sample` 邻域全在范围内时直读 `v.read()` 跳过 border(bit-identical,因 `getIndexWithBorder` 对范围内索引恒等)。REFL≈REPL,CUBIC REFL 4.87×。
- **Phase 7 shape-agnostic sampler**:`sample_impl/fetch_impl` 把 W/H 作显式参数、模板化 View,`TensorView` 与 `VarShapeBatch` 共用同一份插值+border 代码;薄 `sample<T,C>` wrapper 保护固定形状调用点(38 用例无回归)。var-shape 每图 `RemapParams` 在 host 算(device imgW/imgH 拷回 host),kernel 内无除法。

## 4. 怎么编译运行

> 必须在 Intel 物理机上(本机 VM 无 GPU)。连接:本机 → ssh 跳板 → `connect_intel.sh` → Intel 机(凭据不放在公开仓库,见本机记忆 `bevfusion-intel-env`)。

```bash
cd /work/bevfusion_migration/cvcuda_ops/OpRemap
bash build.sh                                   # 宿主机 icpx 编译
export ONEAPI_DEVICE_SELECTOR=opencl:gpu        # 必须!Level Zero 在该机会段错误
./test/test_remap                               # 期望:=== ALL TESTS PASSED ===
```

编译选项:`-fsycl -std=c++17 -O2 -ffp-model=precise -fno-fast-math -fimf-precision=high`。

## 5. 测试覆盖(45 用例:38 固定形状 + 7 var-shape)

- Phase1 F32 C1:identity / shift_x / round_halfup / oor_replicate / batch2
- Phase2 U8 C1/C3/C4:identity / shift_x / oor / batch2 / C4
- Phase3 src LINEAR/CUBIC × 全 5 border:分数 map `(x+0.5,y+0.5)` 触发插值+边缘越界触发 border;含 U8 插值 SaturateCast、U8 C3 多通道插值、CONSTANT border 广播(F32=7/U8=200)
- Phase4 map LINEAR/CUBIC:**mapW≠dstW**(4×4 map,6×6 dst)→ mapScale=4/6 → 分数 mapCoord → map 本身走 LINEAR/CUBIC,混合 `fill_frac_abs` 的逐像素不同坐标。srcInterp 先 NEAREST(隔离 map 插值),再组合 src LINEAR;含 U8 C1/C3、CONSTANT border
- Phase5 3 mvt×alignCorners 矩阵:Absolute(不受 ac 影响,acF/acT 对照)/ AbsNorm×{F,T} / RelNorm×{F,T}。NEAREST×NEAREST 隔离 `getRemapParams` 变换;RelNorm 的 mapScale=(mapW-1)/dstW 走不同公式;`fill_absnorm` mapValue∈[-1,1]、`fill_relnorm` 小相对偏移使边缘越界;含 AbsNorm+src LINEAR、U8 归一化路径

容差:NEAREST=0;LINEAR/CUBIC 设 1e-4(F32)/±1(U8)作安全网。**实测 36 条 maxerr=0**;仅 2 条含 LINEAR 加权求和的触发安全网(`mapcub_linsrc` 5.7e-6、`absn_acF_lin` 7.6e-6,host/device FP 非结合性,仍在容差内)。

- Phase7 var-shape(7 用例):每图独立 src/dst/map 尺寸(N=3,如 src 5×4/7×6/4×5、dst 6×5/5×4/7×6、map 4×4/3×5/5×3),pitched `VarShapeBatch`。覆盖 NEAREST/LINEAR/CUBIC、REPLICATE/REFLECT、F32/U8 C4、ABSOLUTE/AbsNorm、map 广播(map.N==1)。CPU 参照复用同一 `sample_impl` 走 host 指针 VarShapeBatch。**7 用例 maxerr 全 0**。
  - 易错点:map 偏移**不可用 0.5**——会使 mapValue 正好落在 NEAREST 边界 k+0.5,host/device FMA 差 1 ULP 即翻转 floor → 1 像素差。用 0.25/0.75 远离边界即可(测逻辑而非 FP 边界)。

## 6. 验证局限(诚实)

自洽验证(SYCL kernel == 我的 CPU 参照,因复用同一 `sample()`),**非** NVIDIA bit-diff。容差安全网未触发。未与 NVIDIA CV-CUDA 做 bit 级 diff(需 NVIDIA GPU,Intel Arc 跑不了 CUDA 版;同 Resize 局限)。关键逻辑(getRemapParams、GetCubicCoeffs A=-0.5、getIndexWithBorder 5 border、坐标公式)逐字照抄原版以降低理解错风险。

## 7. 快照

每 phase 存 `../snapshots/remap_phaseN_2026-06-30/`,可回退复现(Phase 1-7 已存;Phase 6 含 `PERF_Baseline_vs_Optimized.txt`)。

## 8. 性能(Phase 6)

`test/perf_remap.cpp`,1920×1080×4 batch,best-of-20(逐核 profiling),Intel Arc [0xb080]。

| config | base ms | opt ms | 加速 |
|---|---|---|---|
| f32c1 NEAR REPL | 0.570 | 0.526 | 1.08× |
| f32c1 LIN  REPL | 1.026 | 0.607 | 1.69× |
| f32c1 CUB  REPL | 2.829 | 1.000 | 2.83× |
| f32c1 LIN  REFL | 1.647 | 0.612 | 2.69× |
| f32c1 CUB  REFL | 5.299 | 1.087 | **4.87×** |
| u8c4  NEAR REPL | 0.646 | 0.584 | 1.11× |
| u8c4  LIN  REPL | 1.324 | 0.786 | 1.68× |
| u8c4  CUB  REPL | 3.197 | 1.787 | 1.79× |

**优化**:`sample()` 加内部邻域快速路径——邻域全在范围内时直读 `v.read()`,跳过 `getIndexWithBorder`(REFLECT/WRAP/REFLECT101 的整数取模是热点,CUBIC 每像素 16 次)。**bit-identical**:`getIndexWithBorder` 对范围内索引在所有 border 下都是恒等,故内部直读与 border 路径结果完全相同;host/device 同一 `sample()` 走同一快速路径决策 → 38 用例 maxerr 不变。优化后 **REFL≈REPL**(border 类型不再拖垮性能,只有边缘像素走 border)。

> 测量纪律:Arc 跨 run ±10%+ 噪声,best-of-N;优化版复跑差 <0.3%。GB/s 为"含插值冗余读的有效带宽"(相对指标,冗余读多为 cache 命中故可超硬件上限),主要用于 A/B 相对比较。
