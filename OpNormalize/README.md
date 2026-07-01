# CV-CUDA Normalize — SYCL 移植

把 CV-CUDA 0.16.0 的 `Normalize` 算子(`src/cvcuda/priv/legacy/normalize.cu` tensor path)移植到 Intel SYCL,在 Intel Arc Graphics [0xb080] 上跑通、验证、profiling。这是 cvcuda-ops-sycl 算子移植系列的一个算子(Resize、NMS、CvtColor 已完成,见 `../OpResize/`、`../OpNMS/`、`../OpColorCvt/`)。

## 这是什么

- **算法**:`out = SaturateCast<outT>((src - base) * mul * global_scale + shift)`
  - `mul = scale`(plain 模式)
  - `mul = 1 / sqrt(scale² + epsilon)`(`SCALE_IS_STDDEV` 模式,flag bit 0)
- base/scale 始终 `float*`,按轴广播(N/H/W/C 各自独立为 1 或匹配)。每像素一个线程,通道在内部循环。
- **dtype**:U8/S8/U16/S16/S32/F32(F64/U32/F16 CV-CUDA 不支持 → 拒绝)。**通道** C∈{1,3,4}(C==2 / C>4 拒绝,匹配 CV-CUDA dispatch 表的 null slot)。布局 NHWC,`in_dtype == out_dtype`(单 dtype 入参)。
- **与 CV-CUDA 的差异**:CUDA 用 vector-type 派发(uchar4/float3)经 `funcs[6][4]` + `CreateTensorWrapNHW<T>`;本移植用单 `EltT` 模板 + 运行时 C,数学等价,无 vector 机制。`sat_cast<T>` = `sycl::rint`(round-to-nearest-even,匹配 PTX `cvt.rni`)+ clamp;F32 恒等。plain+stddev 共用一个 kernel(`bool stddev` 分支)。接口返回 **bool**(校验 C/dtype/broadcast),匹配 CV-CUDA INVALID_PARAMETER 语义。

## 目录结构

```
OpNormalize/
├── README.md                       # 本文件
├── 总结.md                         # 移植总结(代码理解 / CUDA→SYCL 映射 / 正确性 / 性能)
├── build.sh                        # 编译脚本(编 3 个 test 二进制;kernel header-only)
├── normalize.hpp                   # header-only:normalize_kernel + dispatch + dtype 派发
│                                   #   + Normalize 接口 + NormalizeImplement + create_normalize
├── normalize_helpers.hpp           # TypeMin/TypeMax/nhwc_ptr(CPU-safe,gold 也 include)
└── test/
    ├── normalize_gold.hpp          # CPU gold 参考(移植自 CV-CUDA TestOpNormalize)
    ├── test_normalize.cpp          # 合成精确值,逐元素对 CPU gold(期望逐 bit 一致)+ 负向
    ├── test_normalize_real.cpp     # 随机数据,正确性 + stddev FP 边界分析 + wall 性能
    └── test_normalize_profile.cpp  # VOX_PROFILE=1,逐 kernel event 计时 + wall + 吞吐
```

**组织原则**(同 OpResize / OpNMS / OpColorCvt):kernel 全部 header-only(`normalize.hpp`),test 直接 include,`build.sh` 只编 3 个 test 二进制;CPU 参考逻辑移植自 CV-CUDA 自带的 `TestOpNormalize`。原始 NVIDIA 源在 CV-CUDA 0.16.0 仓库,此处不重复收录。

## 怎么编译

宿主机有 `icpx`(容器里没有,**必须宿主机编译**):

```bash
cd /work/bevfusion_migration/cvcuda_ops/OpNormalize   # 板上镜像路径(本仓 = OpNormalize/)
bash build.sh
```

编出 3 个二进制(`test_normalize` / `test_normalize_real` / `test_normalize_profile`),放到 `test/` 下。编译选项 `-fsycl -std=c++17 -O2 -g -I. -ffp-model=precise -fno-fast-math -fimf-precision=high`(精确 FP 标志同其它算子移植;对结果无影响,stddev 的 1/sqrt 除法为 GPU 倒数近似,与该标志无关)。

## 怎么运行

> **必须**带 `ONEAPI_DEVICE_SELECTOR=opencl:gpu` —— Level Zero 后端在这台机器有 bug 会段错误。

```bash
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
./test/test_normalize                         # 期望:=== ALL TESTS PASSED ===
./test/test_normalize_real 64 128 128 4 10    # [N] [H] [W] [C] [runs]
VOX_PROFILE=1 ./test/test_normalize_profile 64 128 128 4 20
```

## 算法要点(移植依据)

- 每像素一个线程,`idx → n=idx/(H*W), rem=idx%(H*W), y=rem/W, x=rem%W`。
- 按轴广播索引:`bn = bS.N==1 ? 0 : n`(H/W/C 同理);`base_off = ((bn*bS.H+by)*bS.W+bx)*bS.C`,通道内 `bS.C==1 ? 0 : c`。
- `mul = stddev ? 1/sycl::sqrt(sc*sc+eps) : sc`;`res = (sv-bv)*mul*gscale+shift`;`dp[c] = sat_cast<EltT>(res)`。
- 单 kernel、无原子、无共享内存。plain 与 stddev 共用同一 kernel(`bool stddev` 分支)。

## 正确性

- **合成测试**(`test_normalize`,42 case = 6 dtype × 7 场景):手选精确值(2 的幂 scale、整数 base/shift、stddev eps∈{1,4,16} 使 1/sqrt 为精确 2 的幂)→ 与 CPU gold 逐 bit 一致(mismatch=0)。覆盖 plain+stddev、C∈{1,3,4}、scalar / per-channel / N/H/W 广播。3 个负向测试(C==2、C==5、bad broadcast)拒绝。
- **随机大规模测试**(`test_normalize_real`,N=64 H=128 W=128 C=4):
  - U8/F32 **plain** mismatch=0(无除法,IEEE mul/add 精确,无 FMA 收缩)。
  - **F32 stddev** mismatch=0(tol 1e-4),**maxdiff=1.53e-05** —— ULP 噪声,确认 1/sqrt 除法发散(Intel GPU 倒数近似)。
  - **U8 stddev** mismatch=0(本随机样本的 ULP 偏移未越过 .5 整数边界);整型 divergent 用 `is_boundary`(BOUND_TOL=1e-3)归类。

## 性能

`test_normalize_profile`(N=64 H=128 W=128 C=4,20 runs,GPU warm,best):

```
----- Normalize per-kernel (best) -----
normalize_kernel  best=0.573ms avg=0.722ms (100%)
wall              best=0.704ms avg=...
overhead          =0.132ms (launch + wait)
throughput        kernel=1831 Mpix/s  wall=1489 Mpix/s  (7325 Melem/s kernel)
```

- 单 kernel、embarrassingly parallel、无 host sync、无原子。
- overhead 0.132ms(launch + wait)。后续优化方向(本轮未做):`sycl::vec<float,4>` 通道向量化、tile + 只读 cache 复用广播 base/scale、RPT。

## 验证局限(诚实声明,重要)

1. **自洽验证,非对齐验证**:正确性测试用的是我自己移植的 CPU gold(源自 CV-CUDA `TestOpNormalize`,与 `.cu` kernel 是两份独立实现),证明「SYCL kernel == 我的 CPU 参考」,**不直接证明 == NVIDIA CV-CUDA 实际输出**。若 kernel 与 gold 都从同一份源理解错,二者会一致地错,测试仍 PASS。缓解:kernel 与原 `.cu` 逐行对照;gold 取自源仓独立参考。
2. **未与 NVIDIA CV-CUDA 做 bit 级 diff**:真正的金标准是同参数喂 NVIDIA `cvcudaNormalize` 和本 SYCL kernel 比输出。这一步没做(需 NVIDIA GPU + 编译 CV-CUDA,Intel Arc 跑不了 CUDA 版)。
3. **FP 归因是推断**:stddev maxdiff=1.5e-5 是直接观测;归因到 Intel GPU 倒数近似除法是 **推断**(precedent + ULP 量级),非单独证明的根因。gold 用 `std::sqrt` + IEEE 除法,kernel 用 `sycl::sqrt` + GPU 除法。U8 stddev 显示 0 mismatch 是样本现象 —— 极端 scale/eps 可能产生边界 mismatch(real 测试的 `is_boundary` 会归类)。

## 一点思考

- **plain 模式可做到 bit-exact**:无除法,IEEE mul/add 在 `-ffp-model=precise -fno-fast-math` 下无 FMA 收缩 → 与 gold 完全一致。只有 stddev 的 1/sqrt 引入 ULP 级发散,且发散量级(1.5e-5)远小于 CV-CUDA 原容差。
- **header-only 重构是机械改动**:本仓版本把原 `normalize.cpp` 实现并入 `normalize.hpp`(header-only),并把 `nv::linear_launch` → 原生 `q.parallel_for`,以匹配仓库自包含约定(kernel 数学逐字不变)。launch 机制改动属低风险机械重构,建议在板上重编一次确认(同 OpNMS / OpColorCvt)。
