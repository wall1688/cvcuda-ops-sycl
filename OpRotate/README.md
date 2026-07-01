# CV-CUDA Rotate — SYCL 移植

把 CV-CUDA 0.16.0 的 `Rotate` 算子(`src/cvcuda/priv/legacy/rotate.cu` tensor path)移植到 Intel SYCL,在 Intel Arc Graphics [0xb080] 上跑通、验证、profiling。这是 cvcuda-ops-sycl 算子移植系列的一个算子(Resize、NMS、CvtColor、Normalize 已完成,见 `../OpResize/`、`../OpNMS/`、`../OpColorCvt/`、`../OpNormalize/`)。

## 这是什么

- **算法**:backward(inverse-map)warp。对每个输出像素 `(n,y,x)`:
  - `dx = dstX - xShift, dy = dstY - yShift`(double)
  - `srcX = (float)(dx*cos - dy*sin)`,`srcY = (float)(dx*sin + dy*cos)`
  - 在范围内(`srcX>-0.5 && srcX<W && srcY>-0.5 && srcY<H`)→ `BORDER_REPLICATE` + 插值采样;否则填 0(暴露角填黑)。
  - 插值:NEAREST(`floor(coord+0.5)`)、LINEAR(双线性 4-tap)、CUBIC(`GetCubicCoeffs` a=-0.5,4×4-tap)。
- **cos/sin 在 host 端用 double 算**(CV-CUDA 用 1 线程 device kernel `compute_warpAffine`)→ kernel 与 gold 共享 bit-identical 的 6 个 coeff `[c,s,xShift,-s,c,yShift]`,把发散隔离到 per-pixel float。
- **dtype**:U8/U16/S16/F32(S8/S32/F64/F16 不支持)。**通道** C∈{1,3,4}(C==2 null slot / C>4 拒绝)。输入输出 H/W 可不同(矩形旋转进不同尺寸画布),batch N 相同。布局 NHWC。
- **与 CV-CUDA 的差异**(已记录):(1) coeff 在 host double 计算;(2) 越界 dest 像素写 0(CV-CUDA 不动,靠调用方缓冲);(3) NHWC 裸指针 + 每像素一线程 + 通道循环,替代 `CreateInterpolationWrapNHW<T>` vector-type 派发;(4) 接口返回 **bool**(C/dtype/shape 校验)。

## 目录结构

```
OpRotate/
├── README.md                       # 本文件
├── 总结.md                         # 移植总结(代码理解 / CUDA→SYCL 映射 / 正确性 / 性能)
├── build.sh                        # 编译脚本(编 3 个 test 二进制;kernel header-only)
├── rotate.hpp                      # header-only:rotate_kernel(3 interp 分支)+ dispatch + dtype 派发
│                                   #   + Rotate 接口 + RotateImplement + create_rotate
├── rotate_helpers.hpp              # TypeMin/Max/nhwc_ptr/border_replicate/cubic_coeffs(CPU+device safe)
└── test/
    ├── rotate_gold.hpp             # CPU gold 参考(3 interp + REPLICATE;gold_rotate_res 暴露 pre-saturate float)
    ├── test_rotate.cpp             # 合成精确值 vs gold(NEAREST 任意角 bit-exact / LINEAR 0,180 / CUBIC 0)+ 负向
    ├── test_rotate_real.cpp        # 随机 23.5°,3 interp × U8/F32,FP 边界分析 + wall 性能
    └── test_rotate_profile.cpp     # VOX_PROFILE=1,逐 kernel event 计时 + wall + 吞吐
```

**组织原则**(同 OpResize / OpNMS / OpColorCvt / OpNormalize):kernel 全部 header-only(`rotate.hpp`),test 直接 include,`build.sh` 只编 3 个 test 二进制;CPU 参考逻辑移植自 CV-CUDA 的 `rotate.cu` + `InterpolationWrap.hpp`(CV-CUDA rotate 不带独立 test gold,本 gold 为自移植)。原始 NVIDIA 源在 CV-CUDA 0.16.0 仓库,此处不重复收录。

## 怎么编译

宿主机有 `icpx`(容器里没有,**必须宿主机编译**):

```bash
cd /work/bevfusion_migration/cvcuda_ops/OpRotate   # 板上镜像路径(本仓 = OpRotate/)
bash build.sh
```

编出 3 个二进制(`test_rotate` / `test_rotate_real` / `test_rotate_profile`),放到 `test/` 下。编译选项 `-fsycl -std=c++17 -O2 -g -I. -ffp-model=precise -fno-fast-math -fimf-precision=high`(精确 FP 标志同其它算子移植;LINEAR/CUBIC 仅 mul/add 无 div/sqrt,FP-division quirk 不适用,标志对结果无影响)。

## 怎么运行

> **必须**带 `ONEAPI_DEVICE_SELECTOR=opencl:gpu` —— Level Zero 后端在这台机器有 bug 会段错误。

```bash
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
./test/test_rotate                         # 期望:=== ALL TESTS PASSED ===
./test/test_rotate_real 4 128 3 10         # [N] [S] [C] [runs]
VOX_PROFILE=1 ./test/test_rotate_profile 4 128 3 20
```

## 算法要点(移植依据)

- 每输出像素一个线程,`idx → n=idx/(Hout*Wout), rem=idx%(Hout*Wout), y=rem/Wout, x=rem%Wout`。
- inverse map 用 double 中间量再 cast float(匹配 `rotate.cu:51-55`)。coeff `[c,s,xShift,-s,c,yShift]` 在 host double 算,memcpy 到 6-double device buffer(`d_aCoeffs`)。
- 越界填 0;否则按 interp 采样,`BORDER_REPLICATE` 索引 clamp 到 `[0,dim-1]`。
- CUBIC 用 `cubic_coeffs`(纯 float,无超越函数)→ CPU/GPU bit-identical。
- 单 kernel、无原子、无共享内存。

## 正确性

- **合成测试**(`test_rotate`,40 case = 4 dtype × {NEAREST 0/90/45/30/200°, LINEAR 0/180°, CUBIC 0°} × C∈{1,3,4}):bit-exact mismatch=0。NEAREST 任意角 bit-exact(host 算 coeff → srcX GPU/gold bit-identical → floor 一致);LINEAR@0/180°、CUBIC@0° 权重坍缩为单 1.0 tap → 精确复制。C==2/C==5 拒绝。
- **随机测试**(`test_rotate_real`,23.5°,N=4 S=128 C=3):全 PASS。U8 NEAREST/LINEAR exact;U8 CUBIC maxdiff=1(tol 1, 0 mismatch);F32 NEAREST exact;F32 LINEAR maxdiff=1.19e-7(1 ULP,tol 1e-4);F32 CUBIC maxdiff=2.98e-7(1 ULP,tol 1e-3)。

## 性能

`test_rotate_profile`(VOX_PROFILE,F32,best):
- S=128 N=4:`rotate_kernel` best=0.0322ms avg=0.0331ms,wall=0.0708ms,overhead=0.0386ms,kernel=2036 Mpix/s,wall=926 Mpix/s。
- S=720 N=1:kernel best=0.2435ms(LINEAR)/0.2434ms(CUBIC),wall=0.367ms,overhead=0.123ms,kernel=2129 Mpix/s,wall=1412 Mpix/s。

- **内存带宽受限**:每像素 4(LINEAR)/16(CUBIC)次 REPLICATE 随机读 + 1 写。S=720 时 LINEAR 与 CUBIC kernel 时间几乎相同(CUBIC 的 16-tap 额外读被 cache/延迟掩盖)→ 带宽非计算受限,tile+RPT 收益可忽略,不做(同 colorcvt/normalize)。
- S=128 时 overhead(coeff memcpy + launch + wait)> kernel;S=720 时 overhead ≈ wall 的 34%。无原子,无中段 host sync(forward 内仅一次 48 字节 coeff memcpy+wait)。

## 验证局限(诚实声明,重要)

1. **自洽验证,非对齐验证**:CV-CUDA rotate 不带独立 test gold,本 gold 是我自己从 `rotate.cu` + `InterpolationWrap.hpp` 移植的 CPU 参考 → 证明「SYCL kernel == 我的 CPU 参考」,**不直接证明 == NVIDIA CV-CUDA 实际输出**。若 kernel 与 gold 都从同一份源理解错,二者会一致地错,测试仍 PASS。缓解:kernel 与原 `.cu` + `InterpolationWrap.hpp` 逐行对照是最强独立检查。
2. **未与 NVIDIA CV-CUDA 做 bit 级 diff**:真正的金标准是同参数喂 NVIDIA `cvcudaRotate` 和本 SYCL kernel 比输出。这一步没做(需 NVIDIA GPU + 编译 CV-CUDA,Intel Arc 跑不了 CUDA 版)。
3. **FP 归因**:LINEAR/CUBIC 仅 mul/add(无 div/sqrt)→ Intel GPU float-**除法** quirk 不适用;残差是 FMA 收缩 ULP 噪声(F32 1 ULP),非除法发散。这是直接观测(权重仅乘加),归因较确定。

## 一点思考

- **host 算 coeff 是关键设计**:把 cos/sin 提到 host double,kernel 与 gold 共享 bit-identical coeff,使 NEAREST 能做到任意角 bit-exact(只剩 floor,无浮点发散)。只有 LINEAR/CUBIC 的 per-pixel 加权引入 1 ULP FMA 噪声,且远小于容差。
- **header-only 重构是机械改动**:本仓版本把原 `rotate.cpp` 实现并入 `rotate.hpp`(header-only),并把 `nv::linear_launch` → 原生 `q.parallel_for`,以匹配仓库自包含约定(kernel 数学逐字不变)。launch 机制改动属低风险机械重构,建议在板上重编一次确认(同 OpNMS / OpColorCvt / OpNormalize)。
