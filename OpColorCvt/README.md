# CV-CUDA CvtColor — SYCL 移植

把 CV-CUDA 0.16.0 的 `CvtColor` 算子(`src/cvcuda/priv/legacy/cvt_color.cu` tensor path)移植到 Intel SYCL,在 Intel Arc Graphics [0xb080] 上跑通、验证、profiling。这是 cvcuda-ops-sycl 算子移植系列的一个算子(Resize、NMS 已完成,见 `../OpResize/`、`../OpNMS/`)。

## 这是什么

- **算法**:CV-CUDA 实现的 6 个颜色码族,共 71 个转换码(NVCVColorConversionCode 数值)。每像素一个线程,NHWC 连续布局。
  - A 通道互换/alpha(码 0-5):纯通道重排 ±alpha。
  - B 灰度(6-11):BGR→GRAY(定点 Q15 + float)、GRAY→BGR(复制)。
  - C 打包 YUV(82-85):Q14 定点 + float 正/逆变换。
  - D HSV(40,41,54,55,66,67,70,71):uchar 定点 hdiv/sdiv 表 + 位运算 hue;float 6 段 sector LUT(3 个 int32 位掩码)。
  - E YUV420(90-106,127-134,140-147):NV12/NV21 半平面 + I420/YV12 平面,4:2:0 子采样。
  - F YUV422(107-124,不含 109/110/113/114):YUY2/YVYU/UYVY 4:2:2。
- **"全量"的真实含义**:CV-CUDA 的 `funcs[]` dispatch 表里,XYZ/Lab/Luv/HLS/Bayer/BGR565/555/mRGBA 全部是 `0`(返回 INVALID_PARAMETER,**无 kernel**)。忠实移植 = 移植这 6 族 + 对未实现码返回 unsupported(不凭空发明)。
- **dtype 矩阵(匹配原版)**:A + GRAY→BGR = 全 dtype;BGR→GRAY + 打包 YUV = 8U/16U/32F;HSV = 8U/32F;YUV420/422 = 8U only。16F 走 uint16_t 整数路径;64F 走 double。
- **数据**:NHWC 连续 USM。`in/out` 设备指针,`in_dtype == out_dtype`(单 dtype 入参)。YUV420 标量 `[N,1.5H,W,1]`;YUV422 标量 `[N,H,2W,1]`。

## 目录结构

```
OpColorCvt/
├── README.md                       # 本文件
├── 总结.md                         # 移植总结(代码理解 / CUDA→SYCL 映射 / 正确性 / 性能)
├── build.sh                        # 编译脚本(编 2 个 test 二进制;kernel header-only)
├── colorcvt.hpp                    # header-only:6 族 kernel + dispatch + dtype 派发
│                                   #   + CvtColor 接口 + CvtColorImplement + create_cvt
├── cvt_helpers.hpp                 # SaturateCast/Alpha/TypeMax/TypeMin/NHWC ptr/颜色常量/cv_descale
└── test/
    ├── cvt_gold.hpp                # CPU gold 参考(移植自 CV-CUDA CvtColorUtils,6 族)
    ├── test_cvt.cpp                # 71 码逐族逐 dtype vs gold + 负向测试
    └── test_cvt_profile.cpp        # VOX_PROFILE=1,逐 kernel event 计时 + wall + 吞吐
```

**组织原则**(同 OpResize / OpNMS):kernel 全部 header-only(`colorcvt.hpp`),test 直接 include,`build.sh` 只编 2 个 test 二进制;CPU 参考逻辑移植自 CV-CUDA 自带的 `CvtColorUtils`。原始 NVIDIA 源在 CV-CUDA 0.16.0 仓库,此处不重复收录。

## 怎么编译

宿主机有 `icpx`(容器里没有,**必须宿主机编译**):

```bash
cd /work/bevfusion_migration/cvcuda_ops/OpColorCvt   # 板上镜像路径(本仓 = OpColorCvt/)
bash build.sh
```

编出 2 个二进制(`test_cvt` / `test_cvt_profile`),放到 `test/` 下。编译选项 `-fsycl -std=c++17 -O2 -g -I. -ffp-model=precise -fno-fast-math -fimf-precision=high`(精确 FP 标志同其它算子移植;float 路径做浮点 dot/除法,precise 标志降低非结合性漂移,测试本身也按 CV-CUDA 原容差放行)。

## 怎么运行

> **必须**带 `ONEAPI_DEVICE_SELECTOR=opencl:gpu` —— Level Zero 后端在这台机器有 bug 会段错误。

```bash
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
./test/test_cvt                         # 期望:=== ALL TESTS PASSED ===
VOX_PROFILE=1 ./test/test_cvt_profile   # 逐 kernel event 计时
```

## 算法要点(移植依据)

- 每像素一个线程,`idx → n=idx/(H*W), rem=idx%(H*W), y=rem/W, x=rem%W`。
- `bidx` 在 load 端路由 R/B(BGR vs RGB);store 端按 `bidx` 决定输出顺序。codes 0,1 = copy ±alpha;2,3,4,5 = reverse [2,1,0] ±alpha。
- 整型路径用定点 `CV_DESCALE`(rounds),float 路径用浮点系数;`if constexpr is_integral` 双路径保留。
- `cuda::SaturateCast<T>` → `sat_int`(int→无符号 clamp)/ `sat_cast_f`(float→整,`sycl::rint` round-to-nearest-even + clamp,匹配 PTX `cvt.rni`)。
- YUV420/422 子采样布局逐位对齐原版:semi-planar chroma 在 `H..1.5H` 行交错;planar U/V 平面按 2 子采样行打包;`uidx` 按 code 决定 U/V 顺序。
- `YUV2GRAY_420`(码 106)= `q.memcpy` 逐 sample 拷 Y 平面(原版 `cudaMemcpy2DAsync`)。
- **无原子、无共享内存、无纹理** —— 纯 per-pixel 寄存器运算。

## 正确性

71 个码逐族逐 dtype vs CPU gold,全部 `=== ALL TESTS PASSED ===`:

- A 族:U8/U16/S32/F32/F64 逐 bit 一致(纯重排,tol 0)。
- B 族:BGR→GRAY U8(tol 1)/U16(tol 2)/F32(ERR1_4);GRAY→BGR exact。定点 CV_DESCALE 与 gold float+truncate 在容差内。
- C 族:U8(tol 1)/U16(tol 2/1)/F32(ERR1_4)。SaturateCast 验证。
- D 族 HSV:U8(tol 1)/F32(BGR2HSV ERR2_3 circular hue / HSV2BGR ERR1_4)。uchar 定点 hue + float sector LUT 验证。
- E 族 YUV420:YUV→BGR(tol 2)/BGR→YUV(tol 1)/GRAY(exact)。semi-planar + planar 4:2:0 子采样布局逐位对齐。
- F 族 YUV422:YUV→BGR(tol 2)/GRAY(exact)。4:2:2 宏像素路由验证。
- 负向:16F 加 alpha 拒绝、未实现码(50)拒绝、通道数不符拒绝。

## 性能

`test_cvt_profile`(1280×720,N=1,20 runs,GPU warm,best):

```
code                    kernel_ms   wall_ms      Mpix/s
BGR2BGRA   (A)            0.097      0.137       9513
BGR2RGB    (A)            0.196      0.230       4709
BGR2GRAY   (B)            0.079      0.096      11703
BGR2YUV    (C)            0.174      0.208       5288
BGR2HSV    (D U8)         0.128      0.161       7199
BGR2HSV    (D F32)        0.161      0.245       5715
YUV2BGR_NV12(E)          0.155      0.186       5946
YUV2GRAY_420(E)          0.017      0.023      55126   ← memcpy Y 平面,带宽极限
YUV2BGR_YUY2(F)          0.150      0.183       6131
```

- **内存带宽受限**:每像素读一次写一次,NHWC 连续访问已合并(coalesced)。720p 计算核 4.7-11.7 Gpix/s,memcpy 55 Gpix/s。
- overhead(wall−kernel)仅 0.03-0.08ms(launch + wait),单 kernel 无中段 host sync。
- 原版 `color_conversion_common` 的 tile+RPT 驱动主要利于**计算受限** kernel 的寄存器复用;但本算子**带宽受限 + 访问已合并** → tile+RPT 收益可忽略,不做。进一步提速需算法级改动(融合上下游算子、减少中间写回),属另一阶段工作(同 voxelization/NMS)。

## 验证局限(诚实声明,重要)

1. **自洽验证,非对齐验证**:正确性测试用的是我自己移植的 CPU gold(源自 CV-CUDA `CvtColorUtils`,与 `.cu` kernel 是两份独立实现),证明「SYCL kernel == 我的 CPU 参考」,**不直接证明 == NVIDIA CV-CUDA 实际输出**。若 kernel 与 gold 都从同一份源理解错,二者会一致地错,测试仍 PASS。缓解:kernel 与原 `.cu` 逐行对照;gold 取自源仓独立参考。
2. **未与 NVIDIA CV-CUDA 做 bit 级 diff**:真正的金标准是同参数喂 NVIDIA `cvcudaCvtColor` 和本 SYCL kernel 比输出。这一步没做(需 NVIDIA GPU + 编译 CV-CUDA,Intel Arc 跑不了 CUDA 版)。
3. **容差覆盖 FP 差异**:HSV/YUV float 路径含除法,Intel Arc GPU float 除法非 IEEE-754 正确舍入(倒数近似);这里用 CV-CUDA 原版容差(整型 tol 1-2、F32 ~1e-4)覆盖,未单独做边界归类(不像 NMS 那样做 IoU 边界分析)。归因是推断,非单独证明的根因。

## 一点思考

- **「忠实翻译」优于「重写」**:6 族 kernel 的定点系数、位运算 hue、sector LUT、YUV 子采样布局全部逐行照搬原版,把理解错的风险压到最低;gold 取自源仓独立参考,kernel 与 gold 各自忠实两份原始实现,二者一致是较强的正确性证据。
- **header-only 重构是机械改动**:本仓版本把原 `colorcvt.cpp` 的实现并入 `colorcvt.hpp`(header-only),并把 `nv::linear_launch` → 原生 `q.parallel_for`,以匹配仓库自包含约定(kernel 数学逐字不变)。launch 机制改动属低风险机械重构,建议在板上重编一次确认(同 OpNMS)。
