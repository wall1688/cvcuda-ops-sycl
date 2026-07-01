# cvcuda-ops-sycl

NVIDIA CV-CUDA 0.16.0 算子从 CUDA 移植到 Intel SYCL，目标平台 Intel Arc Graphics [0xb080] (B390) GPU，oneAPI 2025.3.3，OpenCL 后端。

这是 cvcuda 算子移植系列仓库。每个算子一个子目录，独立移植、独立验证。**首批 6 个算子已全部完成。**

## 已移植

| 算子 | 状态 | 功能 / 插值 | 数据类型 | 测试 |
|---|---|---|---|---|
| **OpResize** | ✅ 完成 | NEAREST / LINEAR / CUBIC / AREA + var-shape(NN/LIN/CUB) | F32 / U8 × C1/C3/C4 | 36 用例全 PASS |
| **OpRemap** | ✅ 完成 | src/map 各 NEAREST/LINEAR/CUBIC + 5 border + 3 mapValueType×alignCorners + var-shape | F32 / U8 × C1/C3/C4 | 45 用例全 PASS |
| **OpNMS** | ✅ 完成 | 全对 IoU mask-NMS（单 kernel、无原子、无 hash，输出 per-box keep-mask） | bboxes short4 / scores float / out uint8 | 4 合成（逐 bit）+ 随机 20000 box |
| **OpColorCvt** | ✅ 完成 | 6 族 71 个转换码（通道互换 / 灰度 / 打包YUV / HSV / YUV420 / YUV422） | 8U / 16U / 32F / F64（各族不同） | 71 码逐族逐 dtype + 负向 |
| **OpNormalize** | ✅ 完成 | plain + stddev(1/sqrt(scale²+eps))，按轴广播 | U8 / S8 / U16 / S16 / S32 / F32 × C1/C3/C4 | 42 合成（逐 bit）+ 随机 + 负向 |
| **OpRotate** | ✅ 完成 | NEAREST / LINEAR / CUBIC + BORDER_REPLICATE，backward warp | U8 / U16 / S16 / F32 × C1/C3/C4 | 40 合成（逐 bit）+ 随机 23.5° + 负向 |

> 最近复现验证：2026-06-30 OpResize 36 用例、2026-07-01 OpRemap 45 用例在 Intel Arc [0xb080] (opencl:gpu) 全 PASS，性能数据一致。各算子详见其 `README.md` §0 / 正确性章节。

### OpResize 亮点
- 4 种插值固定尺寸 + var-shape 变尺寸，忠实翻译 `priv/OpResize.cu` + `priv/legacy/resize_var_shape.cu`。
- LINEAR NIX+INTERSECT 读复用优化：U8 C3 上采样 4K **3.27→1.79ms（1.83×）**，F32 经 kNIX 门控无回退。
- 7 个 phase 快照（`snapshots/`），每步可回退复现。
- 详见 `OpResize/README.md`（工作说明 + 性能报告）与 `OpResize/总结.md`（移植总结）。

### OpRemap 亮点
- 2 次采样（map + src 各自独立插值）、5 种 border、3 种 mapValueType×alignCorners，忠实翻译 `priv/OpRemap.cu` + `priv/legacy/remap_var_shape.cu`。
- shape-agnostic sampler（`sample_impl<View>`）：TensorView 与 VarShapeBatch 共用同一份插值 + border 代码。
- 内部邻域快速路径优化：CUBIC REFLECT **4.87×** 加速（5.30→1.09ms），bit-identical 38 用例无回归。
- 7 个 phase 快照（`snapshots/`），每步可回退复现。
- 详见 `OpRemap/README.md` 与 `OpRemap/总结.md`。

### OpNMS 亮点
- 全对 IoU mask-NMS，单 kernel、无原子、无 hash 表、无中段 host sync → overhead 仅 **0.0097ms**，**15.67 Mbox/s**。
- 坐标全程整数运算，仅末尾一次 float 除法；tie-break 严格 `<`，结果确定。
- 随机大规模（20000 box）mismatch 2 个且 100% 为 IoU 恰在阈值的边界 case，归因 Intel GPU float 除法非 IEEE 舍入（设备特性，非 bug）。
- 详见 `OpNMS/README.md` 与 `OpNMS/总结.md`。

### OpColorCvt 亮点
- 6 族 71 个转换码忠实翻译 `priv/legacy/cvt_color.cu`：定点 Q15/Q14 系数、uchar 定点 hue + float sector LUT、YUV420/422 子采样布局逐位对齐。
- dtype 矩阵匹配原版（各族支持的 dtype 不同），未实现码返回 unsupported（不凭空发明）。
- 720p 带宽受限：BGR2GRAY **11.7 Gpix/s**，YUV2GRAY_420 memcpy 路径 **55 Gpix/s**（带宽极限）。
- 详见 `OpColorCvt/README.md` 与 `OpColorCvt/总结.md`。

### OpNormalize 亮点
- `out = SaturateCast((src-base)*mul*gscale+shift)`，plain 与 stddev(1/sqrt) 共用一个 kernel；base/scale 按轴广播（N/H/W/C 各自独立）。
- **plain 模式 bit-exact**（无除法，IEEE mul/add 无 FMA 收缩）；仅 stddev 的 1/sqrt 引入 ULP 级发散（maxdiff 1.5e-5，远小于 CV-CUDA 容差）。
- 1831 Mpix/s，单 kernel embarrassingly parallel。
- 详见 `OpNormalize/README.md` 与 `OpNormalize/总结.md`。

### OpRotate 亮点
- backward(inverse-map)warp，NEAREST/LINEAR/CUBIC + BORDER_REPLICATE；cos/sin 在 host 端 double 计算，kernel 与 gold 共享 bit-identical coeff。
- **NEAREST 任意角 bit-exact**（只剩 floor，无浮点发散）；LINEAR/CUBIC 仅 per-pixel 加权引入 1 ULP FMA 噪声。
- 输入输出 H/W 可不同（矩形旋转进不同尺寸画布）；S=720 时 **2129 Mpix/s**，带宽受限。
- 详见 `OpRotate/README.md` 与 `OpRotate/总结.md`。

## 计划移植

首批 6 个算子（Resize / Remap / NonMaximumSuppression / CvtColor / Normalize / Rotate）已全部完成。后续方向：更多 CV-CUDA 算子、算子链融合（减少中间写回）。

## 环境

- **平台**：Intel Arc Graphics [0xb080] (B390, ~29GB)，oneAPI 2025.3.3，OpenCL 后端。
- **编译**：宿主机 `icpx -fsycl`（容器无 icpx，必须宿主机编译）。
- **运行**：`export ONEAPI_DEVICE_SELECTOR=opencl:gpu`（Level Zero 后端在该机器有 bug，会段错误）。
- 本机为 VM 无真实 GPU，编译+运行在 Intel 物理机上进行。

## 仓库结构

```
├── OpResize/          # Resize（已完成）
│   ├── resize.hpp / resize_varshape.hpp   # header-only 模板 kernel
│   ├── common/                            # tensorwrap / varshape / saturate_cast / math_ops
│   ├── test/                              # 正确性 + profiling 测试
│   ├── README.md / 总结.md / claude_task_resize
│   └── build.sh
├── OpRemap/           # Remap（已完成）
│   ├── remap.hpp / remap_varshape.hpp     # header-only 模板 kernel
│   ├── common/                            # + border / sampler
│   ├── test/                              # 正确性（固定 + var-shape）+ profiling
│   ├── README.md / 总结.md
│   └── build.sh
├── OpNMS/             # NonMaximumSuppression（已完成）
│   ├── nms.hpp                            # header-only kernel
│   ├── test/                              # 正确性 + real + profiling
│   ├── scripts/                           # pexpect SSH 传输/编译脚本（凭据走环境变量）
│   ├── README.md / 总结.md
│   └── build.sh
├── OpColorCvt/        # CvtColor（已完成）
│   ├── colorcvt.hpp / cvt_helpers.hpp     # header-only 6 族 kernel + dispatch
│   ├── test/                              # 正确性（71 码）+ profiling
│   ├── README.md / 总结.md
│   └── build.sh
├── OpNormalize/       # Normalize（已完成）
│   ├── normalize.hpp / normalize_helpers.hpp
│   ├── test/                              # 正确性 + real + profiling
│   ├── README.md / 总结.md
│   └── build.sh
├── OpRotate/          # Rotate（已完成）
│   ├── rotate.hpp / rotate_helpers.hpp
│   ├── test/                              # 正确性 + real + profiling
│   ├── README.md / 总结.md
│   └── build.sh
└── snapshots/         # OpResize / OpRemap 各 7 个 phase 源码快照（可回退复现）
```

## 验证说明

正确性测试采用「同算法 CPU 参考」自洽验证（GPU 输出 == 同算法 CPU 实现），**未与 NVIDIA CV-CUDA 二进制做 bit 级 diff**（需 NVIDIA GPU，未做）。CPU 参考逻辑移植自 CV-CUDA 自带的 gold（GoldNMS / CvtColorUtils / TestOpNormalize 等）或从原 `.cu` 逐行移植，kernel 与 gold 各自忠实两份原始实现，二者一致是较强的正确性证据。关键算法逻辑逐字照抄原版以降低理解错风险。详见各算子文档的「验证局限」章节。

## 许可

- 本仓库的移植代码（SYCL kernel、wrapper、测试）采用 MIT 许可（见 `LICENSE`）。
- 原始算法来自 NVIDIA CV-CUDA 0.16.0（Apache-2.0）。`OpResize/OpResize.cpp` 与 `OpRemap/OpRemap.cpp` 为原始 NVIDIA 框架源，保留作参考，不参与编译。
