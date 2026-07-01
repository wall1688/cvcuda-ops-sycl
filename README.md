# cvcuda-ops-sycl

NVIDIA CV-CUDA 0.16.0 算子从 CUDA 移植到 Intel SYCL，目标平台 Intel Arc Graphics [0xb080] (B390) GPU，oneAPI 2025.3.3，OpenCL 后端。

这是 cvcuda 算子移植系列仓库。每个算子一个子目录，独立移植、独立验证。

## 已移植

| 算子 | 状态 | 插值/功能 | 数据类型 | 测试 |
|---|---|---|---|---|
| **OpResize** | ✅ 完成 | NEAREST / LINEAR / CUBIC / AREA + var-shape(NN/LIN/CUB) | F32 / U8 × C1/C3/C4 | 36 用例全 PASS |
| **OpRemap** | ✅ 完成 | src/map 各 NEAREST/LINEAR/CUBIC + 5 border + 3 mapValueType×alignCorners + var-shape | F32 / U8 × C1/C3/C4 | 45 用例全 PASS |

> 最近复现验证：2026-06-30 在 Intel Arc [0xb080] (opencl:gpu) 重跑 36 用例全 PASS，性能数据一致（见 `OpResize/README.md` §0）。OpRemap 45 用例于 2026-07-01 全 PASS（见 `OpRemap/README.md` §0）。

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

## 计划移植

OpRotate / OpCvtColor / OpNormalize / OpNonMaximumSuppression（待启动）。

## 环境

- **平台**：Intel Arc Graphics [0xb080] (B390, ~29GB)，oneAPI 2025.3.3，OpenCL 后端。
- **编译**：宿主机 `icpx -fsycl`（容器无 icpx，必须宿主机编译）。
- **运行**：`export ONEAPI_DEVICE_SELECTOR=opencl:gpu`（Level Zero 后端在该机器有 bug，会段错误）。
- 本机为 VM 无真实 GPU，编译+运行在 Intel 物理机上进行。

## 仓库结构

```
├── OpResize/          # Resize 算子（已完成）
│   ├── resize.hpp / resize_varshape.hpp   # header-only 模板 kernel
│   ├── common/                            # tensorwrap / varshape / saturate_cast / math_ops
│   ├── test/                              # 正确性 + profiling 测试
│   ├── README.md / 总结.md / claude_task_resize
│   └── build.sh
├── OpRemap/          # Remap 算子（已完成）
│   ├── remap.hpp / remap_varshape.hpp     # header-only 模板 kernel
│   ├── common/                            # tensorwrap / varshape / saturate_cast / math_ops / border / sampler
│   ├── test/                              # 正确性（固定 + var-shape）+ profiling 测试
│   ├── README.md / 总结.md
│   └── build.sh
└── snapshots/        # 各算子 phase 源码快照（OpResize / OpRemap，可回退复现）
```

## 验证说明

正确性测试采用「同算法 CPU 参考」自洽验证（GPU 输出 == 同算法 CPU 实现），**未与 NVIDIA CV-CUDA 二进制做 bit 级 diff**（需 NVIDIA GPU，未做）。关键算法逻辑逐字照抄原版以降低理解错风险。详见各算子文档的「验证局限」章节。

## 许可

- 本仓库的移植代码（SYCL kernel、wrapper、测试）采用 MIT 许可（见 `LICENSE`）。
- 原始算法来自 NVIDIA CV-CUDA 0.16.0（Apache-2.0）。`OpResize/OpResize.cpp` 与 `OpRemap/OpRemap.cpp` 为原始 NVIDIA 框架源，保留作参考，不参与编译。
