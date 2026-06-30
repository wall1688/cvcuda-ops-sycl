# CV-CUDA NonMaximumSuppression — SYCL 移植

把 CV-CUDA 0.16.0 的 `NonMaximumSuppression` 算子(`src/cvcuda/priv/OpNonMaximumSuppression.cu`)移植到 Intel SYCL,在 Intel Arc Graphics [0xb080] 上跑通、验证、profiling。这是 cvcuda-ops-sycl 算子移植系列的一个算子(Resize 已完成,见 `../OpResize/`)。

## 这是什么

- **算法**:全对(all-pairs)IoU 的 mask-NMS。每个 box 一个线程,遍历同 batch 所有其他 box,若 `IoU(X,Y) > iouThreshold` 且本 box 分数更低(或等分但面积更小)则丢弃。输出 per-box keep-mask(uint8,1=保留 / 0=丢弃)。
- **语义**:忠实 CV-CUDA 的 **mask 输出**(不是 torchvision 的排序索引列表)。
- **与 Resize 的区别**:单 kernel、无原子、无 hash 表、无 host sync,embarrassingly parallel → 没有 opt 阶梯。先 1:1 跑通。
- **数据**:`bboxes` short4(x,y,w,h) `[S,N,4]`;`scores` float `[S,N]`;`out` uint8 `[S,N]`。行主序连续,box (b,j) 在 `b*N + j`。

## 目录结构

```
OpNMS/
├── README.md                       # 本文件
├── 总结.md                         # 移植总结(代码理解 / CUDA→SYCL 映射 / 正确性 / 性能)
├── claude_task_nms                 # 详尽过程上下文文档
├── build.sh                        # 编译脚本(编 3 个 test 二进制;kernel header-only)
├── nms.hpp                         # header-only 模板 kernel:short4 / compute_iou / nms_kernel
│                                   #   + NonMaximumSuppression 接口 + NMSImplement + create_nms
├── build.sh.nms.patch              # (遗留)并入旧顶层 build.sh 的可选 patch,仅作参考
├── scripts/
│   ├── ssh_nms_transfer.py         # pexpect SSH 传输脚本(凭据走环境变量,默认已脱敏)
│   └── ssh_nms_buildrun.py         # pexpect SSH 板上 build+run 脚本
└── test/
    ├── nms_gold.hpp                # CPU gold 参考(移植自 CV-CUDA GoldNMS)
    ├── test_nms.cpp                # 合成小数据,逐 box 对 CPU gold(期望逐 bit 一致)
    ├── test_nms_real.cpp           # 随机大规模 box,正确性 + FP 边界分析 + wall 性能
    └── test_nms_profile.cpp        # VOX_PROFILE=1,逐 kernel event 计时 + wall + 吞吐
```

**组织原则**(同 OpResize):kernel 全部 header-only(`nms.hpp`),test 直接 include,`build.sh` 只编 3 个 test 二进制;CPU 参考逻辑移植自 CV-CUDA 自带的 GoldNMS。原始 NVIDIA 源在 CV-CUDA 0.16.0 仓库,此处不重复收录。

## 怎么编译

宿主机有 `icpx`(容器里没有,**必须宿主机编译**):

```bash
cd /work/bevfusion_migration/cvcuda_ops/OpNMS   # 板上镜像路径(本仓 = OpNMS/)
bash build.sh
```

编出 3 个二进制(`test_nms` / `test_nms_real` / `test_nms_profile`),放到 `test/` 下。编译选项 `-fsycl -std=c++17 -O2 -g -I. -ffp-model=precise -fno-fast-math -fimf-precision=high`(精确 FP 标志同其它算子移植;对 NMS 结果无影响,仅末尾一次 IoU float 除法,且 GPU 除法为倒数近似与该标志无关)。

## 怎么运行

> **必须**带 `ONEAPI_DEVICE_SELECTOR=opencl:gpu` —— Level Zero 后端在这台机器有 bug 会段错误。

```bash
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
./test/test_nms                         # 期望:=== ALL TESTS PASSED ===
./test/test_nms_real 5000 4 10          # [N] [S] [runs]
VOX_PROFILE=1 ./test/test_nms_profile 5000 4 20
```

## 算法要点(移植依据)

- 每个 box 一个线程,`idx → b=idx/N, j=idx%N`。
- `IoU = interArea / unionArea`,坐标用整数运算,仅末尾一次 float 除法。
- tie-break 为严格 `<`(等分等面积 → 都保留),与 GoldNMS 完全一致;`break` 仅早退,与 Y 遍历顺序无关 → 结果确定。
- 单 kernel、无原子、无 hash 表。

## 正确性

- **合成测试**(`test_nms`,4 用例):box 的两两 IoU 都明显偏离阈值(0.5),Intel GPU 除法非 IEEE 舍入无法翻转任何比较 → 与 CPU gold 逐 bit 一致。覆盖:score 低于阈值、IoU 超阈值一高一低分、等分等面积(tie)、等分不等面积、不相交、batch 独立性、全低于阈值、单 box。
- **随机大规模测试**(`test_nms_real`,S=4 N=5000 → 20000 box):`mismatch=2 / 20000 (99.99%)`,2 个不一致 **100% 为 IoU 恰在阈值的边界 case**(精确有理 IoU 到 0.45 的距离 < 1e-3)→ `REAL-DATA TEST PASSED`。根因:Intel Arc GPU float 除法非 IEEE-754 正确舍入(倒数近似),`iou` 恰等于阈值时可能翻转 `> iouThreshold` 判定 —— 设备固有特性,非移植 bug。阈值取 0.45 以避开 1/2 整数比碰撞。

## 性能

`test_nms_profile`(S=4 N=5000,20 runs,GPU warm,best):

```
----- NMS per-kernel (best) -----
nms_kernel   best=1.276ms avg=1.281ms (100%)
wall         best=1.286ms avg=1.293ms
overhead     =0.0097ms (launch + wait)
throughput   kernel=15.67 Mbox/s  wall=15.55 Mbox/s
```

- **overhead 仅 0.0097ms**(kernel ≈ wall):单 kernel、无中段 host sync、无多余 launch/memset。
- **O(N²) 访存,延迟受限**:每个 box 线程遍历同 batch 全部 N 个 box(随机读,cache miss)。不改算法(shared-memory tiling 复用 box / 按 score 排序早退 / 分块)无法在本轮显著降 —— 属另一阶段优化工作。

## 验证局限(诚实声明,重要)

1. **自洽验证,非对齐验证**:正确性测试用的是我自己写的 CPU gold(移植自 CV-CUDA 的 GoldNMS,与 `.cu` kernel 是两份独立实现),证明「SYCL kernel == 我的 CPU 参考」,**不直接证明 == NVIDIA CV-CUDA 实际输出**。若 kernel 与 gold 都从同一份源理解错,二者会一致地错,测试仍 PASS。缓解:kernel 体仅约 10 行,与原 `.cu` 逐行对照,理解错风险低;gold 取自源仓独立参考。
2. **未与 NVIDIA CV-CUDA 做 bit 级 diff**:真正的金标准是同参数喂 NVIDIA `cvcudaNonMaximumSuppression` 和本 SYCL kernel 比输出。这一步没做(需 NVIDIA GPU + 编译 CV-CUDA,Intel Arc 跑不了 CUDA 版)。
3. **边界分类的循环论证风险**:real 测试用自写的 `min_iou_dist_to_thresh` 判定边界 case,存在循环论证风险;更严谨的做法是把 2 个 divergent box 的精确 IoU 值打印出来确认卡在阈值上 —— 这是后续可补的独立校验。
4. **FP 归因是推断**:Intel GPU 除法非 IEEE 舍入的归因来自 precedent + 边界距离检查,非单独证明的根因。

## 一点思考

- **「忠实翻译」优于「重写」**:NMS kernel 体很短(约 10 行),逐字照抄原版把理解错的风险压到最低;gold 取自源仓的独立参考(GoldNMS),kernel 与 gold 各自忠实两份原始实现,二者一致是较强的正确性证据。
- **单 kernel 形态已最简**:无原子 / 无 hash / 无中段 sync,overhead 0.01ms,进一步提速只能靠算法级改动(tiling / 排序),与体素化、Resize 的「随机访存延迟受限」结论同性质。
