# SP + LightGlue ONNX Runtime C++ Demo

这是一个基于 **OpenCV + ONNX Runtime(C++)** 的特征匹配示例工程：
- 使用 **SuperPoint** 提取关键点和描述子
- 使用 **LightGlue** 进行特征匹配
- 输出匹配可视化图像，并打印推理耗时

工程内提供三个可执行程序：
- `sp_lg_demo`：完整版（兼容性更强，自动解析 I/O 名称、兼容多种 keypoints/matches 输出形状）
- `sp_lg_demo_simple`：精简版（固定 I/O 名称与常见 shape，支持 one-shot 与常驻交互模式，支持 `--cuda` / `--profile`）
- `sp_lg_demo_gpu`：GPU 默认版（启动即自动尝试 CUDA，失败自动回退 CPU；支持 one-shot 与常驻模式）

---

## 1. 项目功能

### `src/sp_lg_demo.cpp`（`sp_lg_demo`）
主要特点：
- 自动打印并解析 ONNX 模型 I/O 名称和 shape
- SuperPoint 关键点解析兼容：
  - 元素类型：`float/double/int32/int64`
  - shape：`[N,2]`、`[1,N,2]`、`[1,2,N]`、`[2,N]`
- 可自动判断并恢复 float keypoints 的归一化坐标（`[0,1]` 或 `[-1,1]`）到像素坐标
- LightGlue 匹配解析兼容：
  - `matches0` 支持 `[M,2]` 或 mapping 形式（`[L]` / `[1,L]`）
  - `mscores0` 可选，并支持阈值过滤
- 绘制匹配连线并保存结果图

适合：模型导出格式不稳定、需要强鲁棒兼容时使用。

### `src/sp_lg_demo_simple.cpp`（`sp_lg_demo_simple`）
主要特点：
- 假设模型 I/O 固定：
  - SuperPoint: `image -> keypoints, scores, descriptors`
  - LightGlue: `kpts0, kpts1, desc0, desc1 -> matches0, mscores0`
- 支持两种运行模式：
  - **one-shot**：命令行一次处理一对图像
  - **resident**：常驻进程，循环读入多对图像（减少重复加载模型开销）
- 支持 `--cuda` 自动尝试启用 CUDA Execution Provider（失败自动回退 CPU）
- 支持 `--profile` 输出 ONNX Runtime profiling 文件（用于确认节点执行 EP）
- 支持 `warmup` 和多次 `iters` 统计平均耗时

适合：你当前这类固定导出模型、关注推理效率测试与批量跑图场景。

### `src/sp_lg_demo_gpu.cpp`（`sp_lg_demo_gpu`）
主要特点：
- 基于 `sp_lg_demo_simple.cpp`，运行时默认自动尝试启用 CUDA EP（`device_id=0`）
- 如果 GPU 不可用或算子不支持，会自动回退到 CPU EP
- 不需要命令行传 `--cuda` 或 `--device_id`
- 支持 `--profile` 生成 ORT profiling 文件
- 支持 `min_kpt_dist` 参数：在输入 LightGlue 前按最小点间距过滤 SuperPoint 关键点，用于减少匹配耗时
- 当 `min_kpt_dist=-1` 时，禁用距离过滤，保留全部关键点

---

## 2. 目录结构

```text
sp_lg_demo/
├── CMakeLists.txt
├── src/
│   ├── sp_lg_demo.cpp
│   ├── sp_lg_demo_simple.cpp
│   └── sp_lg_demo_gpu.cpp
├── models/
│   ├── superpoint.onnx
│   └── lightglue_sim.onnx
├── data/
│   ├── img0.png
│   └── img1.png
└── output/
```

> `output/` 可自行创建：`mkdir -p output`

---

## 3. 依赖与版本

### 必需依赖
- **CMake** >= `3.10`
- **C++ 编译器**：支持 `C++17`（如 `g++` >= 9）
- **OpenCV**：建议 `4.x`
- **ONNX Runtime C/C++**：建议 `1.14+`，本项目默认路径指向 `1.24.2 GPU` 发行包

`CMakeLists.txt` 中默认：
- `CMAKE_CXX_STANDARD 17`
- `ORT_DIR=/opt/onnxruntime/onnxruntime-linux-x64-gpu-1.24.2`

> 若你的 ORT 安装路径不同，编译时请通过 `-DORT_DIR=...` 显式指定。

### CUDA（可选）
若需要 `--cuda`：
- 你安装的 ONNX Runtime 必须是包含 `CUDAExecutionProvider` 的 GPU 版本
- 机器需具备可用 NVIDIA 驱动与 CUDA 运行环境
- 若 CUDA EP 不可用，程序会自动回退 CPU，并打印日志提示

---

## 4. 编译

在项目根目录执行：

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORT_DIR=/opt/onnxruntime/onnxruntime-linux-x64-gpu-1.24.2
cmake --build build -j
```

编译后可执行文件：
- `build/sp_lg_demo`
- `build/sp_lg_demo_simple`
- `build/sp_lg_demo_gpu`

---

## 5. 运行方式

## 5.1 `sp_lg_demo`（完整版）

```bash
./build/sp_lg_demo \
  ./models/superpoint.onnx \
  ./models/lightglue_sim.onnx \
  ./data/img0.png \
  ./data/img1.png \
  ./output/matches.png \
  480 752
```

参数说明：
1. `sp.onnx`
2. `lg.onnx`
3. `img0`
4. `img1`
5. `out.png`
6. `H`
7. `W`

## 5.2 `sp_lg_demo_simple`（推荐日常使用）

### A) one-shot 模式

```bash
./build/sp_lg_demo_simple \
  ./models/superpoint.onnx \
  ./models/lightglue_sim.onnx \
  ./data/img0.png \
  ./data/img1.png \
  ./output/matches.png \
  480 752 10 --cuda --profile
```

含义：
- `10`：迭代次数 `iters`
- `--cuda`：尝试启用 CUDA EP
- `--profile`：开启 ORT profiling（退出后写 profile 文件）

### B) resident 常驻模式

启动：

```bash
./build/sp_lg_demo_simple \
  ./models/superpoint.onnx \
  ./models/lightglue_sim.onnx \
  480 752 10 10 --cuda --profile
```

含义：
- `480 752`：输入尺寸 `H W`
- 第一个 `10`：`iters`
- 第二个 `10`：`warmup`

启动后在终端逐行输入：

```text
<img0_path> <img1_path> [out_path]
```

输入 `q` / `quit` / `exit` 退出。

## 5.3 `sp_lg_demo_gpu`（默认优先 GPU）

### A) one-shot 模式

```bash
./build/sp_lg_demo_gpu \
  ./models/superpoint.onnx \
  ./models/lightglue_sim.onnx \
  ./data/img0.png \
  ./data/img1.png \
  ./output/matches.png \
  480 752 10 15 --profile
```

### B) resident 常驻模式

```bash
./build/sp_lg_demo_gpu \
  ./models/superpoint.onnx \
  ./models/lightglue_sim.onnx \
  480 752 10 10 15 --profile
```

说明：
- 程序会自动尝试 CUDA，失败时自动回退 CPU
- 不需要再传 `--cuda` 或 `--device_id`
- `min_kpt_dist` 含义：最小关键点间距（像素）
  - 值越大，保留关键点越少，LightGlue 通常越快
  - 建议从 `8~20` 开始调参
  - 传 `-1` 表示禁用过滤，保留全部关键点

参数顺序：
- one-shot：`<sp.onnx> <lg.onnx> <img0> <img1> <out.png> <H> <W> [iters] [min_kpt_dist] [--profile]`
- resident：`<sp.onnx> <lg.onnx> <H> <W> [iters] [warmup] [min_kpt_dist] [--profile]`

禁用过滤示例：

```bash
./build/sp_lg_demo_gpu \
  ./models/superpoint.onnx \
  ./models/lightglue_sim.onnx \
  ./data/img0.png \
  ./data/img1.png \
  ./output/matches_all_kpts.png \
  480 752 10 -1 --profile
```

---

## 6. 模型 I/O 约定

你当前使用的模型通常满足：

### SuperPoint
- Input: `image`，shape `[1,1,H,W]`，`float32`，值域 `[0,1]`
- Outputs:
  - `keypoints`：常见 `[N,2]` 或 `[1,N,2]`
  - `scores`
  - `descriptors`：常见 `[1,N,D]`

### LightGlue
- Inputs:
  - `kpts0` `[1,N0,2]`
  - `kpts1` `[1,N1,2]`
  - `desc0` `[1,N0,D]`
  - `desc1` `[1,N1,D]`
- Outputs:
  - `matches0`
  - `mscores0`

如果你替换了导出的 ONNX 模型且 I/O 名称或 shape 有变化，优先使用 `sp_lg_demo`（`sp_lg_demo.cpp`）进行兼容与排查。

---

## 7. SuperPoint 与 LightGlue 引用说明

本仓库中的 `superpoint.onnx` 与 `lightglue_sim.onnx` 属于第三方模型转换产物。

建议参考：
- SuperPoint（论文）：*SuperPoint: Self-Supervised Interest Point Detection and Description*
- LightGlue（论文）：*LightGlue: Local Feature Matching at Light Speed*
- LightGlue 官方仓库：https://github.com/cvg/LightGlue



---

## 8. 常见问题

- 运行时报 `libonnxruntime.so` 找不到：
  - 确认 `-DORT_DIR=...` 正确，且该目录下存在 `lib/` 或 `lib64/` 中的 `libonnxruntime.so`
- `--cuda` 未生效：
  - 查看日志中 `available providers` 是否包含 `CUDAExecutionProvider`
  - 可用 `--profile`，退出后查看 profile JSON 中节点执行 EP
- 匹配速度慢：
  - 优先使用 `sp_lg_demo_gpu`
  - 增大 `min_kpt_dist`（如 `10/12/15/20`）可减少输入到 LightGlue 的关键点数量
  - 若需要对照完整精度，可将 `min_kpt_dist` 设为 `-1` 禁用过滤
- 图像读取失败：
  - 检查输入路径是否存在，确认格式可被 OpenCV 解码

---

## 9. License 提示

本仓库代码（`src/*.cpp`）可按你的项目许可证发布；
但模型权重与第三方项目代码/导出脚本的许可证可能不同，请务必分别遵循其 License 条款。
