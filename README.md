Scan Algorithm Module (TensorRT Pipeline) / 扫描算法模块（TensorRT 流水线）
========================================================================

# Overview / 概述
---------------
This project builds a shared library (`libAickTensorrt.so`) and a test
runner (`pipeline_test`) for a 10X image pipeline:
1) YOLO detection (1600x1600 input)
2) Impurities classification (2-class)
3) Regression scoring (length / dispersion / num)

本项目生成动态库 `libAickTensorrt.so` 和测试程序 `pipeline_test`，用于 10X大图流水线：
1）YOLO 检测（输入 1600x1600）  
2）杂质二分类  
3）回归打分（长度/分散度/数量）

The pipeline is driven by `10Xconfig.txt` and logs to `./Log/10X/`.  
流水线由 `10Xconfig.txt` 驱动，日志输出到 `./Log/10X/`。

# Project Layout / 目录结构
-------------------------
- `src/` / `include/`: core implementation (TensorRT + OpenCV)  /核心实现（TensorRT + OpenCV）
- `engine/`: TensorRT engine files (*.engine)  /TensorRT 引擎文件
- `10Xconfig.txt`: runtime configuration  /运行配置
- `main.cpp`: example runner used for quick tests  /测试入口
- `make.sh`: one-shot build + run script  /一键编译运行脚本
- `output/`: default image output directory (if enabled)  /结果图默认输出目录

# Platform / 平台与版本
---------------------
**Tested**  
- Jetson Orin Nano Super(8GB)
- JetPack 6.2.1  (L4T 36.x)
- TensorRT 10.3.0  
- OpenCV 4.14.0 (with CUDA)

**已验证**  
- Jetson Orin Nano Super(8GB)
- JetPack 6.2.1  (L4T 36.x)
- TensorRT 10.3.0  
- OpenCV 4.14.0（CUDA 版）

Build / 编译
------------
**Requirements**: GCC/G++, CMake, OpenCV (pkg-config `opencv4`), CUDA +
TensorRT runtime.

**依赖**：GCC/G++、CMake、OpenCV（pkg-config `opencv4`）、CUDA +
TensorRT 运行时。

Recommended build + run / 推荐编译运行：
```bash
./make.sh
```

`make.sh` will / `make.sh` 会执行：
1. Re-generate build directory (CMake) / 重新生成构建目录
2. Build `libAickTensorrt.so` / 编译动态库
3. Compile `main.cpp` into `pipeline_test` / 编译测试程序
4. Run `./pipeline_test ./test.jpg` if `test.jpg` exists / 若存在 `test.jpg` 则直接运行

# Run (manual) / 手动运行
-----------------------
```bash
g++ main.cpp -o pipeline_test \
  -I./include \
  -L./build/lib -lAickTensorrt \
  `pkg-config --cflags --libs opencv4` \
  -Wl,-rpath,./build/lib

./pipeline_test ./test.jpg [get_num]
```

# Notes / 注意
------------
- `10Xconfig.txt` must be in the **current working directory**.  
  `10Xconfig.txt` 必须在 **当前工作目录**。
- Input is resized to **1600x1600** in `main.cpp` before inference.  
  `main.cpp` 中会把输入 resize 到 **1600x1600**。
- `get_num` is optional; default is 10.  
  `get_num` 可选，默认 10。

# Configuration (10Xconfig.txt) / 配置说明
----------------------------------------
Paths / 路径:
- `10X_detection`: YOLO engine (1600x1600, multi-output)  
  `10X_detection`：YOLO 引擎（1600x1600，多输出头）
- `Impurity_score`: 2-class impurities engine  
  `Impurity_score`：杂质二分类引擎
- `Chromosome_score`: regression engine (len/disp/num)  
  `Chromosome_score`：回归打分引擎
- `Foreground_seg`: segmentation engine (skipped when `operationModel=1`)  
  `Foreground_seg`：分割引擎（`operationModel=1` 时跳过）

# Common knobs:
- `conf_thres`, `iou_thres`: YOLO thresholds  
- `impurityThreshold`: impurities class threshold  
- `lenBottom`, `dispBottom`, `numBottom`: regression thresholds  
- `meanScore_th`: minimum base score to allow bonus  
- `scoreFilter`: final score filter  
- `isSaveImg`, `saveImgPath`: whether to write cropped ROIs  

# 常用参数：
- `conf_thres`, `iou_thres`：YOLO 阈值  
- `impurityThreshold`：杂质分类阈值  
- `lenBottom`, `dispBottom`, `numBottom`：回归阈值  
- `meanScore_th`：允许加分的基础分阈值  
- `scoreFilter`：最终筛选阈值  
- `isSaveImg`, `saveImgPath`：是否保存 ROI 图像  

# Outputs / 输出
--------------
- Logs / 日志: `./Log/10X/10X_YYYYMMDD.txt`  
- ROI images / 结果图: `saveImgPath` in `10Xconfig.txt`  
- Detected points / 点结果: printed by `pipeline_test` after `GetShootingPoint(...)`

# Model I/O Details / 模型输入输出说明
----------------------------------
- YOLO (10X_detection): input `1600x1600x3`, multi-output heads.  
  Output[1] is the main head (Size ~ 945000). Output[2..4] are auxiliary heads.
- Impurities (Impurity_score): input `224x224x3`, output `2` (2-class).  
- Regression (Chromosome_score): input `224x224x3`, output `3` (len/disp/num).  
- Foreground segmentation (Foreground_seg): input `960x960x3`, output mask
  (disabled when `operationModel=1`).

- YOLO（10X_detection）：输入 `1600x1600x3`，多输出头。  
  Output[1] 为主输出头（Size ~ 945000），Output[2..4] 为辅助输出头。  
- 杂质分类（Impurity_score）：输入 `224x224x3`，输出 `2`（二分类）。  
- 回归打分（Chromosome_score）：输入 `224x224x3`，输出 `3`（长度/分散度/数量）。  
- 前景分割（Foreground_seg）：输入 `960x960x3`，输出 mask
  （当 `operationModel=1` 时跳过）。

# Public API (C++) / API 用法
---------------------------
From `include/AickTensorrt.h` / `include/AickTensorrt.h` 中的主要接口：
```cpp
DeepLearningFuncs::AickTensorrtStarter* api = AickTensorrtStarter::get_instance();
api->InitParam(params);
api->InintModel();               // loads YOLO + impurities + regression (per config)
api->AickTensorrt(image, w, h, stageInfo);
api->GetShootingPoint(points, names, get_num);
```

# Current Status / 当前进度
-------------------------
Only the 10X pipeline (YOLO → classification → regression scoring) has
been tested so far.  
目前仅验证 10X 流水线（YOLO → 分类 → 回归打分）。

## Performance / 性能表现 (Estimated)
| Stage | Input Size | Latency (FP16) |
| :--- | :--- | :--- |
| **YOLO Detection** | 1600 x 1600 | ~25ms - 35ms |
| **ROI Batch Scoring** | 224 x 224 (x10) | ~15ms |
| **Total Pipeline** | Full 10X Process | < 100ms |

# Troubleshooting / 常见问题
--------------------------
- **Pipeline failed**: check `10Xconfig.txt` paths and engine files.  
  **流水线失败**：检查 `10Xconfig.txt` 与引擎路径。
- **No output images**: set `isSaveImg=1` and ensure `saveImgPath` is writable.  
  **不出图**：确保 `isSaveImg=1` 且 `saveImgPath` 可写。
- **Scores are zero**: verify regression outputs and thresholds.  
  **得分为 0**：检查回归输出和阈值配置。

# Jetson Deployment Notes / Jetson 部署与运行注意事项
-------------------------------------------------

- Use JetPack 6.2.1 + TensorRT 10.3.0 (tested).
- Engines should be built on the same device model to avoid TRT warnings or
  performance issues.
- If CUDA OOM or `createInferRuntime` fails, reduce concurrency, keep batch B=1,
  or unload unused models.
- Ensure `./output/` exists and is writable when `isSaveImg=1`.
- Run from project root so `10Xconfig.txt` and `Log/` resolve correctly.
- Existing engines from TensorRT 8.x will NOT work. You must re-export or re-build `.engine` files using `trtexec` or the built-in initializer on the Orin Nano device.

- 建议使用 JetPack 6.2.1 + TensorRT 10.3.0（已验证）。
- Engine 最好在同型号设备上生成，避免 TRT 警告或性能异常。
- 如遇 CUDA OOM 或 `createInferRuntime` 失败，降低并发、固定 B=1，
  或只加载必要模型。
- `isSaveImg=1` 时请确保 `./output/` 可写。
- 建议在项目根目录运行，保证 `10Xconfig.txt` 与 `Log/` 路径正确。
- TRT 8.x 的引擎文件在 10.x 下不可用，必须在本地重新生成