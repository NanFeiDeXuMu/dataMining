# Gesture Lab：18 类实时手势识别

课程 Mini Project 的完整本地实现：MediaPipe 提取 21 个关键点，PyTorch MLP 预测 18 类手势，OpenCV 显示摄像头、手部骨架、标签和分数。训练、验证、测试与模型激活均有可回溯产物。

当前真实结果和硬件验收状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。准备阶段历史见 [PREPARATION_STATUS.md](PREPARATION_STATUS.md)。

## 图片演示

当前电脑暂未连接摄像头，先用本地图片演示。以下命令在 `miniProject` 目录执行：

```powershell
.\.venv\Scripts\python.exe -m gesture_project.demo --image gesture_9000\like\train-00005_row15910.jpg --output reports\my-demo.png
```

输出路径必须尚不存在。命令打印预测结果并保存带关键点和信息面板的图片；不会修改输入图片。已生成的完整示例集在 `reports/image-demo-final/`，包含 18 类示例、低分提示和检测失败案例。

重新生成整组图片演示：

```powershell
.\.venv\Scripts\python.exe -m gesture_project.gallery
```

每类选择按路径排序的第一张可用测试图片，不按预测正确与否或分数挑选；额外附两类检测失败案例。汇总图是候选标签与分数概览，完整界面在低于阈值时显示 `Uncertain`。输出目录自动使用新时间戳，也可用 `--output-dir` 指定尚不存在的目录。

## 摄像头启动

连接摄像头后，从仓库根目录执行：

```powershell
.\miniProject\launch_demo.ps1
```

也可以双击 `launch_demo.cmd`。摄像头打开后，`Q` / `Esc` 退出，`M` 切换镜像显示，`S` 保存截图到 `reports/screenshots/`。镜像只改变显示，模型输入保持原来的方向；文字不会反转。

无手时立即清空旧预测，多只手时提示只展示一只手，置信分数低于 0.60 时显示 `Uncertain`。连续有效帧使用 5 帧均值平滑；手消失、位置突变、左右手切换或时间间隔过长时清空平滑历史。

摄像头读取失败会说明原因。检查摄像头连接、其他程序占用以及 Windows「隐私和安全性 → 摄像头 → 允许桌面应用访问摄像头」。可尝试：

```powershell
.\miniProject\launch_demo.ps1 --camera 1
.\miniProject\launch_demo.ps1 --backend msmf
```

## 环境与复现

使用 Python 3.12 和独立虚拟环境。已验证的版本写入 `requirements.lock.txt`，首次重建优先使用锁定文件：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt --index-url https://pypi.org/simple --no-cache-dir
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` 提供允许的版本范围；锁定文件记录实际安装版本。本机默认 pip 缓存写入曾造成长时间停顿，使用 `--no-cache-dir` 可避开；不修改全局配置。程序的 Matplotlib 缓存也放在本目录 `data/` 内。

当前 PyTorch 为 CPU build。小型 MLP 在 CPU 上完成正式训练；GPU 存在不代表本环境已启用 CUDA。

## 完整构建

```powershell
.\.venv\Scripts\python.exe -m gesture_project.build
```

流程如下：

1. 审计全部图片与来源清单，检查解码、标签、字节数、重复和缺失；原始资源不改写。
2. 每类先划分原始图片：训练 350、验证 75、测试 75，种子 42。全部共 6,300 / 1,350 / 1,350 张。
3. 提取关键点，并记录无手、多手歧义和无效几何。失败样本保留原始 split，评价分母不随筛选变化。
4. 只使用训练集拟合标准化器。训练 63 → 128 → 64 → 18 的 MLP，用验证 loss 选择模型和早停；另训练固定参数逻辑回归基线。
5. 冻结模型选择后，在测试集评价两种模型，生成逐图预测、逐类指标、loss 曲线、混淆矩阵和实验报告。
6. 校验产物哈希、分区对齐、完整分母，独立重算指标并加载模型复核保存预测。全部成功后原子更新 `data/processed/active.json`。

所有派生结果追加到新的 `data/processed/<YYYY-MM>/run-*/`。构建失败不会改变当前有效模型；原始图片、历史运行和准备阶段冒烟模型保留。

可以分阶段运行：

```powershell
.\.venv\Scripts\python.exe -m gesture_project.build --extract-only
.\.venv\Scripts\python.exe -m gesture_project.train --run-dir data\processed\<YYYY-MM>\<run-directory>
.\.venv\Scripts\python.exe -m gesture_project.validate --run-dir data\processed\<YYYY-MM>\<run-directory>
```

把 `<...>` 替换为构建输出的真实路径。已经开始训练的运行不会重复训练或覆盖测试结果；新实验创建新运行。纯验证会重放既有模型和预测证据，不做选参或再次训练。完整已训练产物可用 `validate --activate` 在核验后恢复激活。

## 指标含义

| 指标 | 分母 |
|---|---|
| 关键点可用率 | 对应分区全部原始图片 |
| 可用样本分类准确率 | 对应分区中成功提取特征的图片 |
| 端到端正确率 | 对应分区全部原始图片；检测失败计为错误 |
| Macro-F1 / 逐类 precision、recall | 成功提取特征的图片，18 类等权平均 |
| 拒识后的准确率 | 分数达到阈值的预测 |
| 拒识后的覆盖率 | 对应分区全部原始图片 |

主分类指标采用最高分标签，不应用演示的置信阈值和时间平滑。拒识指标另行报告。softmax 分数没有校准，不是可靠的正确概率。

## 核心产物

| 文件 | 用途 |
|---|---|
| `data_audit.json`、`image_inventory.csv` | 原始资源审计及逐图哈希 |
| `split_manifest.csv`、`detections.csv` | 全部原始图片的分区、标签、检测状态及布尔质量标记 |
| `features.npz`、`extraction.json` | 特征、布尔可用掩码、分区和提取统计 |
| `classifier.pt`、`baseline.joblib` | 正式 MLP 与逻辑回归基线，包含训练集标准化信息 |
| `hand_landmarker.task` | 本次运行使用的预训练模型副本 |
| `training_history.csv`、`selection.json` | 逐轮训练记录及验证集模型选择依据 |
| `evaluation.json`、`predictions.csv`、`per_class_test.csv` | 完整指标和逐图预测证据 |
| `training_curves.png`、`confusion_matrix.png`、`REPORT.md` | 可展示的曲线、混淆矩阵与中文实验报告 |
| `config.json`、`source/`、`artifacts.json` | 参数、训练时源码/依赖快照及产物哈希 |
| `validation.json`、`status.json` | 独立核验结果与运行状态 |

加载模型前会核对核心产物哈希，拒绝准备阶段的 `smoke_test_only` 模型、错误标签映射或不兼容的特征版本。

## 代码入口

- `gesture_project/config.py`：默认检测及训练配置。
- `audit.py`、`dataset.py`：输入审计、原始分区和全量关键点提取。
- `features.py`：训练与演示共用的方向保留特征处理。
- `network.py`、`predictor.py`、`train.py`：分类器、标准化、训练、保存和推理。
- `metrics.py`、`validate.py`：明确分母的评价、证据重算及安全激活。
- `detector.py`、`presentation.py`、`demo.py`：图像/视频检测、平滑、界面和摄像头资源管理。
- `gallery.py`：实际运行模型生成全类别图片演示和失败案例。
- `prepare.py`、`smoke.py`：保留的全量审计和小样本工程验证入口。

## 测试与演示验收

```powershell
$gestureTestPath = Join-Path $PWD ("data\test-tmp-" + [guid]::NewGuid().ToString("N"))
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp $gestureTestPath
.\.venv\Scripts\python.exe -m pip check
```

测试覆盖数据完整性、无泄漏分区、特征几何、训练标准化、明确评价分母、最佳验证 checkpoint、模型加载限制、平滑重置、异常时释放摄像头，以及失败验证不影响 active.json。

摄像头诊断不打开窗口、不保存画面，且必须明确帧数上限：

```powershell
.\.venv\Scripts\python.exe -m gesture_project.demo --headless --max-frames 20
```

现场验收应逐类演示 18 个手势，并测试手移出画面、出现两只手、换手、光照、镜像、保存截图和退出。诊断能读取帧不等于手势准确率已经人工验收。

## 数据和能力边界

18 类：`call`、`dislike`、`fist`、`four`、`like`、`mute`、`ok`、`one`、`palm`、`peace`、`peace_inverted`、`rock`、`stop`、`stop_inverted`、`three`、`three2`、`two_up`、`two_up_inverted`。

清单没有人员 ID，因此不能声称跨人员独立评价；当前审计没有识别近似重复图像。手部关键点不能确认 `mute` 是否真的位于嘴唇前。当前模型只训练上述 18 类，置信阈值不能保证识别所有未知动作。

原始图片、模型、课件、虚拟环境、缓存、日志和派生产物不提交 Git。运行依赖本机已下载的 `gesture_9000/` 和 `models/hand_landmarker.task`；原 UrbanPulse 项目环境与业务代码独立保留。
