# UrbanPulse · 数据、模式与预测实验室

城市出行数据体检与探索台：从真实原始数据得到可复核的观察，为学期项目提供数据基础。

当前版本 v0.3.1：双向置信度、Kulc、IR、四格列联表、独立计数核对、分页和完整导出，以及下一小时区域行程数预测、严格时间切分和误差回溯。已完成范围与真实结果见 [当前状态](CURRENT_STATUS.md)，后续工作见 [L3/L4 优化路线](docs/L3_L4_OPTIMIZATION_ROADMAP.md)。

## 快速运行

使用 Python 3.12。若已安装 `.venv` 并完成数据构建，可直接运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开 <http://127.0.0.1:8501>。也可以运行 `./start.ps1`。

新电脑首次安装：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m urbanpulse prepare --month 2025-01
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`requirements.lock.txt` 记录验证环境的精确版本；`requirements.txt` 是可兼容升级的依赖范围。数据下载需要联网，下载后分析可离线运行。数据、虚拟环境、课件、临时目录和派生文件不进入 Git。

## 可以做什么

- 原始/处理后切换；改变排除规则，观察均值、中位数、趋势与排名变化。
- 按上车区域、日期筛选；支持无记录窗口和空清洗规则。
- 比较工作日/周末的日均小时订单量，查看费用、距离、时长分布。
- 查看 13 项质量标记及原始记录示例，理解规则重叠。
- 查看字段类型、单位、缺失比例和文件 SHA-256。
- 导出当前筛选窗口分析报告、最多 100 条排查示例。
- 使用 8 个类别维度表示行程事务，查看频繁项集和关联规则。
- 调整支持度、置信度和提升度阈值，并回溯规则对应的原始行程。
- 按支持度、lift 或 Kulc 排序；正反规则折叠为一组，展开查看双向置信度和后件基准比例。
- 从完整事务表独立复算四格计数，核对总分母、共同计数和保存的规则计数。
- 分页浏览全部匹配关系；CSV 导出全部匹配方向规则，包含计数、指标、快照和筛选口径。
- 比较 Apriori 与 FP-Growth；结果不一致时拒绝激活快照。
- 比较昨日/上周同小时基线与监督模型，查看验证集或测试集预测曲线。
- 按区域追踪模型误差最大的目标小时，核对实际值和基线预测。

使用全月数据计算指标，不用演示假数。排查示例不是随机样本；直方图明确报告显示范围外的记录数。

## 可复现的数据管道

```powershell
# 下载并构建；已有原始文件不覆盖，校验不通过会停止
.\.venv\Scripts\python.exe -m urbanpulse prepare --month 2025-01

# 只使用本地输入重新构建
.\.venv\Scripts\python.exe -m urbanpulse build --month 2025-01

# 增加另一个月份，完成后刷新页面选择快照
.\.venv\Scripts\python.exe -m urbanpulse prepare --month 2025-02

```

可用 `--data-dir PATH` 指定数据目录；启动页面时设置同一个 `URBANPULSE_DATA_DIR` 环境变量。

```text
data/raw/<月份>/
  yellow_tripdata_<月份>.parquet   官方原始文件，不修改
  taxi_zone_lookup.csv            区域对照表
  source_manifest.json            下载地址、时间、字节数与校验值
data/processed/<月份>/
  active.json                     指向最后一次成功构建
  run-<版本>/
    urbanpulse.duckdb              全量原始字段、派生字段、逐条质量标记
    cleaned_trips.parquet          默认规则保留的行程，含稳定 record_id
    zone_hour.parquet              区域 × 小时网格，后续模型输入
    forecast_features.parquet      只使用目标小时前历史的监督学习特征
    forecast_predictions.parquet   验证/测试预测、基线与实际值
    forecast_metrics.parquet       MAE、WAPE、RMSE 及活跃区域口径
    forecast_model.joblib          训练+验证重拟合的固定模型
    forecast_report.json           切分、选参、评价、版本与泄漏策略
    trip_transactions.parquet       每条默认清洗行程的 8 项事务表示
    frequent_itemsets.parquet       Apriori 与 FP-Growth 一致的频繁项集
    association_rules.parquet       全部覆盖方向规则、四格计数、双向置信度、Kulc/IR 等
    pattern_report.json             版本、挖掘覆盖、展示默认值、算法比较与典型规则
    report.json                   统计、字典、规则、来源与版本
    report.md                     自动生成的质量检查与发现报告
```

新构建不覆盖旧版本；只有验证成功才更新 `active.json`。失败版本不激活。UI 中临时修改的规则不会覆盖批处理的默认产物。

## 开发与验证

提交前运行全套离线测试（包含 Streamlit 页面交互）和依赖检查：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

若 Windows 上已有 pytest 临时目录或缓存存在权限问题，可改用工作区内全新目录并禁用缓存；仍运行同一套测试：

```powershell
$urbanpulseTestPath = Join-Path (Get-Location) ('tmp/pytest-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp $urbanpulseTestPath
.\.venv\Scripts\python.exe -m pip check
```

版本号统一维护在 `urbanpulse/__init__.py`，页面、CLI 和构建报告复用它。关联规则生成函数保存全部覆盖方向，查询层返回全部匹配结果；折叠与分页在页面处理，CSV 导出使用完整查询结果。旧快照读取逻辑保留，以兼容其已保存的计数与覆盖边界。

## 数据合同与解释边界

- 记录单位为 TLC 记录的一次行程。`trip_count` 是观测记录数，**不是全部真实需求**。
- 英里不是公里；`fare_amount` 不是 `total_amount`；现金小费不在数据中。
- 区域、支付类型和费率是类别编码，不因整数存储就成为连续数值。
- 默认排除：时间缺失、非正时长、上车时间在目标月外、无效距离、费用缺失/非有限、上/下车区域未知。
- 负费用、零距离、超过 300 美元、超过 240 分钟、超过 100 mph、乘客数缺失只作待核查提示，不默认删除；阈值不是官方错误定义。
- 重复外观的行程可能是不同订单，没有可信唯一订单 ID 时不自动去重。
- 保留率不等于正确率；质量标记允许重叠，计数不可相加。
- 时间图只统计目标月有效上车时间；工作日定义为周一至周五，尚未区分法定节假日。
- `zone_hour` 补齐可用区域和目标月每个本地小时。零表示未观测到保留记录，不证明区域没有出行或数据源完整。
- 本版使用 TLC 本地墙上时间，未转换 UTC；夏令时回拨的重复小时会合并。后续跨月时序建模需先制定时区策略。
- 关联规则使用默认清洗视图和固定分箱；支持度不低于 1%，最大项集长度为 3。
- 新快照保存上述项集覆盖的全部方向规则；置信度 0.35、lift 1.05 是默认展示条件，可降至 0。联合支持度低于挖掘门槛的模式仍不可恢复。
- 旧快照可继续读取和补算新指标，但保留其 confidence/lift 保存下限；运行 `python -m urbanpulse build --month 2025-01` 才会生成覆盖更完整的新快照，不改旧文件。
- Kulc 是双向置信度的均值；IR 描述前后件边际计数的不平衡。相对于 A、B 均不出现的“空事务”不改变 Kulc/IR，但会影响支持度和 lift；空事务不等于数据缺失。
- 页面选中关系后从完整事务表核对计数；原始行程最多展示 100 条按 `record_id` 排序的共同出现示例。CSV 不受页面分页上限影响，正反方向各自保留。
- 关联规则只描述共同出现，不能证明因果关系或未来必然发生。
- 预测目标是下一小时已记录上车行程数，不等于全部真实需求；所有历史特征止于目标小时前一小时。
- 预测按时间切分且不打乱；验证集选择参数，测试集只在最终重拟合后评价一次。
- 验证和测试采用滚动一步评价：模型保持固定，已经发生的真实观测可进入后续小时的滞后特征。
- 当前只有一个月，测试指标用于验证实现，不足以证明跨月份泛化；不使用对大量零值不稳健的 MAPE。

## 代码导航

| 文件 | 学习任务 |
|---|---|
| `urbanpulse/config.py` | 理解字段语义、规则及默认选择 |
| `urbanpulse/download.py` | 数据来源、不可变输入与校验 |
| `urbanpulse/pipeline.py` | SQL 清洗、关联、质量标记、特征聚合 |
| `urbanpulse/analytics.py` | 描述统计、筛选分母、时间归一化 |
| `urbanpulse/patterns.py` | 带权 Apriori、FP-Growth、规则评价与事务产物 |
| `urbanpulse/forecasting.py` | 滞后/滚动特征、时间切分、基线、模型与实验评价 |
| `app.py` | 交互式探索与证据展示 |
| `tests/` | 用反例验证数据与界面行为 |
| `docs/LESSON1.md` | 第一课数据理解、清洗与统计练习 |
| `docs/LESSON2.md` | 第二课事务表示、算法与规则评价练习 |
| `docs/LESSON3.md` | 第三课监督学习、泄漏检查与误差分析练习 |

## 来源

- [NYC TLC 行程数据与区域对照表](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [Yellow Taxi 官方字段字典](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
- [DuckDB 文档](https://duckdb.org/docs/)
- [Streamlit 应用测试](https://docs.streamlit.io/develop/api-reference/app-testing)

完整学期规划见 [PRD.md](PRD.md)。项目练习编号与课程课件编号不同：`LESSON3.md` 对应监督学习，课程 L3/L4 对应模式挖掘。
