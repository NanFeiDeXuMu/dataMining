# UrbanPulse · 数据与关联模式实验室

城市出行数据体检与探索台：从真实原始数据得到可复核的观察，为学期项目提供数据基础。

v0.2 在数据理解、质量检查与描述统计上新增事务特征、Apriori、FP-Growth 和关联规则；不训练预测模型，不接 LLM，不发布到外部网站。

## 快速运行

推荐 Python 3.12。当前工作区已安装独立 `.venv`，并已准备 2025-01 真实数据，直接运行：

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

`requirements.lock.txt` 记录本次验证环境的精确版本；`requirements.txt` 是可兼容升级的依赖范围。数据下载需要联网，下载后分析可离线运行。数据、虚拟环境和派生文件不进入 Git。

## 可以做什么

- 原始/处理后切换；改变排除规则，观察均值、中位数、趋势与排名变化。
- 按上车区域、日期筛选；支持无记录窗口和空清洗规则。
- 比较工作日/周末的日均小时订单量，查看费用、距离、时长分布。
- 查看 13 项质量标记及原始记录示例，理解规则重叠。
- 查看字段类型、单位、缺失比例和文件 SHA-256。
- 导出当前筛选窗口分析报告、最多 100 条排查示例。
- 使用 8 个类别维度表示行程事务，查看频繁项集和关联规则。
- 调整支持度、置信度和提升度阈值，并回溯规则对应的原始行程。
- 比较 Apriori 与 FP-Growth；结果不一致时拒绝激活快照。

使用全月数据计算指标，不用演示假数。排查示例不是随机样本；直方图明确报告显示范围外的记录数。

## 可复现的数据管道

```powershell
# 下载并构建；已有原始文件不覆盖，校验不通过会停止
.\.venv\Scripts\python.exe -m urbanpulse prepare --month 2025-01

# 只使用本地输入重新构建
.\.venv\Scripts\python.exe -m urbanpulse build --month 2025-01

# 增加另一个月份，完成后刷新页面选择快照
.\.venv\Scripts\python.exe -m urbanpulse prepare --month 2025-02

# 离线测试，包括 Streamlit 页面交互测试
.\.venv\Scripts\python.exe -m pytest -q
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
    trip_transactions.parquet       每条默认清洗行程的 8 项事务表示
    frequent_itemsets.parquet       Apriori 与 FP-Growth 一致的频繁项集
    association_rules.parquet       带分子、分母和三项评价指标的规则
    pattern_report.json             分箱、阈值、算法比较与典型规则
    report.json                   统计、字典、规则、来源与版本
    report.md                     自动生成的质量检查与发现报告
```

新构建不覆盖旧版本；只有验证成功才更新 `active.json`。失败版本不激活。UI 中临时修改的规则不会覆盖批处理的默认产物。

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
- 关联规则只描述共同出现，不能证明因果关系或未来必然发生。

## 代码导航

| 文件 | 学习任务 |
|---|---|
| `urbanpulse/config.py` | 理解字段语义、规则及默认选择 |
| `urbanpulse/download.py` | 数据来源、不可变输入与校验 |
| `urbanpulse/pipeline.py` | SQL 清洗、关联、质量标记、特征聚合 |
| `urbanpulse/analytics.py` | 描述统计、筛选分母、时间归一化 |
| `urbanpulse/patterns.py` | 带权 Apriori、FP-Growth、规则评价与事务产物 |
| `app.py` | 交互式探索与证据展示 |
| `tests/` | 用反例验证数据与界面行为 |
| `docs/LESSON1.md` | 第一课数据理解、清洗与统计练习 |
| `docs/LESSON2.md` | 第二课事务表示、算法与规则评价练习 |

## 来源

- [NYC TLC 行程数据与区域对照表](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [Yellow Taxi 官方字段字典](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
- [DuckDB 文档](https://duckdb.org/docs/)
- [Streamlit 应用测试](https://docs.streamlit.io/develop/api-reference/app-testing)

完整学期规划见 [PRD.md](PRD.md)。当前派生表和规则结果是后续分类、聚类和异常检测的共同入口。
