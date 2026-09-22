# UrbanPulse 协作指南

## 目标

UrbanPulse 是可复现、可解释的城市出行数据挖掘课程项目。所有结论必须来自真实产物，并能回溯到数据、SQL、算法配置或测试。

## 开发约定

- 使用 Python 3.12、DuckDB、Parquet 和 Streamlit；优先复用现有模块，避免无必要依赖。
- `data/raw` 是不可变输入；派生结果写入新的 `data/processed/<month>/run-*`，验证成功后才能更新 `active.json`。
- 质量标记保留为布尔字段；默认清洗规则集中维护在 `urbanpulse/config.py`。
- 指标须明确分母和筛选范围；关联规则仅表示共同出现，不解释为因果关系。
- 新功能同时补充测试和对应文档。提交前运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

- 不提交 `.venv`、`data/raw`、`data/processed`、缓存、日志和课程课件。

## 代码入口

- `urbanpulse/pipeline.py`：数据合同、清洗、质量报告和产物构建
- `urbanpulse/patterns.py`：事务表示、Apriori、FP-Growth 和关联规则
- `urbanpulse/forecasting.py`：严格过去式特征、时间切分、基线、模型与评价
- `urbanpulse/analytics.py`：面向页面的参数化 DuckDB 查询
- `app.py`：交互展示和证据回溯
- `tests/`：算法、管道和页面行为验收

开始工作前阅读 `PRD.md`、`DECISIONS.md` 和 `CURRENT_STATUS.md`。
