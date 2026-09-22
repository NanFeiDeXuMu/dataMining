"""Batch pipeline: immutable raw data -> annotated records -> reusable features.

DuckDB does the large scans. The browser receives only aggregates and small previews.
Each successful build is a new snapshot; activation happens only after verification.
"""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import json
import secrets

import duckdb

from . import __version__
from .analytics import Explorer, insights
from .config import DEFAULT_RULES, FIELD_INFO, REQUIRED_COLUMNS, RULES, keep_expression, month_bounds
from .download import sha256, write_json
from .forecasting import build_forecast_assets
from .patterns import build_pattern_assets


def literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def create_run_dir(base: Path) -> Path:
    """Use inherited ACLs; tempfile.mkdtemp(mode=0700) isolates Windows users."""
    for _ in range(100):
        run = base / f"run-{secrets.token_hex(4)}"
        try:
            run.mkdir()
            return run
        except FileExistsError:
            continue
    raise FileExistsError("无法创建唯一的构建快照目录。")


def build(month: str, root: Path) -> Path:
    start, end = month_bounds(month)
    source = root / "raw" / month
    trips = source / f"yellow_tripdata_{month}.parquet"
    lookup = source / "taxi_zone_lookup.csv"
    manifest_file = source / "source_manifest.json"
    if not all(path.exists() for path in (trips, lookup, manifest_file)):
        raise FileNotFoundError(f"缺少原始数据或来源清单。先运行 python -m urbanpulse download --month {month}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("month") != month:
        raise ValueError("来源清单月份与请求月份不一致。")
    entries = {entry["file"]: entry for entry in manifest["files"]}
    for path in (trips, lookup):
        if path.name not in entries or sha256(path) != entries[path.name]["sha256"]:
            raise ValueError(f"原始数据校验失败: {path.name}；拒绝使用被修改的快照。")
    base = root / "processed" / month
    base.mkdir(parents=True, exist_ok=True)
    run = create_run_dir(base)
    db = run / "urbanpulse.duckdb"
    print("校验来源后开始构建；所有清洗只影响派生视图，不修改原始记录。", flush=True)
    con = duckdb.connect(str(db))
    try:
        con.execute("SET threads = 4")
        con.execute("SET memory_limit = '2GB'")
        con.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet({literal(trips)}, file_row_number=true)")
        schema = con.execute("DESCRIBE source").fetchall()
        names = [row[0] for row in schema if row[0] != "file_row_number"]
        missing = set(REQUIRED_COLUMNS) - set(names)
        if missing:
            raise ValueError(f"原始数据缺少必要字段: {sorted(missing)}")
        derived = {"record_id", "pickup_time", "dropoff_time", "pickup_zone_id", "dropoff_zone_id",
                   "pickup_zone", "dropoff_zone", "pickup_borough", "dropoff_borough", "distance_miles", "meter_fare",
                   "amount", "passengers", "duration_min", "speed_mph", "day_type"}
        if derived.intersection(names) or any(name.startswith("q_") for name in names):
            raise ValueError("输入与派生字段重名，请使用原始 TLC 行程文件。")
        checks = ",".join(f"count(*) FILTER (WHERE {identifier(name)} IS NULL)" for name in names)
        counts = con.execute(f"SELECT count(*), {checks} FROM source").fetchone()
        if not counts[0]:
            raise ValueError("原始文件没有记录。")
        types = {row[0]: row[1] for row in schema}
        profile = [
            {"字段": name, "存储类型": types[name],
             "属性类型": FIELD_INFO.get(name, ("其他数值/字段", "参见官方字典"))[0],
             "说明": FIELD_INFO.get(name, ("", "参见官方字典"))[1],
             "缺失数": counts[i + 1], "缺失比例": counts[i + 1] / counts[0]}
            for i, name in enumerate(names)
        ]
        con.execute(f"""CREATE TABLE zones AS SELECT
            TRY_CAST(LocationID AS INTEGER) AS zone_id, Zone AS zone, Borough AS borough,
            coalesce(TRY_CAST(LocationID AS INTEGER) BETWEEN 1 AND 263
                AND Zone NOT IN ('Unknown', 'N/A'), false) AS usable
            FROM read_csv_auto({literal(lookup)}, header=true)""")
        duplicate_zones = con.execute("SELECT zone_id FROM zones GROUP BY zone_id HAVING count(*) > 1").fetchall()
        if duplicate_zones or con.execute("SELECT count(*) FROM zones WHERE zone_id IS NULL").fetchone()[0]:
            raise ValueError("区域对照表包含重复或无效 ID，拒绝造成关联膨胀。")
        passengers = 'TRY_CAST(r.passenger_count AS DOUBLE)' if "passenger_count" in names else "NULL::DOUBLE"
        con.execute(f"""
            CREATE TABLE annotated AS
            WITH typed AS (
                SELECT r.* EXCLUDE (file_row_number), file_row_number AS record_id,
                    TRY_CAST(tpep_pickup_datetime AS TIMESTAMP) AS pickup_time,
                    TRY_CAST(tpep_dropoff_datetime AS TIMESTAMP) AS dropoff_time,
                    CASE WHEN TRY_CAST(PULocationID AS DOUBLE) = TRY_CAST(PULocationID AS INTEGER)
                        THEN TRY_CAST(PULocationID AS INTEGER) END AS pickup_zone_id,
                    CASE WHEN TRY_CAST(DOLocationID AS DOUBLE) = TRY_CAST(DOLocationID AS INTEGER)
                        THEN TRY_CAST(DOLocationID AS INTEGER) END AS dropoff_zone_id,
                    TRY_CAST(trip_distance AS DOUBLE) AS distance_miles,
                    TRY_CAST(total_amount AS DOUBLE) AS amount,
                    TRY_CAST(fare_amount AS DOUBLE) AS meter_fare, {passengers} AS passengers
                FROM source r
            ), enriched AS (
                SELECT t.*, z.zone AS pickup_zone, z.borough AS pickup_borough,
                    d.zone AS dropoff_zone, d.borough AS dropoff_borough,
                    NOT coalesce(z.usable, false) AS q_unknown_pickup,
                    NOT coalesce(d.usable, false) AS q_unknown_dropoff,
                    date_diff('microsecond', pickup_time, dropoff_time) / 60000000.0 AS duration_min,
                    CASE WHEN pickup_time IS NULL THEN NULL
                        WHEN isodow(pickup_time) IN (6, 7) THEN '周末' ELSE '工作日' END AS day_type
                FROM typed t LEFT JOIN zones z ON t.pickup_zone_id = z.zone_id
                LEFT JOIN zones d ON t.dropoff_zone_id = d.zone_id
            ), measured AS (
                SELECT *, CASE WHEN duration_min > 0 AND isfinite(distance_miles)
                    THEN distance_miles / duration_min * 60 END AS speed_mph FROM enriched
            )
            SELECT *,
                pickup_time IS NULL OR dropoff_time IS NULL AS q_missing_time,
                coalesce(duration_min <= 0, false) AS q_invalid_duration,
                coalesce(pickup_time < DATE {literal(start)} OR pickup_time >= DATE {literal(end)}, false) AS q_outside_month,
                distance_miles IS NULL OR NOT isfinite(distance_miles) OR coalesce(distance_miles < 0, false) AS q_invalid_distance,
                amount IS NULL OR NOT isfinite(amount) OR meter_fare IS NULL OR NOT isfinite(meter_fare) AS q_missing_amount,
                coalesce(distance_miles = 0, false) AS q_zero_distance,
                coalesce(amount < 0 OR meter_fare < 0, false) AS q_negative_amount,
                coalesce(isfinite(amount) AND amount > 300, false) AS q_high_amount,
                coalesce(duration_min > 240, false) AS q_long_duration,
                coalesce(speed_mph > 100, false) AS q_high_speed,
                passengers IS NULL OR NOT isfinite(passengers) AS q_missing_passengers
            FROM measured
        """)
        count = con.execute("SELECT count(*) FROM annotated").fetchone()[0]
        if count != counts[0]:
            raise AssertionError("关联改变了记录数。")
        con.execute(f"CREATE VIEW cleaned AS SELECT * FROM annotated WHERE {keep_expression(DEFAULT_RULES)}")
        con.execute(f"COPY cleaned TO {literal(run / 'cleaned_trips.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
        print(f"已标记 {count:,} 条记录，正在生成区域 × 小时特征。", flush=True)
        con.execute(f"""
            CREATE TABLE zone_hour AS
            WITH counts AS (
                SELECT pickup_zone_id AS zone_id, date_trunc('hour', pickup_time) AS hour_start,
                    count(*) AS trip_count, avg(amount) AS mean_total_amount,
                    median(distance_miles) AS median_distance_miles
                FROM cleaned GROUP BY 1, 2
            ), grid AS (
                SELECT z.zone_id, z.zone, z.borough, h.hour_start FROM zones z
                CROSS JOIN range(TIMESTAMP {literal(start)}, TIMESTAMP {literal(end)}, INTERVAL 1 HOUR) h(hour_start)
                WHERE z.usable
            )
            SELECT g.*, coalesce(c.trip_count, 0)::BIGINT AS trip_count,
                coalesce(c.trip_count, 0) > 0 AS has_observed_trip,
                c.mean_total_amount, c.median_distance_miles
            FROM grid g LEFT JOIN counts c USING (zone_id, hour_start)
            ORDER BY g.zone_id, g.hour_start
        """)
        con.execute(f"COPY zone_hour TO {literal(run / 'zone_hour.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
        clean_n = con.execute("SELECT count(*) FROM cleaned").fetchone()[0]
        grid_n, trip_sum = con.execute("SELECT count(*), coalesce(sum(trip_count), 0) FROM zone_hour").fetchone()
        if trip_sum != clean_n:
            raise AssertionError("区域-小时汇总与清洗后总数不守恒。")
        print("区域-小时特征完成；正在训练下一小时预测模型。", flush=True)
        forecast_report = build_forecast_assets(con, run, month)
        print("预测产物完成；正在挖掘 Apriori 与 FP-Growth 频繁模式。", flush=True)
        pattern_report = build_pattern_assets(con, run, names)
        if pattern_report["transaction_count"] != clean_n:
            raise AssertionError("行程事务数与清洗后总数不守恒。")
        con.execute("DROP VIEW source")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    with Explorer(db, month) as explorer:
        raw = explorer.stats()
        clean = explorer.stats(DEFAULT_RULES)
        quality = explorer.quality()
        observations = insights(raw, clean, explorer.hourly(DEFAULT_RULES), month)
    metadata = {
        "month": month, "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": manifest, "policy": list(DEFAULT_RULES),
        "rules": [{"key": r.key, "label": r.label, "reason": r.reason, "default": r.default} for r in RULES],
        "raw": raw, "clean": clean, "zone_hour_rows": grid_n,
        "forecast": forecast_report, "patterns": pattern_report,
        "quality": quality.to_dict(orient="records"), "schema": profile, "insights": observations,
        "versions": {"urbanpulse": __version__, **{package: version(package) for package in
                     ("duckdb", "pandas", "pyarrow", "scikit-learn")}},
        "limitations": [
            "数据为 TLC 已记录行程，不代表全部真实需求或未满足需求。",
            "可疑不等于错误或欺诈；负费用和极端值默认保留，重复外观记录不自动去重。",
            "时间图只使用目标月有效上车时间；均值/分位数只使用有限数值，并报告分母。",
            "原始时间按 TLC 本地墙上时间保留；夏令时回拨小时会合并，不是 UTC 时间轴。",
            "小时网格 0 表示该文件中未观测到保留记录，不是确认区域无出行或数据源完整。",
            "不含现金小费，不能据此比较不同支付方式的全部小费。",
            "全文件保留率和筛选窗口保留率可能不同，问题规则之间允许重叠。",
            "关联规则描述共同出现，不证明因果关系；结果依赖默认清洗与固定分箱。",
            "预测目标是已记录的下一小时上车行程数，不等于未满足需求；当前单月评价只用于开发验证。",
        ],
    }
    write_json(run / "report.json", metadata)
    write_json(run / "pattern_report.json", pattern_report)
    forecast_improvement = forecast_report["test_wape_improvement_vs_lag_168"]
    forecast_improvement_line = (
        f"- 测试集相对 lag-168 的 WAPE 改善：{forecast_improvement:.2%}。"
        if forecast_improvement is not None else
        "- 测试集实际行程总数为零，WAPE 及相对改善不可计算。"
    )
    report = [f"# UrbanPulse 数据体检报告 | {month}", "", "## 三条可复核观察", ""]
    report.extend(f"- {line}" for line in observations)
    report.extend(["", "## 质量检查（规则重叠，不可相加）", "", "| 检查项 | 记录数 | 占全文件比例 | 默认排除 |", "|---|---:|---:|---|"])
    for row in metadata["quality"]:
        report.append(f"| {row['检查项']} | {row['记录数']:,} | {row['占原始窗口比例']:.2%} | {'是' if row['默认排除'] else '否'} |")
    report.extend(["", "## 解释边界", ""] + [f"- {x}" for x in metadata["limitations"]])
    report.extend(["", "## 关联规则", "",
                   f"- 事务：{pattern_report['transaction_count']:,}；聚合篮子：{pattern_report['unique_baskets']:,}",
                   f"- 频繁项集：{pattern_report['frequent_itemsets']:,}；完整方向规则：{pattern_report['rules']:,}；默认展示阈值匹配：{pattern_report['default_display_rules']:,}",
                   "- 保存已挖掘项集的全部方向规则；支持度与最大长度限定覆盖范围，置信度/lift 为展示筛选。",
                   "- 规则附带 N、前后件计数、四格列联表、双向置信度、Kulc 和 IR；页面从事务表独立复算证据。",
                   "- Apriori 与 FP-Growth 已验证得到完全相同的频繁项集；关联不代表因果。",
                   "", "## 下一小时预测", "",
                   f"- 严格时间切分：训练 {forecast_report['split_rows']['train']:,}；验证 {forecast_report['split_rows']['validation']:,}；测试 {forecast_report['split_rows']['test']:,} 个区域小时。",
                   f"- 模型：{forecast_report['model']['name']}（Poisson 损失）；与 lag-24、lag-168 在相同测试行上比较。",
                   forecast_improvement_line,
                   "- 任何滚动或滞后特征都止于目标小时前一小时；单月结果不代表长期泛化。",
                   "", "## 可复现信息", "", f"- 默认清洗规则：`{', '.join(DEFAULT_RULES)}`", f"- 区域-小时特征行数：{grid_n:,}", "- 原始文件未修改；全部原始字段、稳定记录号和检查标记存于数据库 annotated 表。"])
    for entry in manifest["files"]:
        report.append(f"- {entry['file']}：SHA-256 `{entry['sha256']}`；来源 {entry.get('url', '本地测试夹具')}")
    (run / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(base / "active.json", {"run": run.name})
    print(f"构建完成：原始 {count:,}；默认规则保留 {clean_n:,}；特征 {grid_n:,} 行；"
          f"预测测试 {forecast_report['split_rows']['test']:,} 行；关联规则 {pattern_report['rules']:,} 条。\n输出：{run}", flush=True)
    return run
