"""Small, parameterized SQL queries shared by the app, reports and tests."""

from datetime import date, timedelta
from pathlib import Path
import json
from math import isfinite

import duckdb
import pandas as pd

from .config import RULES, keep_expression, month_bounds
from .patterns import evaluated_rule_frame, rule_coverage, rule_metrics


def active_run(root: Path, month: str) -> Path:
    month_bounds(month)
    base = (root / "processed" / month).resolve()
    pointer = json.loads((base / "active.json").read_text(encoding="utf-8"))
    run = (base / pointer["run"]).resolve()
    if run.parent != base or not run.is_dir():
        raise ValueError("无效的数据版本指针。")
    return run


def available_months(root: Path) -> list[str]:
    return sorted((p.parent.name for p in (root / "processed").glob("*/active.json")), reverse=True)


class Explorer:
    def __init__(self, db: Path, month: str):
        self.run = Path(db).parent
        self.con = duckdb.connect(str(db), read_only=True)
        self.month = month
        self.start, self.end = month_bounds(month)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.con.close()

    @staticmethod
    def scope(start: date | None = None, end: date | None = None, zone: int | None = None):
        terms, values = ["TRUE"], []
        if start is not None:
            terms.append("pickup_time >= ?")
            values.append(start)
        if end is not None:
            terms.append("pickup_time < ?")
            values.append(end + timedelta(days=1))
        if zone is not None:
            terms.append("pickup_zone_id = ?")
            values.append(int(zone))
        return " AND ".join(terms), values

    def query(self, sql: str, params=None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).fetchdf()

    def stats(self, rules=(), start=None, end=None, zone=None) -> dict:
        scope, values = self.scope(start, end, zone)
        cursor = self.con.execute(f"""
            SELECT count(*) AS records,
                count(*) FILTER (WHERE isfinite(amount)) AS amount_n,
                avg(amount) FILTER (WHERE isfinite(amount)) AS mean_amount,
                median(amount) FILTER (WHERE isfinite(amount)) AS median_amount,
                quantile_cont(amount, 0.95) FILTER (WHERE isfinite(amount)) AS p95_amount,
                avg(distance_miles) FILTER (WHERE isfinite(distance_miles)) AS mean_distance,
                median(duration_min) AS median_duration,
                count(*) FILTER (WHERE q_missing_time) AS missing_time_n,
                count(*) FILTER (WHERE q_outside_month) AS outside_month_n,
                count(*) FILTER (WHERE q_negative_amount) AS negative_amount_n
            FROM annotated WHERE {scope} AND {keep_expression(rules)}
        """, values)
        return dict(zip([d[0] for d in cursor.description], cursor.fetchone()))

    def quality(self, start=None, end=None, zone=None) -> pd.DataFrame:
        scope, values = self.scope(start, end, zone)
        fields = ", ".join(f"count(*) FILTER (WHERE q_{r.key})" for r in RULES)
        counts = self.con.execute(f"SELECT {fields}, count(*) FROM annotated WHERE {scope}", values).fetchone()
        n = counts[-1]
        return pd.DataFrame([
            {"key": rule.key, "检查项": rule.label, "记录数": counts[i],
             "占原始窗口比例": counts[i] / n if n else 0.0,
             "默认排除": rule.default, "说明": rule.reason}
            for i, rule in enumerate(RULES)
        ])

    def zones(self) -> pd.DataFrame:
        return self.query("SELECT zone_id, zone, borough FROM zones ORDER BY zone_id")

    def calendar_window(self, start=None, end=None):
        left = max(start or self.start, self.start)
        right = min(end or (self.end - timedelta(days=1)), self.end - timedelta(days=1))
        return left, right

    def hourly(self, rules=(), start=None, end=None, zone=None) -> pd.DataFrame:
        left, right = self.calendar_window(start, end)
        scope, values = self.scope(left, right, zone)
        counts = self.query(f"""
            SELECT hour(pickup_time)::INTEGER AS hour, day_type, count(*) AS trips
            FROM annotated WHERE {scope} AND {keep_expression(rules)}
            GROUP BY hour, day_type
        """, values)
        days = {"工作日": 0, "周末": 0}
        for d in pd.date_range(left, right):
            days["周末" if d.weekday() >= 5 else "工作日"] += 1
        frame = pd.DataFrame([{"hour": h, "day_type": kind, "calendar_days": n}
                              for kind, n in days.items() for h in range(24)])
        frame = frame.merge(counts, how="left", on=["hour", "day_type"])
        frame["trips"] = frame["trips"].fillna(0).astype("int64")
        frame["daily_average"] = frame["trips"] / frame["calendar_days"].replace(0, float("nan"))
        return frame

    def daily(self, rules=(), start=None, end=None, zone=None) -> pd.DataFrame:
        left, right = self.calendar_window(start, end)
        scope, values = self.scope(left, right, zone)
        frame = self.query(f"""
            SELECT CAST(pickup_time AS DATE) AS date, count(*) AS trips
            FROM annotated WHERE {scope} AND {keep_expression(rules)} GROUP BY date ORDER BY date
        """, values)
        calendar = pd.DataFrame({"date": pd.date_range(left, right)})
        frame = calendar.merge(frame, on="date", how="left")
        frame["trips"] = frame["trips"].fillna(0).astype("int64")
        return frame

    def ranking(self, rules=(), start=None, end=None, zone=None) -> pd.DataFrame:
        scope, values = self.scope(start, end, zone)
        return self.query(f"""
            SELECT pickup_zone_id AS zone_id, coalesce(pickup_zone, '未匹配/缺失') AS zone,
                coalesce(pickup_borough, '未知') AS borough, count(*) AS trips
            FROM annotated WHERE {scope} AND {keep_expression(rules)}
            GROUP BY 1, 2, 3 ORDER BY trips DESC, zone_id NULLS LAST LIMIT 15
        """, values)

    def histogram(self, field="amount", rules=(), start=None, end=None, zone=None):
        options = {"amount": (-20, 200, 5), "distance_miles": (0, 50, 1), "duration_min": (0, 120, 5)}
        if field not in options:
            raise ValueError("不支持的分布字段")
        lo, hi, width = options[field]
        scope, values = self.scope(start, end, zone)
        base = f"FROM annotated WHERE {scope} AND {keep_expression(rules)}"
        data = self.query(f"""
            SELECT floor(({field} - {lo}) / {width}) * {width} + {lo} AS bin_start, count(*) AS trips
            {base} AND isfinite({field}) AND {field} >= {lo} AND {field} < {hi}
            GROUP BY 1 ORDER BY 1
        """, values)
        tails = self.con.execute(f"""SELECT
            count(*) FILTER (WHERE isfinite({field}) AND {field} < {lo}),
            count(*) FILTER (WHERE isfinite({field}) AND {field} >= {hi}),
            count(*) FILTER (WHERE {field} IS NULL OR NOT isfinite({field})) {base}""", values).fetchone()
        return data, {"low": lo, "high": hi, "width": width, "below": tails[0], "above": tails[1], "missing": tails[2]}

    def sample(self, rules=(), start=None, end=None, zone=None, issue=None, limit=100):
        scope, values = self.scope(start, end, zone)
        issue_filter = "TRUE"
        if issue:
            keep_expression([issue])  # whitelist before interpolating
            issue_filter = f"q_{issue}"
        reasons = ",".join(f"CASE WHEN q_{r.key} THEN '{r.label}' END" for r in RULES)
        return self.query(f"""
            SELECT record_id, pickup_time, dropoff_time, pickup_zone, dropoff_zone,
                distance_miles, duration_min, amount,
                concat_ws('；', {reasons}) AS quality_notes
            FROM annotated WHERE {scope} AND {keep_expression(rules)} AND {issue_filter}
            ORDER BY record_id LIMIT ?
        """, values + [int(limit)])

    def pattern_metadata(self):
        path = self.run / "pattern_report.json"
        if not path.is_file():
            raise ValueError("当前快照缺少模式元数据，请重新构建。")
        return json.loads(path.read_text(encoding="utf-8"))

    def pattern_rules(self, min_support=None, min_confidence=None, min_lift=None,
                      required_items=(), sort_by="lift"):
        """Return every matching direction; the page owns folding and pagination."""
        metadata = self.pattern_metadata()
        cfg, coverage = metadata["config"], rule_coverage(metadata)
        min_support = cfg["min_support"] if min_support is None else float(min_support)
        min_confidence = cfg["min_confidence"] if min_confidence is None else float(min_confidence)
        min_lift = cfg["min_lift"] if min_lift is None else float(min_lift)
        if not coverage["min_support"] <= min_support <= 1:
            raise ValueError(f"支持度不能低于快照挖掘阈值 {coverage['min_support']:.2%}。")
        required_items = tuple(item for item in required_items if item)
        if (not 0 <= min_confidence <= 1 or not isfinite(min_lift) or min_lift < 0
                or len(required_items) > metadata["items_per_transaction"]
                or sort_by not in {"support", "lift", "kulc"}):
            raise ValueError("关联规则筛选参数无效。")
        if min_confidence < coverage["min_confidence"] or min_lift < coverage["min_lift"]:
            raise ValueError("旧快照已预过滤规则，不能降低到保存阈值以下；请重新构建。")
        required = "".join(
            " AND list_contains(str_split(concat_ws('；', antecedent, consequent), '；'), ?)"
            for _ in required_items)
        stored = self.query(f"""
            SELECT antecedent, consequent, antecedent_size, consequent_size,
                support_count, antecedent_count, consequent_count
            FROM association_rules
            WHERE support >= ? AND confidence >= ? AND lift >= ? {required}
        """, [min_support, min_confidence, min_lift, *required_items])
        # Reuse the same count-based definitions for old and new snapshot schemas.
        rows = [{**row, **rule_metrics(metadata["transaction_count"], row["antecedent_count"],
                                     row["consequent_count"], row["support_count"])}
                for row in stored.to_dict(orient="records")]
        frame = evaluated_rule_frame(rows)
        order = list(dict.fromkeys([sort_by, "confidence", "support", "antecedent", "consequent"]))
        frame = frame.sort_values(order, ascending=[c in {"antecedent", "consequent"} for c in order])
        return frame.reset_index(drop=True)

    def _pattern_items(self, antecedent, consequent):
        left, right = (side.split("；") for side in (antecedent, consequent))
        items = left + right
        max_size = self.pattern_metadata()["config"]["max_itemset_size"]
        if not all(items) or len(set(items)) != len(items) or len(items) > max_size:
            raise ValueError("规则前后件须非空、互不重叠，且不能超过快照最大项集长度。")
        return left, right

    def pattern_evidence(self, antecedent, consequent):
        """Independently recount all four cells from the full trip transaction table."""
        left, right = self._pattern_items(antecedent, consequent)
        counts = self.query("""
            WITH membership AS (
                SELECT list_has_all(items, ?) AS has_a, list_has_all(items, ?) AS has_b
                FROM trip_transactions
            )
            SELECT count(*) AS total, count(*) FILTER (WHERE has_a) AS a,
                count(*) FILTER (WHERE has_b) AS b, count(*) FILTER (WHERE has_a AND has_b) AS ab
            FROM membership
        """, [left, right]).iloc[0]
        result = rule_metrics(int(counts.total), int(counts.a), int(counts.b), int(counts.ab))
        if result["transaction_count"] != self.pattern_metadata()["transaction_count"]:
            raise ValueError("事务证据总数与快照元数据不一致。")
        stored = self.query("""
            SELECT support_count, antecedent_count, consequent_count FROM association_rules
            WHERE antecedent = ? AND consequent = ?
        """, [antecedent, consequent])
        if not stored.empty and any(int(stored.iloc[0][key]) != result[key] for key in stored.columns):
            raise ValueError("保存的规则计数与事务证据不一致，拒绝展示未经核对的指标。")
        return result

    def pattern_sample(self, antecedent: str, consequent: str, limit=100):
        left, right = self._pattern_items(antecedent, consequent)
        items = left + right
        limit = int(limit)
        if not 1 <= limit <= 100:
            raise ValueError("关联规则示例参数无效。")
        conditions = " AND ".join("list_contains(t.items, ?)" for _ in items)
        return self.query(f"""
            SELECT a.record_id, a.pickup_time, a.pickup_zone, a.dropoff_zone,
                a.distance_miles, a.amount, t.items
            FROM trip_transactions t JOIN annotated a USING (record_id)
            WHERE {conditions}
            ORDER BY a.record_id LIMIT ?
        """, items + [limit])

    @staticmethod
    def forecast_scope(split="test", zone=None):
        if split not in {"validation", "test"}:
            raise ValueError("预测分区必须是 validation 或 test。")
        terms, values = ["split = ?"], [split]
        if zone is not None:
            terms.append("zone_id = ?")
            values.append(int(zone))
        return " AND ".join(terms), values

    def forecast_metrics(self, split="test", zone=None) -> pd.DataFrame:
        scope, values = self.forecast_scope(split, zone)
        queries = []
        parameters = []
        for model, column in (("lag_24", "lag_24_prediction"),
                              ("lag_168", "lag_168_prediction"),
                              ("hist_gradient_boosting_poisson", "model_prediction")):
            queries.append(f"""
                SELECT '{model}' AS model, count(*) AS rows, sum(target) AS actual_sum,
                    avg(abs(target - {column})) AS mae,
                    CASE WHEN sum(target) > 0 THEN sum(abs(target - {column})) / sum(target) END AS wape,
                    sqrt(avg(pow(target - {column}, 2))) AS rmse
                FROM forecast_predictions WHERE {scope}
            """)
            parameters.extend(values)
        return self.query(" UNION ALL ".join(queries), parameters)

    def forecast_series(self, split="test", zone=None) -> pd.DataFrame:
        scope, values = self.forecast_scope(split, zone)
        return self.query(f"""
            SELECT target_hour, sum(target) AS actual,
                sum(lag_168_prediction) AS lag_168_prediction,
                sum(model_prediction) AS model_prediction
            FROM forecast_predictions WHERE {scope}
            GROUP BY target_hour ORDER BY target_hour
        """, values)

    def forecast_errors(self, split="test", zone=None, limit=100) -> pd.DataFrame:
        scope, values = self.forecast_scope(split, zone)
        limit = int(limit)
        if not 1 <= limit <= 500:
            raise ValueError("预测误差示例数量必须在 1 到 500 之间。")
        return self.query(f"""
            SELECT target_hour, zone_id, zone, borough, target AS actual,
                model_prediction, lag_168_prediction,
                abs(target - model_prediction) AS absolute_error
            FROM forecast_predictions WHERE {scope}
            ORDER BY absolute_error DESC, target_hour, zone_id LIMIT ?
        """, values + [limit])


def insights(raw: dict, clean: dict, hourly: pd.DataFrame, month: str) -> list[str]:
    """Deterministic observations, not generated causal stories."""
    n, kept = raw["records"], clean["records"]
    if not n:
        return ["当前窗口没有记录。请调整日期、区域或清洗规则。"]
    lines = [f"当前窗口原始 {n:,} 条，所选规则保留 {kept:,} 条（{kept / n:.2%}）。这是规则影响，不是已证实的错误率。"]
    if clean["amount_n"]:
        lines.append(f"处理后总额均值 {clean['mean_amount']:.2f} 美元，中位数 {clean['median_amount']:.2f} 美元，P95 为 {clean['p95_amount']:.2f} 美元；基于 {clean['amount_n']:,} 条有限费用记录，可进一步检查长尾影响。")
    eligible = hourly[(hourly["calendar_days"] > 0) & (hourly["trips"] > 0)]
    if not eligible.empty:
        peak = eligible.loc[eligible["daily_average"].idxmax()]
        lines.append(f"{month} 所选日期中，{peak['day_type']} {int(peak['hour']):02d}:00 的日均观测订单量最高（{peak['daily_average']:,.1f} 条/日；分母 {int(peak['calendar_days'])} 个日历日）。不是未来预测，也不等于未被满足的需求。")
    else:
        lines.append("目标月所选日期没有保留的可定位时间记录，不能据此推断城市没有出行。")
    return lines
