"""Weighted Apriori and FP-Growth for interpretable trip-pattern mining."""

from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations
from math import ceil, sqrt
from numbers import Integral
from pathlib import Path
from time import perf_counter

import pandas as pd

from .config import PATTERN_CONFIG, PATTERN_FIELDS, PatternConfig


def _transactions(baskets):
    normalized = []
    for items, weight in baskets:
        transaction = tuple(sorted(set(items)))
        weight = int(weight)
        if transaction and weight > 0:
            normalized.append((transaction, weight))
    return normalized


def _minimum_count(total: int, support: float) -> int:
    if not 0 < support <= 1:
        raise ValueError("最小支持度必须在 (0, 1]。")
    return max(1, ceil(total * support)) if total else 0


def apriori(baskets, min_support=0.01, max_size=3):
    """Return every frequent itemset and its weighted support count."""
    transactions = _transactions(baskets)
    total = sum(weight for _, weight in transactions)
    min_count = _minimum_count(total, min_support)
    if not total or max_size < 1:
        return {}
    singles = Counter()
    for items, weight in transactions:
        singles.update({item: weight for item in items})
    level = {frozenset((item,)) for item, count in singles.items() if count >= min_count}
    result = {itemset: singles[next(iter(itemset))] for itemset in level}
    for size in range(2, max_size + 1):
        universe = sorted({item for itemset in level for item in itemset})
        candidates = {
            frozenset(parts) for parts in combinations(universe, size)
            if all(frozenset(subset) in level for subset in combinations(parts, size - 1))
        }
        if not candidates:
            break
        counts = Counter()
        for items, weight in transactions:
            for parts in combinations(items, size):
                candidate = frozenset(parts)
                if candidate in candidates:
                    counts[candidate] += weight
        level = {candidate for candidate, count in counts.items() if count >= min_count}
        result.update({candidate: counts[candidate] for candidate in level})
        if not level:
            break
    return result


class _FPNode:
    __slots__ = ("item", "count", "parent", "children")

    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}


def _fp_tree(transactions, min_count):
    supports = Counter()
    for items, weight in transactions:
        supports.update({item: weight for item in items})
    supports = Counter({item: count for item, count in supports.items() if count >= min_count})
    if not supports:
        return None, {}, supports
    order = {item: rank for rank, item in enumerate(sorted(supports, key=lambda x: (-supports[x], x)))}
    root = _FPNode(None)
    headers = defaultdict(list)
    for items, weight in transactions:
        current = root
        for item in sorted((item for item in items if item in supports), key=lambda x: order[x]):
            if item not in current.children:
                current.children[item] = _FPNode(item, 0, current)
                headers[item].append(current.children[item])
            current = current.children[item]
            current.count += weight
    return root, headers, supports


def fp_growth(baskets, min_support=0.01, max_size=3):
    """Mine weighted transactions with an FP-tree and conditional pattern bases."""
    transactions = _transactions(baskets)
    total = sum(weight for _, weight in transactions)
    min_count = _minimum_count(total, min_support)
    if not total or max_size < 1:
        return {}
    result = {}

    def visit(conditional, prefix):
        _, headers, supports = _fp_tree(conditional, min_count)
        for item in sorted(supports, key=lambda x: (supports[x], x)):
            itemset = frozenset((*prefix, item))
            result[itemset] = supports[item]
            if len(itemset) >= max_size:
                continue
            paths = []
            for node in headers[item]:
                path, parent = [], node.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    paths.append((tuple(path), node.count))
            if paths:
                visit(paths, itemset)

    visit(transactions, frozenset())
    return result


def frequent_frame(itemsets, total):
    columns = ["itemset", "size", "support_count", "support"]
    rows = [{
        "itemset": "；".join(sorted(itemset)), "size": len(itemset),
        "support_count": int(count), "support": count / total,
    } for itemset, count in itemsets.items()]
    rows.sort(key=lambda row: (row["size"], -row["support_count"], row["itemset"]))
    return pd.DataFrame(rows, columns=columns)


RULE_COUNTS = ["transaction_count", "support_count", "antecedent_count", "consequent_count",
               "n11", "n10", "n01", "n00"]
RULE_METRICS = ["support", "antecedent_support", "consequent_support", "confidence",
                "reverse_confidence", "lift", "kulc", "imbalance_ratio", "jaccard", "cosine"]


def rule_metrics(total, antecedent_count, consequent_count, support_count):
    """Evaluate one disjoint A/B rule from exact counts; undefined ratios stay null."""
    counts = (total, antecedent_count, consequent_count, support_count)
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in counts):
        raise ValueError("规则计数必须是非负整数。")
    total, a, b, ab = map(int, counts)
    if not max(0, a + b - total) <= ab <= min(a, b) or max(a, b) > total:
        raise ValueError("规则计数不能形成有效的列联表。")
    notes = []

    def ratio(numerator, denominator, label):
        if denominator == 0:
            notes.append(f"{label}不可计算：分母为零")
            return None
        return numerator / denominator

    confidence = ratio(ab, a, "A→B 置信度")
    reverse = ratio(ab, b, "B→A 置信度")
    kulc = (confidence + reverse) / 2 if confidence is not None and reverse is not None else None
    if kulc is None:
        notes.append("Kulc 不可计算：至少一个方向的置信度不可计算")
    result = {
        "transaction_count": total, "support_count": ab,
        "antecedent_count": a, "consequent_count": b,
        "n11": ab, "n10": a - ab, "n01": b - ab, "n00": total - a - b + ab,
        "support": ratio(ab, total, "支持度"),
        "antecedent_support": ratio(a, total, "A 基准比例"),
        "consequent_support": ratio(b, total, "B 基准比例"),
        "confidence": confidence, "reverse_confidence": reverse,
        "lift": ratio(total * ab, a * b, "提升度"), "kulc": kulc,
        "imbalance_ratio": ratio(abs(a - b), a + b - ab, "IR"),
        "jaccard": ratio(ab, a + b - ab, "Jaccard"),
        "cosine": ratio(ab, sqrt(a * b), "cosine"),
    }
    return {**result, "metric_notes": "；".join(notes)}


def evaluated_rule_frame(rows):
    """Keep numeric Parquet/SQL types even when a snapshot contains no rules."""
    columns = ["antecedent", "consequent", "antecedent_size", "consequent_size",
               *RULE_COUNTS, *RULE_METRICS, "metric_notes"]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.astype({**{c: "string" for c in ["antecedent", "consequent", "metric_notes"]},
                         **{c: "int64" for c in ["antecedent_size", "consequent_size", *RULE_COUNTS]},
                         **{c: "float64" for c in RULE_METRICS}})


def rule_coverage(metadata):
    """Old snapshots persisted only rules passing confidence/lift thresholds."""
    cfg = metadata["config"]
    return metadata.get("rule_coverage", {
        "min_support": cfg["min_support"], "max_itemset_size": cfg["max_itemset_size"],
        "min_confidence": cfg["min_confidence"], "min_lift": cfg["min_lift"],
        "complete_within_itemsets": False,
    })


def fold_rule_pairs(rules):
    """Keep the best matching direction per A/B partition, preserving input order."""
    frame = rules.copy()
    frame["pair"] = [tuple(sorted((a, b))) for a, b in zip(frame.antecedent, frame.consequent)]
    frame["matched_directions"] = frame.groupby("pair")["antecedent"].transform("size")
    return frame.drop_duplicates("pair").reset_index(drop=True)


def association_rules(itemsets, total):
    """Generate every direction covered by frequent itemsets; filter at query time."""
    if not total:
        return evaluated_rule_frame([])
    rows = []
    for itemset, joint_count in itemsets.items():
        if len(itemset) < 2:
            continue
        ordered = sorted(itemset)
        for size in range(1, len(ordered)):
            for left in combinations(ordered, size):
                antecedent = frozenset(left)
                consequent = itemset - antecedent
                left_count = itemsets[antecedent]
                right_count = itemsets[consequent]
                metrics = rule_metrics(total, left_count, right_count, joint_count)
                rows.append({
                    "antecedent": "；".join(sorted(antecedent)),
                    "consequent": "；".join(sorted(consequent)),
                    "antecedent_size": len(antecedent), "consequent_size": len(consequent),
                    **metrics,
                })
    rows.sort(key=lambda row: (-row["lift"], -row["confidence"], -row["support"],
                               row["antecedent"], row["consequent"]))
    return evaluated_rule_frame(rows)


def build_pattern_assets(con, run: Path, source_columns, config: PatternConfig = PATTERN_CONFIG):
    """Create transactions, mine both algorithms, verify equality and persist evidence."""
    payment = """CASE TRY_CAST(payment_type AS INTEGER)
        WHEN 1 THEN '支付=信用卡' WHEN 2 THEN '支付=现金' WHEN 3 THEN '支付=免付'
        WHEN 4 THEN '支付=争议' WHEN 5 THEN '支付=未知' WHEN 6 THEN '支付=作废'
        ELSE '支付=其他/缺失' END""" if "payment_type" in source_columns else "'支付=字段缺失'"
    con.execute(f"""
        CREATE TABLE trip_transactions AS
        WITH bucketed AS (
            SELECT record_id,
                concat('日期=', day_type) AS day_item,
                CASE WHEN hour(pickup_time) < 6 THEN '时段=凌晨(00-05)'
                    WHEN hour(pickup_time) < 10 THEN '时段=早高峰(06-09)'
                    WHEN hour(pickup_time) < 16 THEN '时段=白天(10-15)'
                    WHEN hour(pickup_time) < 20 THEN '时段=晚高峰(16-19)'
                    ELSE '时段=夜间(20-23)' END AS time_item,
                concat('上车行政区=', pickup_borough) AS pickup_item,
                concat('下车行政区=', dropoff_borough) AS dropoff_item,
                CASE WHEN distance_miles = 0 THEN '距离=零'
                    WHEN distance_miles < 2 THEN '距离=短途(<2mi)'
                    WHEN distance_miles < 5 THEN '距离=中途(2-5mi)'
                    WHEN distance_miles < 15 THEN '距离=长途(5-15mi)'
                    ELSE '距离=超长途(>=15mi)' END AS distance_item,
                CASE WHEN amount < 0 THEN '费用=负值'
                    WHEN amount < 15 THEN '费用=低(<$15)'
                    WHEN amount < 30 THEN '费用=中($15-30)'
                    WHEN amount < 75 THEN '费用=高($30-75)'
                    ELSE '费用=极高(>=$75)' END AS amount_item,
                {payment} AS payment_item,
                CASE WHEN passengers IS NULL OR NOT isfinite(passengers) THEN '乘客=缺失'
                    WHEN passengers <= 0 THEN '乘客=零'
                    WHEN passengers = 1 THEN '乘客=1'
                    WHEN passengers = 2 THEN '乘客=2'
                    ELSE '乘客=3+' END AS passenger_item
            FROM cleaned
        )
        SELECT *, [day_item, time_item, pickup_item, dropoff_item, distance_item,
            amount_item, payment_item, passenger_item] AS items
        FROM bucketed
    """)
    con.execute("CREATE TABLE transaction_baskets AS SELECT items, count(*)::BIGINT AS transaction_count FROM trip_transactions GROUP BY items")
    transaction_count = con.execute("SELECT count(*) FROM trip_transactions").fetchone()[0]
    baskets = [(items, count) for items, count in con.execute(
        "SELECT items, transaction_count FROM transaction_baskets ORDER BY items").fetchall()]
    if sum(weight for _, weight in baskets) != transaction_count:
        raise AssertionError("聚合事务篮子的权重与行程事务数不守恒。")

    started = perf_counter()
    apriori_sets = apriori(baskets, config.min_support, config.max_itemset_size)
    apriori_seconds = perf_counter() - started
    started = perf_counter()
    fp_sets = fp_growth(baskets, config.min_support, config.max_itemset_size)
    fp_seconds = perf_counter() - started
    if apriori_sets != fp_sets:
        raise AssertionError("Apriori 与 FP-Growth 的频繁项集不一致。")

    frequent = frequent_frame(apriori_sets, transaction_count)
    # Mining coverage is independent of the default UI confidence/lift filters.
    rules = association_rules(apriori_sets, transaction_count)
    default_rules = rules[(rules.confidence >= config.min_confidence) & (rules.lift >= config.min_lift)]
    expected_rules = sum(2 ** len(items) - 2 for items in apriori_sets if len(items) >= 2)
    if len(rules) != expected_rules or rules.duplicated(["antecedent", "consequent"]).any():
        raise AssertionError("已挖掘项集的方向规则不完整或重复。")
    con.register("_frequent_frame", frequent)
    con.register("_rule_frame", rules)
    con.execute("CREATE TABLE frequent_itemsets AS SELECT * FROM _frequent_frame")
    con.execute("CREATE TABLE association_rules AS SELECT * FROM _rule_frame")
    con.unregister("_frequent_frame")
    con.unregister("_rule_frame")
    con.execute(f"COPY trip_transactions TO '{str(run / 'trip_transactions.parquet').replace("'", "''")}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY frequent_itemsets TO '{str(run / 'frequent_itemsets.parquet').replace("'", "''")}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY association_rules TO '{str(run / 'association_rules.parquet').replace("'", "''")}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    return {
        "schema_version": 2,
        "transaction_count": int(transaction_count), "unique_baskets": len(baskets),
        "items_per_transaction": len(PATTERN_FIELDS), "fields": list(PATTERN_FIELDS),
        "config": asdict(config), "frequent_itemsets": len(frequent), "rules": len(rules),
        "default_display_rules": len(default_rules),
        "rule_coverage": {
            "min_support": config.min_support, "max_itemset_size": config.max_itemset_size,
            "min_confidence": 0.0, "min_lift": 0.0, "complete_within_itemsets": True,
        },
        "metric_policy": {
            "denominator": "全月默认清洗后的行程事务数；规则筛选不改变分母。",
            "null_transactions": "相对于 A、B 均不出现的事务；不等于缺失记录。",
            "undefined": "分母为零时指标为空，并记录不可计算原因。",
            "kulc": "(nAB/nA + nAB/nB)/2",
            "imbalance_ratio": "abs(nA-nB)/(nA+nB-nAB)",
        },
        "algorithms": {
            "apriori_itemsets": len(apriori_sets), "fpgrowth_itemsets": len(fp_sets),
            "identical": True, "apriori_seconds": round(apriori_seconds, 6),
            "fpgrowth_seconds": round(fp_seconds, 6),
        },
        "top_rules": default_rules.head(10).to_dict(orient="records"),
        "limitations": [
            "规则描述共同出现关系，不证明因果关系或未来必然发生。",
            "模式基于默认清洗策略和固定分箱；改变口径可能改变结果。",
            "最低支持度以下的项集未保存，页面不能恢复未挖掘的低频规则。",
            "已保存项集的全部方向规则可筛选；低于联合支持度门槛的负模式不在覆盖范围内。",
            "Kulc 衡量双向重合，IR 描述边际不平衡；它们不证明因果或统计显著性。",
        ],
    }
