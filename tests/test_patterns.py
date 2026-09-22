import random
import json

import duckdb
import pandas as pd
import pytest

from urbanpulse.analytics import Explorer
from urbanpulse.config import PATTERN_CONFIG, PatternConfig
from urbanpulse.download import sha256, write_json
from urbanpulse.patterns import (apriori, association_rules, build_pattern_assets, fold_rule_pairs,
                                 fp_growth, rule_coverage, rule_metrics)


def test_weighted_algorithms_match_expected_counts():
    baskets = [
        (("A", "B", "C"), 2),
        (("A", "B"), 2),
        (("A", "C"), 1),
        (("B", "C"), 1),
    ]
    expected = {
        frozenset(("A",)): 5, frozenset(("B",)): 5, frozenset(("C",)): 4,
        frozenset(("A", "B")): 4, frozenset(("A", "C")): 3,
        frozenset(("B", "C")): 3, frozenset(("A", "B", "C")): 2,
    }
    assert apriori(baskets, 0.3, 3) == expected
    assert fp_growth(baskets, 0.3, 3) == expected


def test_weighted_baskets_equal_expanded_transactions():
    weighted = [(("A", "B"), 3), (("A", "C"), 2), (("B",), 1)]
    expanded = [(items, 1) for items, weight in weighted for _ in range(weight)]
    assert apriori(weighted, 0.2, 3) == apriori(expanded, 0.2, 3)
    assert fp_growth(weighted, 0.2, 3) == fp_growth(expanded, 0.2, 3)


def test_algorithms_agree_on_deterministic_random_baskets():
    rng = random.Random(20250904)
    universe = list("ABCDEFGH")
    baskets = []
    for _ in range(80):
        size = rng.randint(1, 6)
        baskets.append((tuple(rng.sample(universe, size)), rng.randint(1, 5)))
    assert apriori(baskets, 0.08, 3) == fp_growth(baskets, 0.08, 3)


def test_rule_metrics_use_explicit_denominators():
    itemsets = apriori([(("A", "B"), 4), (("A",), 1), (("B",), 1)], 0.1, 2)
    rules = association_rules(itemsets, 6)
    rule = rules[(rules.antecedent == "A") & (rules.consequent == "B")].iloc[0]
    assert rule.support_count == 4
    assert rule.support == pytest.approx(4 / 6)
    assert rule.confidence == pytest.approx(4 / 5)
    assert rule.lift == pytest.approx((4 / 5) / (5 / 6))


def test_rule_generation_keeps_low_confidence_and_lift():
    itemsets = apriori([(("A", "B"), 1), (("A",), 9), (("B",), 1)], 0.01, 2)
    rules = association_rules(itemsets, 11)
    assert set(zip(rules.antecedent, rules.consequent)) == {("A", "B"), ("B", "A")}
    assert rules.loc[rules.antecedent == "A", "confidence"].iloc[0] == pytest.approx(0.1)
    assert rules.lift.tolist() == pytest.approx([0.55, 0.55])


def test_pattern_assets_and_traceable_evidence(dataset):
    _, run = dataset
    for name in ("trip_transactions.parquet", "frequent_itemsets.parquet",
                 "association_rules.parquet", "pattern_report.json"):
        assert (run / name).is_file()
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        transactions = ex.query("SELECT record_id, items FROM trip_transactions ORDER BY record_id")
        assert len(transactions) == 11
        for items in transactions["items"]:
            assert len(items) == 8
            assert len({item.split("=", 1)[0] for item in items}) == 8
        assert ex.query("SELECT sum(transaction_count) AS n FROM transaction_baskets").iloc[0, 0] == 11
        rules = ex.pattern_rules()
        assert not rules.empty
        weekday = ex.pattern_rules(required_items=("日期=工作日",))
        assert not weekday.empty
        assert all("日期=工作日" in f"{row.antecedent}；{row.consequent}"
                   for row in weekday.itertuples())
        selected = rules.iloc[0]
        evidence = ex.pattern_sample(selected.antecedent, selected.consequent)
        assert not evidence.empty
        required = set((selected.antecedent + "；" + selected.consequent).split("；"))
        assert all(required.issubset(set(items)) for items in evidence["items"])
        with pytest.raises(ValueError):
            ex.pattern_rules(PATTERN_CONFIG.min_support / 2)


def test_l3_high_confidence_can_have_lift_below_one():
    metrics = rule_metrics(1000, 600, 750, 400)
    assert [metrics[k] for k in ("n11", "n10", "n01", "n00")] == [400, 200, 350, 50]
    assert metrics["confidence"] == pytest.approx(2 / 3)
    assert metrics["consequent_support"] == 0.75
    assert metrics["lift"] == pytest.approx(8 / 9)
    assert metrics["reverse_confidence"] == pytest.approx(8 / 15)
    assert metrics["kulc"] == pytest.approx(0.6)
    assert metrics["imbalance_ratio"] == pytest.approx(150 / 950)


def test_null_transactions_change_lift_but_not_null_invariant_metrics():
    original = rule_metrics(1000, 600, 750, 400)
    enlarged = rule_metrics(101000, 600, 750, 400)
    for key in ("confidence", "reverse_confidence", "kulc", "imbalance_ratio", "jaccard", "cosine"):
        assert original[key] == enlarged[key]
    assert enlarged["lift"] > original["lift"]
    assert enlarged["support"] < original["support"]
    assert enlarged["n00"] - original["n00"] == 100000


def test_swapping_rule_sides_preserves_symmetric_metrics():
    forward, reverse = rule_metrics(1000, 600, 750, 400), rule_metrics(1000, 750, 600, 400)
    for key in ("support", "lift", "kulc", "imbalance_ratio", "jaccard", "cosine"):
        assert forward[key] == reverse[key]
    assert forward["confidence"] == reverse["reverse_confidence"]
    assert forward["n10"] == reverse["n01"]


def test_zero_denominators_are_undefined_and_explained():
    empty = rule_metrics(0, 0, 0, 0)
    assert all(empty[key] is None for key in ("support", "confidence", "lift", "kulc", "imbalance_ratio"))
    assert "分母为零" in empty["metric_notes"]
    absent_a = rule_metrics(10, 0, 5, 0)
    assert absent_a["confidence"] is None and absent_a["kulc"] is None
    assert absent_a["reverse_confidence"] == 0
    assert absent_a["imbalance_ratio"] == 1
    assert rule_metrics(10, 4, 5, 0)["kulc"] == 0


@pytest.mark.parametrize("counts", [(10, 9, 9, 1), (10, 2, 3, 4), (10, 11, 3, 3),
                                    (-1, 0, 0, 0), (10, 2.5, 3, 1)])
def test_inconsistent_or_noninteger_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        rule_metrics(*counts)


def test_new_snapshot_saves_all_directions_with_default_filters_separate(dataset):
    _, run = dataset
    metadata = json.loads((run / "pattern_report.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        rules = ex.pattern_rules(min_confidence=0, min_lift=0)
        frequent = ex.query("SELECT size FROM frequent_itemsets")
        assert len(rules) == sum(2 ** size - 2 for size in frequent["size"] if size >= 2)
        assert len(rules) == metadata["rules"] > metadata["default_display_rules"]
        assert len(ex.pattern_rules()) == metadata["default_display_rules"]
        assert (rules.lift < PATTERN_CONFIG.min_lift).any()
        assert (rules.confidence < PATTERN_CONFIG.min_confidence).any()
        assert set(zip(rules.antecedent, rules.consequent)) == set(zip(rules.consequent, rules.antecedent))
        saved = pd.read_parquet(run / "association_rules.parquet")
        assert len(saved) == len(rules)
        assert (saved[["n11", "n10", "n01", "n00"]].sum(axis=1) == metadata["transaction_count"]).all()
        assert (saved[["n11", "n10", "n01", "n00"]] >= 0).all().all()


def test_independent_evidence_matches_transactions_and_saved_metrics(dataset):
    with Explorer(dataset[1] / "urbanpulse.duckdb", "2025-01") as ex:
        all_items = [set(items) for items in ex.query("SELECT items FROM trip_transactions")["items"]]
        for rule in ex.pattern_rules().head(5).itertuples():
            left, right = set(rule.antecedent.split("；")), set(rule.consequent.split("；"))
            evidence = ex.pattern_evidence(rule.antecedent, rule.consequent)
            assert evidence["antecedent_count"] == sum(left <= items for items in all_items)
            assert evidence["consequent_count"] == sum(right <= items for items in all_items)
            assert evidence["support_count"] == sum(left | right <= items for items in all_items)
            for key in ("support", "confidence", "lift", "kulc", "imbalance_ratio"):
                assert evidence[key] == pytest.approx(getattr(rule, key))


def test_legacy_snapshot_is_read_only_and_declares_missing_rule_coverage(legacy_dataset):
    run = legacy_dataset[1]
    before = sha256(run / "urbanpulse.duckdb")
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        coverage = rule_coverage(ex.pattern_metadata())
        assert not coverage["complete_within_itemsets"]
        rules = ex.pattern_rules()
        assert not rules.empty and rules.kulc.notna().all()
        row = rules.iloc[0]
        assert ex.pattern_evidence(row.antecedent, row.consequent)["kulc"] == pytest.approx(row.kulc)
        for kwargs in ({"min_confidence": 0}, {"min_lift": 0}):
            with pytest.raises(ValueError, match="旧快照已预过滤"):
                ex.pattern_rules(**kwargs)
    assert sha256(run / "urbanpulse.duckdb") == before


def test_snapshot_thresholds_and_sample_length_override_global_defaults(dataset, tmp_path):
    run = tmp_path / "run-custom"
    run.mkdir()
    cfg = PatternConfig(min_support=0.2, min_confidence=0.8, min_lift=0.4, max_itemset_size=4)
    with duckdb.connect(str(run / "urbanpulse.duckdb")) as con:
        con.execute("CREATE TABLE cleaned AS SELECT * FROM read_parquet(?)", [str(dataset[1] / "cleaned_trips.parquet")])
        con.execute("CREATE TABLE annotated AS SELECT * FROM cleaned")
        metadata = build_pattern_assets(con, run, ["payment_type"], cfg)
    write_json(run / "pattern_report.json", metadata)
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        defaults = ex.pattern_rules()
        assert defaults.support.min() >= 0.2 and defaults.confidence.min() >= 0.8
        with pytest.raises(ValueError, match="快照挖掘阈值"):
            ex.pattern_rules(min_support=0.1)
        rules = ex.pattern_rules(min_confidence=0, min_lift=0)
        long_rule = rules[rules.antecedent_size + rules.consequent_size == 4].iloc[0]
        assert not ex.pattern_sample(long_rule.antecedent, long_rule.consequent).empty
        assert ex.pattern_evidence(long_rule.antecedent, long_rule.consequent)["support_count"] == long_rule.support_count
        with pytest.raises(ValueError, match="互不重叠"):
            ex.pattern_sample(long_rule.antecedent, long_rule.antecedent)


def test_sorting_full_export_and_pair_folding_do_not_lose_directions(dataset):
    with Explorer(dataset[1] / "urbanpulse.duckdb", "2025-01") as ex:
        for sort_by in ("lift", "support", "kulc"):
            full = ex.pattern_rules(min_confidence=0, min_lift=0, sort_by=sort_by)
            assert len(full) > 200
            assert full[sort_by].is_monotonic_decreasing
            assert len(full) == ex.query("SELECT count(*) AS n FROM association_rules").iloc[0, 0]
            pairs = fold_rule_pairs(full)
            assert len(pairs) * 2 == len(full)
            assert (pairs.matched_directions == 2).all()
            assert pairs.pair.is_unique
        partial = ex.pattern_rules()
        assert fold_rule_pairs(partial).matched_directions.sum() == len(partial)
        assert fold_rule_pairs(partial.iloc[:0]).empty
        with pytest.raises(ValueError):
            ex.pattern_rules(sort_by="lift; DROP TABLE association_rules")


def test_evidence_rejects_stale_saved_counts(legacy_dataset):
    run = legacy_dataset[1]
    with duckdb.connect(str(run / "urbanpulse.duckdb")) as con:
        a, b = con.execute("SELECT antecedent, consequent FROM association_rules LIMIT 1").fetchone()
        con.execute("UPDATE association_rules SET support_count = support_count + 1 WHERE antecedent = ? AND consequent = ?", [a, b])
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        with pytest.raises(ValueError, match="计数与事务证据不一致"):
            ex.pattern_evidence(a, b)


def test_empty_rule_frame_keeps_numeric_schema():
    frame = association_rules({frozenset({"A"}): 1}, 1)
    assert frame.empty
    assert frame.transaction_count.dtype == "int64"
    assert frame.kulc.dtype == "float64"
