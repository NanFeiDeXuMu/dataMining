import random

import pytest

from urbanpulse.analytics import Explorer
from urbanpulse.config import PATTERN_CONFIG
from urbanpulse.patterns import apriori, association_rules, fp_growth


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
    rules = association_rules(itemsets, 6, min_confidence=0, min_lift=0)
    rule = rules[(rules.antecedent == "A") & (rules.consequent == "B")].iloc[0]
    assert rule.support_count == 4
    assert rule.support == pytest.approx(4 / 6)
    assert rule.confidence == pytest.approx(4 / 5)
    assert rule.lift == pytest.approx((4 / 5) / (5 / 6))


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
