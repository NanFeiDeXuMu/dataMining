from datetime import date
import json

import duckdb
import pandas as pd
import pytest

from urbanpulse.analytics import Explorer, active_run
from urbanpulse.config import DEFAULT_RULES, RULES, keep_expression, month_bounds
from urbanpulse.download import sha256, write_json
from urbanpulse.pipeline import build


def test_preserves_all_rows_and_duplicate_looking_trips(dataset):
    _, run = dataset
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        assert ex.stats()["records"] == 22
        ids = ex.query(f"SELECT record_id FROM annotated WHERE {keep_expression(DEFAULT_RULES)} ORDER BY record_id").record_id.tolist()
        assert ids == [0, 1, 2, 3, 4, 5, 13, 14, 15, 19, 21]


def test_optional_flags_not_default_errors(dataset):
    _, run = dataset
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        assert ex.stats(DEFAULT_RULES)["records"] == 11
        assert ex.stats(DEFAULT_RULES + ("negative_amount",))["records"] == 10
        assert ex.stats(DEFAULT_RULES + ("zero_distance",))["records"] == 9
        assert ex.stats(DEFAULT_RULES + ("high_amount",))["records"] == 10
        flags = ex.query("SELECT q_missing_passengers, q_high_amount FROM annotated WHERE record_id = 3").iloc[0]
        assert bool(flags.q_missing_passengers)


def test_nonfinite_missing_and_noninteger_id(dataset):
    _, run = dataset
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        rows = ex.query("SELECT record_id, q_missing_amount, q_unknown_pickup FROM annotated WHERE record_id IN (10, 17, 20) ORDER BY record_id")
        assert rows.q_missing_amount.tolist() == [True, False, True]
        assert rows.q_unknown_pickup.tolist() == [False, True, False]
        assert ex.stats()["amount_n"] == 21


def test_time_boundaries_and_subsecond_duration(dataset):
    _, run = dataset
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        assert ex.stats(DEFAULT_RULES, date(2025, 1, 31), date(2025, 1, 31))["records"] == 1
        flags = ex.query("SELECT q_outside_month FROM annotated WHERE record_id IN (8, 18, 19) ORDER BY record_id")
        assert flags.q_outside_month.tolist() == [True, True, False]
        assert ex.query("SELECT duration_min FROM annotated WHERE record_id=21").iloc[0, 0] > 0


def test_rule_flags_never_null_and_overlap_not_additive(dataset):
    _, run = dataset
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        for rule in RULES:
            assert ex.query(f"SELECT count(*) AS n FROM annotated WHERE q_{rule.key} IS NULL").iloc[0, 0] == 0
        assert ex.quality()["记录数"].sum() > ex.stats()["records"] - ex.stats(DEFAULT_RULES)["records"]


def test_full_grid_and_count_conservation(dataset):
    _, run = dataset
    with duckdb.connect() as con:
        grid = con.read_parquet(str(run / "zone_hour.parquet")).df()
        assert len(grid) == 3 * 31 * 24
        assert int(grid.trip_count.sum()) == 11
        assert int(grid.loc[grid.zone_id == 3, "trip_count"].sum()) == 0
        assert grid.loc[grid.trip_count == 0, "mean_total_amount"].isna().all()
        assert not grid.duplicated(["zone_id", "hour_start"]).any()


def test_calendar_normalization_and_daily_gaps(dataset):
    _, run = dataset
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        hourly = ex.hourly(DEFAULT_RULES)
        assert set(hourly.loc[hourly.day_type == "工作日", "calendar_days"]) == {23}
        assert set(hourly.loc[hourly.day_type == "周末", "calendar_days"]) == {8}
        assert len(hourly) == 48
        assert int(hourly.trips.sum()) == 11
        assert len(ex.daily(DEFAULT_RULES)) == 31
        weekend = ex.hourly(DEFAULT_RULES, date(2025, 1, 4), date(2025, 1, 4))
        assert weekend.loc[weekend.day_type == "工作日", "daily_average"].isna().all()


def test_empty_scope_and_safe_queries(dataset):
    _, run = dataset
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        stats = ex.stats(DEFAULT_RULES, zone=3)
        assert stats["records"] == 0 and stats["mean_amount"] is None
        assert ex.sample(zone=3).empty
        assert ex.ranking(zone=3).empty
        with pytest.raises(ValueError):
            ex.histogram("amount);DROP TABLE annotated;--")
        with pytest.raises(ValueError):
            ex.stats(("invented_rule",))


def test_histogram_counts_include_tails_and_missing(dataset):
    _, run = dataset
    with Explorer(run / "urbanpulse.duckdb", "2025-01") as ex:
        for rules in ((), DEFAULT_RULES):
            for field in ("amount", "distance_miles", "duration_min"):
                histogram, tails = ex.histogram(field, rules)
                assert histogram.trips.sum() + tails["below"] + tails["above"] + tails["missing"] == ex.stats(rules)["records"]


def test_source_checksum_protects_snapshot(raw_dataset):
    lookup = raw_dataset / "raw/2025-01/taxi_zone_lookup.csv"
    lookup.write_text(lookup.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="校验失败"):
        build("2025-01", raw_dataset)
    assert not (raw_dataset / "processed/2025-01/active.json").exists()


def test_duplicate_lookup_rejected_without_losing_active(raw_dataset):
    original = build("2025-01", raw_dataset)
    folder = raw_dataset / "raw/2025-01"
    lookup = folder / "taxi_zone_lookup.csv"
    rows = pd.read_csv(lookup)
    pd.concat([rows, rows.iloc[:1]]).to_csv(lookup, index=False)
    manifest = json.loads((folder / "source_manifest.json").read_text(encoding="utf-8"))
    manifest["files"][1]["sha256"] = sha256(lookup)
    write_json(folder / "source_manifest.json", manifest)
    with pytest.raises(ValueError, match="重复"):
        build("2025-01", raw_dataset)
    assert active_run(raw_dataset, "2025-01") == original


def test_rebuild_stable_and_previous_run_preserved(raw_dataset):
    first = build("2025-01", raw_dataset)
    second = build("2025-01", raw_dataset)
    assert first != second and first.exists()
    one = json.loads((first / "report.json").read_text(encoding="utf-8"))
    two = json.loads((second / "report.json").read_text(encoding="utf-8"))
    for key in ("raw", "clean", "zone_hour_rows", "quality", "schema", "insights"):
        assert one[key] == two[key]
    with duckdb.connect() as con:
        assert con.read_parquet(str(first / "cleaned_trips.parquet")).df().equals(con.read_parquet(str(second / "cleaned_trips.parquet")).df())
    assert active_run(raw_dataset, "2025-01") == second


@pytest.mark.parametrize("month", ["2025-00", "2025-13", "2025/01", "../2025-01", "2025-1"])
def test_reject_invalid_month(month):
    with pytest.raises(ValueError):
        month_bounds(month)


def test_month_rollover():
    assert month_bounds("2025-12") == (date(2025, 12, 1), date(2026, 1, 1))
