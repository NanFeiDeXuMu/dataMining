from datetime import datetime
import json

import duckdb
import joblib
import numpy as np
import pytest

from urbanpulse.analytics import Explorer
from urbanpulse.forecasting import FEATURE_COLUMNS, regression_metrics, split_boundaries


def test_split_boundaries_are_ordered_and_reproducible():
    bounds = split_boundaries("2025-01")
    assert bounds == {
        "feature_start": datetime(2025, 1, 8),
        "train_end": datetime(2025, 1, 22),
        "validation_end": datetime(2025, 1, 27),
        "test_end": datetime(2025, 2, 1),
    }


def test_metrics_use_explicit_denominator_and_reject_invalid_values():
    metrics = regression_metrics([0, 2, 4], [1, 1, 5])
    assert metrics["mae"] == pytest.approx(1)
    assert metrics["wape"] == pytest.approx(3 / 6)
    assert metrics["rmse"] == pytest.approx(1)
    assert regression_metrics([0, 0], [0, 1])["wape"] is None
    with pytest.raises(ValueError):
        regression_metrics([1], [-1])


def test_forecast_assets_are_leakage_safe_and_traceable(dataset):
    _, run = dataset
    for name in ("forecast_features.parquet", "forecast_predictions.parquet",
                 "forecast_metrics.parquet", "forecast_model.joblib", "forecast_report.json"):
        assert (run / name).is_file()
    report = json.loads((run / "forecast_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["forecast_horizon_hours"] == 1
    assert report["evaluation_mode"].startswith("rolling one-hour-ahead")
    assert report["split_rows"] == {"train": 1008, "validation": 360, "test": 360}
    assert set(report["feature_columns"]) == set(FEATURE_COLUMNS)
    assert len(report["model"]["candidates"]) == 3

    with duckdb.connect(str(run / "urbanpulse.duckdb"), read_only=True) as con:
        mismatch = con.execute("""
            SELECT count(*) FROM forecast_features f
            JOIN zone_hour z ON z.zone_id = f.zone_id
                AND z.hour_start = f.target_hour - INTERVAL 1 HOUR
            WHERE f.lag_1 <> z.trip_count
        """).fetchone()[0]
        assert mismatch == 0
        target_leak = con.execute("""
            SELECT count(*) FROM forecast_features f
            JOIN zone_hour z ON z.zone_id = f.zone_id AND z.hour_start = f.target_hour
            WHERE f.target <> z.trip_count
        """).fetchone()[0]
        assert target_leak == 0
        rolling_mismatch = con.execute("""
            SELECT count(*) FROM forecast_features f
            WHERE abs(f.rolling_mean_3 - (
                SELECT avg(z.trip_count) FROM zone_hour z
                WHERE z.zone_id = f.zone_id
                  AND z.hour_start >= f.target_hour - INTERVAL 3 HOUR
                  AND z.hour_start < f.target_hour
            )) > 1e-12
        """).fetchone()[0]
        assert rolling_mismatch == 0
        predictions = con.execute("SELECT * FROM forecast_predictions").fetchdf()
        assert set(predictions.split) == {"validation", "test"}
        assert len(predictions) == 720
        values = predictions[["lag_24_prediction", "lag_168_prediction", "model_prediction"]]
        assert np.isfinite(values.to_numpy()).all() and (values >= 0).all().all()
        assert con.execute("SELECT count(*) FROM forecast_metrics").fetchone()[0] == 12
        feature_sql = ", ".join(f"f.{name}" for name in FEATURE_COLUMNS)
        sample = con.execute(f"""
            SELECT {feature_sql}, p.model_prediction
            FROM forecast_features f JOIN forecast_predictions p USING (target_hour, zone_id)
            WHERE p.split = 'test' ORDER BY f.target_hour, f.zone_id LIMIT 10
        """).fetchdf()
    bundle = joblib.load(run / "forecast_model.joblib")
    assert tuple(bundle["feature_columns"]) == FEATURE_COLUMNS
    reproduced = bundle["model"].predict(sample.loc[:, FEATURE_COLUMNS])
    np.testing.assert_allclose(reproduced, sample.model_prediction, rtol=1e-12, atol=1e-12)

    with Explorer(run / "urbanpulse.duckdb", "2025-01") as explorer:
        assert len(explorer.forecast_metrics("test")) == 3
        assert len(explorer.forecast_series("test")) == 5 * 24
        assert not explorer.forecast_errors("test", zone=1).empty
        with pytest.raises(ValueError):
            explorer.forecast_metrics("train")
