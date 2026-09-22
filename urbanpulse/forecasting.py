"""Leakage-safe one-hour-ahead trip-count experiments.

Each row is indexed by the hour being predicted. Every learned feature ends at
least one hour before that target, while target-hour calendar fields are known
in advance. Model selection uses validation only; test is evaluated once after
refitting on train + validation.
"""

from dataclasses import asdict
from datetime import datetime, time, timedelta
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import FORECAST_CONFIG, ForecastConfig, month_bounds
from .download import write_json


FEATURE_COLUMNS = (
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
    "rolling_mean_3", "rolling_std_3", "rolling_mean_24", "rolling_std_24",
    "rolling_mean_168", "rolling_std_168", "zone_past_mean",
    "same_hour_past_mean", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "borough_bronx", "borough_brooklyn", "borough_manhattan",
    "borough_queens", "borough_staten_island", "borough_other",
)

MODEL_CANDIDATES = (
    {"learning_rate": 0.08, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"learning_rate": 0.08, "max_leaf_nodes": 31, "l2_regularization": 10.0},
)


def _literal(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def create_forecast_features(con: duckdb.DuckDBPyConnection) -> None:
    """Create features whose observation windows never include the target row."""
    columns = {row[0] for row in con.execute("DESCRIBE zone_hour").fetchall()}
    required = {"zone_id", "zone", "borough", "hour_start", "trip_count"}
    if missing := required - columns:
        raise ValueError(f"zone_hour 缺少预测字段: {sorted(missing)}")
    duplicate_n = con.execute("""
        SELECT count(*) - count(DISTINCT (zone_id, hour_start)) FROM zone_hour
    """).fetchone()[0]
    if duplicate_n:
        raise ValueError("zone_hour 包含重复的区域小时，拒绝训练。")
    invalid_n = con.execute("""
        SELECT count(*) FROM zone_hour WHERE trip_count IS NULL OR trip_count < 0
    """).fetchone()[0]
    if invalid_n:
        raise ValueError("预测目标包含缺失或负数。")
    gap_n = con.execute("""
        WITH ordered AS (
            SELECT zone_id, hour_start,
                lag(hour_start) OVER (PARTITION BY zone_id ORDER BY hour_start) AS previous_hour
            FROM zone_hour
        )
        SELECT count(*) FROM ordered
        WHERE previous_hour IS NOT NULL
          AND date_diff('hour', previous_hour, hour_start) <> 1
    """).fetchone()[0]
    if gap_n:
        raise ValueError("zone_hour 时间轴不连续，拒绝跨缺口生成滞后特征。")

    con.execute("""
        CREATE OR REPLACE TABLE forecast_features AS
        WITH calendar AS (
            SELECT zone_id, zone, borough, hour_start AS target_hour,
                trip_count::DOUBLE AS target,
                ((isodow(hour_start) - 1) * 24 + hour(hour_start))::INTEGER AS hour_of_week,
                sin(2 * pi() * hour(hour_start) / 24.0) AS hour_sin,
                cos(2 * pi() * hour(hour_start) / 24.0) AS hour_cos,
                sin(2 * pi() * (isodow(hour_start) - 1) / 7.0) AS dow_sin,
                cos(2 * pi() * (isodow(hour_start) - 1) / 7.0) AS dow_cos,
                (isodow(hour_start) IN (6, 7))::INTEGER AS is_weekend,
                (borough = 'Bronx')::INTEGER AS borough_bronx,
                (borough = 'Brooklyn')::INTEGER AS borough_brooklyn,
                (borough = 'Manhattan')::INTEGER AS borough_manhattan,
                (borough = 'Queens')::INTEGER AS borough_queens,
                (borough = 'Staten Island')::INTEGER AS borough_staten_island,
                (borough NOT IN ('Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'))::INTEGER
                    AS borough_other,
                trip_count
            FROM zone_hour
        ), featured AS (
            SELECT * EXCLUDE (trip_count),
                lag(trip_count, 1) OVER zone_time AS lag_1,
                lag(trip_count, 2) OVER zone_time AS lag_2,
                lag(trip_count, 3) OVER zone_time AS lag_3,
                lag(trip_count, 24) OVER zone_time AS lag_24,
                lag(trip_count, 48) OVER zone_time AS lag_48,
                lag(trip_count, 168) OVER zone_time AS lag_168,
                avg(trip_count) OVER (PARTITION BY zone_id ORDER BY target_hour
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_mean_3,
                stddev_samp(trip_count) OVER (PARTITION BY zone_id ORDER BY target_hour
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_std_3,
                avg(trip_count) OVER (PARTITION BY zone_id ORDER BY target_hour
                    ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS rolling_mean_24,
                stddev_samp(trip_count) OVER (PARTITION BY zone_id ORDER BY target_hour
                    ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS rolling_std_24,
                avg(trip_count) OVER (PARTITION BY zone_id ORDER BY target_hour
                    ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING) AS rolling_mean_168,
                stddev_samp(trip_count) OVER (PARTITION BY zone_id ORDER BY target_hour
                    ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING) AS rolling_std_168,
                avg(trip_count) OVER (PARTITION BY zone_id ORDER BY target_hour
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS zone_past_mean,
                avg(trip_count) OVER (PARTITION BY zone_id, hour_of_week ORDER BY target_hour
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS same_hour_past_mean
            FROM calendar
            WINDOW zone_time AS (PARTITION BY zone_id ORDER BY target_hour)
        )
        SELECT * FROM featured WHERE lag_168 IS NOT NULL
        ORDER BY target_hour, zone_id
    """)
    if con.execute("SELECT count(*) FROM forecast_features").fetchone()[0] == 0:
        raise ValueError("历史不足 168 小时，无法构建一周季节滞后特征。")
    null_features = con.execute(
        "SELECT count(*) FROM forecast_features WHERE "
        + " OR ".join(f"{name} IS NULL OR NOT isfinite({name})" for name in FEATURE_COLUMNS)
    ).fetchone()[0]
    if null_features:
        raise ValueError("预测特征包含缺失或非有限数值。")


def split_boundaries(month: str, config: ForecastConfig = FORECAST_CONFIG) -> dict[str, datetime]:
    start, end = month_bounds(month)
    start_dt = datetime.combine(start, time())
    end_dt = datetime.combine(end, time())
    feature_start = start_dt + timedelta(hours=max(config.lags))
    eligible_days = (end_dt - feature_start).days
    train_days = int(eligible_days * config.train_fraction)
    remainder = eligible_days - train_days
    validation_days = remainder // 2
    test_days = remainder - validation_days
    if min(train_days, validation_days, test_days) < 2:
        raise ValueError("目标月份在预热后没有足够的训练、验证和测试天数。")
    train_end = feature_start + timedelta(days=train_days)
    validation_end = train_end + timedelta(days=validation_days)
    return {
        "feature_start": feature_start,
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": end_dt,
    }


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("评价数组必须非空且长度相同。")
    if (actual < 0).any() or (predicted < 0).any() or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("评价值必须是有限非负数。")
    error = np.abs(actual - predicted)
    denominator = float(actual.sum())
    return {
        "rows": int(actual.size),
        "actual_sum": denominator,
        "mae": float(error.mean()),
        "wape": float(error.sum() / denominator) if denominator else None,
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
    }


def _model(params: dict, config: ForecastConfig):
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=220, min_samples_leaf=30, early_stopping=False,
        random_state=config.random_state, **params,
    )


def _metric_rows(frame: pd.DataFrame, split: str, active_zones: set[int]) -> list[dict]:
    subset = frame[frame["split"] == split]
    rows = []
    for cohort, scoped in (("all", subset), ("train_active", subset[subset.zone_id.isin(active_zones)])):
        for model, column in (("lag_24", "lag_24_prediction"),
                              ("lag_168", "lag_168_prediction"),
                              ("hist_gradient_boosting_poisson", "model_prediction")):
            rows.append({"split": split, "cohort": cohort, "model": model,
                         **regression_metrics(scoped.target, scoped[column])})
    return rows


def build_forecast_assets(con: duckdb.DuckDBPyConnection, run: Path, month: str,
                          config: ForecastConfig = FORECAST_CONFIG) -> dict:
    create_forecast_features(con)
    bounds = split_boundaries(month, config)
    frame = con.execute("SELECT * FROM forecast_features ORDER BY target_hour, zone_id").fetchdf()
    frame["target_hour"] = pd.to_datetime(frame["target_hour"])
    frame["split"] = np.select(
        [frame.target_hour < bounds["train_end"], frame.target_hour < bounds["validation_end"]],
        ["train", "validation"], default="test",
    )
    split_counts = frame.groupby("split").size().to_dict()
    if set(split_counts) != {"train", "validation", "test"}:
        raise ValueError("时间切分为空或顺序无效。")
    matrix = frame.loc[:, FEATURE_COLUMNS].astype("float64")
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("模型矩阵包含缺失或非有限数值。")
    train = frame.split == "train"
    validation = frame.split == "validation"
    test = frame.split == "test"
    if frame.loc[train, "target"].sum() <= 0:
        raise ValueError("训练窗口的观测行程总数为零，无法拟合 Poisson 模型。")

    candidates = []
    selected_params = None
    selected_score = None
    selected_validation_prediction = None
    for params in MODEL_CANDIDATES:
        candidate = _model(params, config)
        started = perf_counter()
        candidate.fit(matrix.loc[train], frame.loc[train, "target"])
        prediction = candidate.predict(matrix.loc[validation])
        metrics = regression_metrics(frame.loc[validation, "target"], prediction)
        score = metrics["wape"] if metrics["wape"] is not None else metrics["mae"]
        candidates.append({"params": params, "validation": metrics,
                           "fit_seconds": round(perf_counter() - started, 6)})
        if selected_score is None or score < selected_score:
            selected_score = score
            selected_params = params
            selected_validation_prediction = prediction

    refit = _model(selected_params, config)
    refit_started = perf_counter()
    refit.fit(matrix.loc[train | validation], frame.loc[train | validation, "target"])
    test_prediction = refit.predict(matrix.loc[test])
    refit_seconds = perf_counter() - refit_started

    prediction_parts = []
    for mask, split, prediction in (
        (validation, "validation", selected_validation_prediction),
        (test, "test", test_prediction),
    ):
        part = frame.loc[mask, ["target_hour", "zone_id", "zone", "borough", "target", "lag_24", "lag_168"]].copy()
        part["split"] = split
        part["lag_24_prediction"] = part.pop("lag_24")
        part["lag_168_prediction"] = part.pop("lag_168")
        part["model_prediction"] = prediction
        prediction_parts.append(part)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    prediction_columns = ("lag_24_prediction", "lag_168_prediction", "model_prediction")
    if (predictions.loc[:, prediction_columns] < 0).any().any() or not np.isfinite(
            predictions.loc[:, prediction_columns].to_numpy()).all():
        raise AssertionError("模型产生了负数或非有限预测。")

    active_zones = set(frame.loc[train].groupby("zone_id")["target"].sum().loc[lambda x: x > 0].index)
    metrics = pd.DataFrame(_metric_rows(predictions, "validation", active_zones)
                           + _metric_rows(predictions, "test", active_zones))
    con.register("forecast_predictions_frame", predictions)
    con.register("forecast_metrics_frame", metrics)
    con.execute("CREATE OR REPLACE TABLE forecast_predictions AS SELECT * FROM forecast_predictions_frame")
    con.execute("CREATE OR REPLACE TABLE forecast_metrics AS SELECT * FROM forecast_metrics_frame")
    con.unregister("forecast_predictions_frame")
    con.unregister("forecast_metrics_frame")
    con.execute(f"COPY forecast_features TO {_literal(run / 'forecast_features.parquet')} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY forecast_predictions TO {_literal(run / 'forecast_predictions.parquet')} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY forecast_metrics TO {_literal(run / 'forecast_metrics.parquet')} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")
    joblib.dump({
        "model": refit, "feature_columns": FEATURE_COLUMNS, "config": asdict(config),
        "selected_params": selected_params, "trained_before": bounds["validation_end"].isoformat(),
        "sklearn_version": version("scikit-learn"),
    }, run / "forecast_model.joblib", compress=3)

    diagnostics = con.execute("""
        WITH zone_totals AS (SELECT zone_id, sum(trip_count) AS trips FROM zone_hour GROUP BY zone_id)
        SELECT (SELECT count(*) FROM zone_hour),
            (SELECT count(*) FROM zone_hour WHERE trip_count = 0),
            (SELECT count(DISTINCT zone_id) FROM zone_hour),
            (SELECT count(*) FROM zone_totals WHERE trips = 0)
    """).fetchone()
    test_model = metrics[(metrics.split == "test") & (metrics.cohort == "all") &
                         (metrics.model == "hist_gradient_boosting_poisson")].iloc[0]
    test_baseline = metrics[(metrics.split == "test") & (metrics.cohort == "all") &
                            (metrics.model == "lag_168")].iloc[0]
    improvement = None
    if (pd.notna(test_baseline.wape) and test_baseline.wape
            and pd.notna(test_model.wape)):
        improvement = float((test_baseline.wape - test_model.wape) / test_baseline.wape)
    metric_records = metrics.astype(object).where(pd.notna(metrics), None).to_dict(orient="records")
    report = {
        "status": "complete",
        "target": "每个上车区域在目标小时的默认清洗后观测行程数",
        "forecast_horizon_hours": config.horizon_hours,
        "evaluation_mode": "rolling one-hour-ahead; model fixed, previously observed outcomes update lag features",
        "config": asdict(config),
        "feature_columns": list(FEATURE_COLUMNS),
        "splits": {name: value.isoformat() for name, value in bounds.items()},
        "split_rows": {key: int(value) for key, value in split_counts.items()},
        "model": {
            "name": "HistGradientBoostingRegressor",
            "loss": "poisson",
            "selected_params": selected_params,
            "candidates": candidates,
            "refit_seconds": round(refit_seconds, 6),
            "random_state": config.random_state,
        },
        "metrics": metric_records,
        "test_wape_improvement_vs_lag_168": improvement,
        "diagnostics": {
            "zone_hour_rows": int(diagnostics[0]), "zero_rows": int(diagnostics[1]),
            "zero_fraction": diagnostics[1] / diagnostics[0], "zones": int(diagnostics[2]),
            "all_zero_zones": int(diagnostics[3]), "train_active_zones": len(active_zones),
        },
        "leakage_policy": [
            "样本以被预测的 target_hour 编号，所有统计窗口止于 1 PRECEDING。",
            "按时间顺序切分且不打乱；候选参数只比较验证集，测试集只在重拟合后评价一次。",
            "验证和测试采用滚动一步评价：模型参数保持固定，但下一时点可使用此前已经发生的真实观测。",
            "目标小时的 trip_count、has_observed_trip、费用和距离统计均不作为特征。",
        ],
        "limitations": [
            "当前单月结果是开发性评估，不代表跨月份或长期泛化能力。",
            "目标是已记录上车行程数，不是未满足的真实需求。",
            "模型尚未包含天气、节假日和事件；突发变化可能产生较大误差。",
            "TLC 本地墙上时间尚未解决夏令时重复或缺失小时问题。",
            "区域小时中大量零值；需同时查看总体、训练期活跃区域和分组指标。",
        ],
        "versions": {"scikit-learn": version("scikit-learn"), "joblib": version("joblib")},
    }
    write_json(run / "forecast_report.json", report)
    return report
