from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from urbanpulse.analytics import Explorer
from urbanpulse.config import DEFAULT_RULES, ROOT


def create_app(monkeypatch, dataset):
    monkeypatch.setenv("URBANPULSE_DATA_DIR", str(dataset[0]))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    return app


def tables_with(app, *columns):
    required = set(columns)
    return [element.value for element in app.dataframe if required.issubset(element.value.columns)]


def test_initial_and_view_switch(monkeypatch, dataset):
    app = create_app(monkeypatch, dataset)
    assert app.metric[0].value == "11"
    app.radio(key="view").set_value("原始").run()
    assert not app.exception and app.metric[0].value == "22"
    app.radio(key="view").set_value("处理后").run()
    assert app.metric[0].value == "11"


def test_rule_changes_and_reset(monkeypatch, dataset):
    app = create_app(monkeypatch, dataset)
    app.multiselect(key="rules").set_value(list(DEFAULT_RULES) + ["negative_amount"]).run()
    assert not app.exception and app.metric[0].value == "10"
    app.multiselect(key="rules").set_value([]).run()
    assert not app.exception and app.metric[0].value == "22"
    app.button[0].click().run()
    assert not app.exception and app.metric[0].value == "11"


def test_empty_filter_and_date_range(monkeypatch, dataset):
    app = create_app(monkeypatch, dataset)
    app.selectbox(key="zone").set_value(3).run()
    assert not app.exception and app.metric[0].value == "0"
    app.selectbox(key="zone").set_value(None).run()
    app.checkbox(key="date_filter").check().run()
    app.date_input(key="dates_2025-01").set_value((date(2025, 1, 31), date(2025, 1, 31))).run()
    assert not app.exception and app.metric[0].value == "1"


def test_distribution_and_inspection(monkeypatch, dataset):
    app = create_app(monkeypatch, dataset)
    for field in ("distance_miles", "duration_min", "amount"):
        app.selectbox(key="field").set_value(field).run()
        assert not app.exception
    app.selectbox(key="issue").set_value("negative_amount").run()
    assert not app.exception
    assert len(app.dataframe[1].value) == 1


def test_pattern_filters_and_evidence(monkeypatch, dataset):
    app = create_app(monkeypatch, dataset)
    assert app.metric[4].value == "11"
    assert app.metric[6].value
    rules = tables_with(app, "A组", "B组", "Kulc", "IR")[0]
    assert not rules.empty and (rules["支持度"] >= 0.01).all()
    assert app.selectbox(key="pattern_rule").value
    assert not tables_with(app, "record_id", "items")[0].empty
    contingency = tables_with(app, "A是否出现", "包含 B", "不包含 B", "合计")[0]
    assert contingency.iloc[-1]["合计"] == 11
    assert contingency.iloc[:2][["包含 B", "不包含 B"]].to_numpy().sum() == 11
    app.selectbox(key="pattern_day").set_value("日期=工作日").run()
    assert not app.exception
    filtered_rules = tables_with(app, "A组", "B组", "Kulc", "IR")[0]
    combined_items = filtered_rules["A组"] + "；" + filtered_rules["B组"]
    assert combined_items.str.contains("日期=工作日", regex=False).all()
    app.number_input(key="pattern_support").set_value(0.2).run()
    assert not app.exception
    filtered_tables = tables_with(app, "支持度", "A→B置信度", "提升度")
    if filtered_tables:
        assert (filtered_tables[0]["支持度"] >= 0.2).all()


def test_pattern_pagination_sort_and_empty_results(monkeypatch, dataset):
    app = create_app(monkeypatch, dataset)
    app.number_input(key="pattern_confidence").set_value(0.0)
    app.number_input(key="pattern_lift").set_value(0.0)
    app.selectbox(key="pattern_page_size").set_value(25).run()
    assert not app.exception
    first = tables_with(app, "A组", "B组", "Kulc")[0]
    assert len(first) == 25 and (first["命中方向数"] == 2).all()
    app.number_input(key="pattern_page").set_value(2).run()
    assert not app.exception
    second = tables_with(app, "A组", "B组", "Kulc")[0]
    assert not first.equals(second)
    app.selectbox(key="pattern_sort").set_value("kulc").run()
    assert not app.exception
    assert app.number_input(key="pattern_page").value == 1
    assert tables_with(app, "A组", "B组", "Kulc")[0]["Kulc"].is_monotonic_decreasing
    app.number_input(key="pattern_lift").set_value(1000.0).run()
    assert not app.exception
    assert any("已保存范围内没有匹配规则" in message.value for message in app.info)
    assert not tables_with(app, "A组", "B组", "Kulc")


def test_pattern_export_includes_every_matching_direction_and_scope(monkeypatch, dataset):
    captured = {}
    original = st.download_button

    def capture(label, data, **kwargs):
        if kwargs.get("key") == "pattern_export":
            captured["csv"] = data
        return original(label, data, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    app = create_app(monkeypatch, dataset)
    app.number_input(key="pattern_confidence").set_value(0.0)
    app.number_input(key="pattern_lift").set_value(0.0).run()
    assert not app.exception
    exported = pd.read_csv(BytesIO(captured["csv"]))
    with Explorer(dataset[1] / "urbanpulse.duckdb", "2025-01") as ex:
        expected = ex.pattern_rules(min_confidence=0, min_lift=0)
    assert len(exported) == len(expected) > 200
    assert exported.snapshot.eq(dataset[1].name).all()
    assert exported.transaction_count.eq(11).all()
    assert exported.data_scope.eq("全月默认清洗事务").all()
    assert exported.complete_within_itemsets.all()


def test_legacy_pattern_page_explains_coverage_and_recounts(monkeypatch, legacy_dataset):
    app = create_app(monkeypatch, legacy_dataset)
    assert any("旧快照已预过滤" in message.value for message in app.warning)
    assert app.number_input(key="pattern_confidence").value == 0.35
    assert app.number_input(key="pattern_lift").value == 1.05
    assert not tables_with(app, "A组", "B组", "Kulc", "IR")[0].empty
    assert len(tables_with(app, "方向", "置信度", "后件基准比例")[0]) == 2
    assert tables_with(app, "A是否出现", "合计")[0].iloc[-1]["合计"] == 11


def test_forecast_comparison_and_split_switch(monkeypatch, dataset):
    app = create_app(monkeypatch, dataset)
    assert app.radio(key="forecast_split").value == "validation"
    comparisons = tables_with(app, "方法", "MAE", "WAPE", "RMSE")
    errors = tables_with(app, "model_prediction", "absolute_error")
    assert len(comparisons[0]) == 3
    assert not errors[0].empty
    app.radio(key="forecast_split").set_value("test").run()
    assert not app.exception
    assert app.radio(key="forecast_split").value == "test"


def test_no_data_has_setup_instructions(monkeypatch, tmp_path):
    monkeypatch.setenv("URBANPULSE_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert "prepare" in app.code[0].value
