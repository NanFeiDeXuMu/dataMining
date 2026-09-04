from datetime import date

from streamlit.testing.v1 import AppTest

from urbanpulse.config import DEFAULT_RULES, ROOT


def create_app(monkeypatch, dataset):
    monkeypatch.setenv("URBANPULSE_DATA_DIR", str(dataset[0]))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    return app


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
    rules = app.dataframe[3].value
    assert not rules.empty and (rules["支持度"] >= 0.01).all()
    assert app.selectbox(key="pattern_rule").value
    assert not app.dataframe[4].value.empty
    app.selectbox(key="pattern_day").set_value("日期=工作日").run()
    assert not app.exception
    filtered_rules = app.dataframe[3].value
    combined_items = filtered_rules["前件"] + "；" + filtered_rules["后件"]
    assert combined_items.str.contains("日期=工作日", regex=False).all()
    app.slider(key="pattern_support").set_value(0.2).run()
    assert not app.exception
    if len(app.dataframe) > 3:
        assert (app.dataframe[3].value["支持度"] >= 0.2).all()


def test_no_data_has_setup_instructions(monkeypatch, tmp_path):
    monkeypatch.setenv("URBANPULSE_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert "prepare" in app.code[0].value
