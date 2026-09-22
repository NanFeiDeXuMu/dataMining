"""UrbanPulse: traceable rule evaluation and leakage-safe forecasting.

Run: python -m streamlit run app.py
"""

from datetime import timedelta
from pathlib import Path
import json

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from urbanpulse import __version__
from urbanpulse.analytics import Explorer, active_run, available_months, insights
from urbanpulse.config import DEFAULT_RULES, DICTIONARY_URL, RULES, RULE_BY_KEY, SOURCE_PAGE, data_root, month_bounds
from urbanpulse.patterns import fold_rule_pairs, rule_coverage

st.set_page_config(page_title="UrbanPulse · 出行数据实验室", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1440px; padding-top: 4.5rem; padding-bottom: 3rem;}
h1 {letter-spacing: -.025em;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #dce6ed; border-radius: 10px; padding: 16px;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
.eyebrow {color:#087f8c; font-size:.78rem; letter-spacing:.15em; font-weight:600;}
.scope-note {color:#50687b; margin: .5rem 0 1.3rem;}
</style>""", unsafe_allow_html=True)

PALETTE = ["#087f8c", "#e18c3b", "#657b91"]


def chart(fig, height=330):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=12, t=30, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, Arial, sans-serif", color="#183043"),
        legend=dict(orientation="h", y=1.16, x=0, title=None),
        hoverlabel=dict(font_family="Microsoft YaHei, Arial, sans-serif"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e2eaf0", zerolinecolor="#cbd8e2")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def money(value):
    return "—" if value is None else f"${value:,.2f}"


@st.cache_data(max_entries=32, show_spinner=False)
def aggregates(run_dir, month, keys, view, start, end, zone, field):
    with Explorer(Path(run_dir) / "urbanpulse.duckdb", month) as ex:
        active_keys = keys if view == "处理后" else ()
        args = (start, end, zone)
        raw, clean = ex.stats((), *args), ex.stats(keys, *args)
        histogram, tails = ex.histogram(field, active_keys, *args)
        return {
            "raw": raw, "clean": clean, "quality": ex.quality(*args),
            "hourly": ex.hourly(active_keys, *args),
            "daily_raw": ex.daily((), *args), "daily_clean": ex.daily(keys, *args),
            "ranking": ex.ranking(active_keys, *args), "histogram": histogram, "tails": tails,
            "insights": insights(raw, clean, ex.hourly(keys, *args), month),
        }


@st.cache_data(max_entries=32, show_spinner=False)
def example_rows(run_dir, month, keys, start, end, zone, issue):
    with Explorer(Path(run_dir) / "urbanpulse.duckdb", month) as ex:
        return ex.sample(keys, start, end, zone, issue)


@st.cache_data(max_entries=32, show_spinner=False)
def pattern_rows(run_dir, month, support, confidence, lift, day_item, pickup_item, sort_by):
    with Explorer(Path(run_dir) / "urbanpulse.duckdb", month) as ex:
        return ex.pattern_rules(support, confidence, lift, (day_item, pickup_item), sort_by=sort_by)


@st.cache_data(max_entries=32, show_spinner=False)
def pattern_examples(run_dir, month, antecedent, consequent):
    with Explorer(Path(run_dir) / "urbanpulse.duckdb", month) as ex:
        return ex.pattern_evidence(antecedent, consequent), ex.pattern_sample(antecedent, consequent)


@st.cache_data(max_entries=32, show_spinner=False)
def forecast_rows(run_dir, month, split, zone):
    with Explorer(Path(run_dir) / "urbanpulse.duckdb", month) as ex:
        return {
            "metrics": ex.forecast_metrics(split, zone),
            "series": ex.forecast_series(split, zone),
            "errors": ex.forecast_errors(split, zone),
        }


st.markdown('<div class="eyebrow">URBANPULSE / DATA LAB 03</div>', unsafe_allow_html=True)
st.title("城市出行数据实验室")
st.caption("从可信数据走向可复核预测。数据质量 → 模式发现 → 时间切分 → 基线比较 → 下一小时预测")

root = data_root()
months = available_months(root)
if not months:
    st.info("还没有构建数据快照。请先在项目目录运行以下命令，再刷新页面。")
    st.code("python -m urbanpulse prepare --month 2025-01", language="shell")
    st.markdown(f"数据将从 [NYC TLC 官方网站]({SOURCE_PAGE}) 下载到本地，不上传文件。")
    st.stop()

with st.sidebar:
    st.header("分析设置")
    month = st.selectbox("数据快照", months, key="month")
    view = st.radio("当前视图", ["处理后", "原始"], horizontal=True, key="view")
    st.caption("切换只改变查看方式，绝不修改原始文件。")
    st.divider()
    with st.expander("调整清洗规则", expanded=False):
        if st.button("恢复默认清洗规则", width="stretch"):
            st.session_state["rules"] = list(DEFAULT_RULES)
        keys = tuple(st.multiselect("排除命中以下规则的记录", [r.key for r in RULES],
                                    default=list(DEFAULT_RULES), format_func=lambda k: RULE_BY_KEY[k].label, key="rules"))
        st.caption("“待核查”项默认不排除；勾选后观察统计变化。空规则 = 保留全部。")
    st.caption(f"当前排除 {len(keys)} 项规则；其余记录保留供分析。")
    st.divider()

try:
    run = active_run(root, month)
    metadata = json.loads((run / "report.json").read_text(encoding="utf-8"))
    with Explorer(run / "urbanpulse.duckdb", month) as ex:
        zones = ex.zones()
except (OSError, ValueError, KeyError, duckdb.Error) as exc:
    st.error(f"快照无法读取：{exc}")
    st.code(f"python -m urbanpulse build --month {month}")
    st.stop()

first, next_month = month_bounds(month)
last = next_month - timedelta(days=1)
zone_labels = {int(row.zone_id): f"{int(row.zone_id)} · {row.zone}" for row in zones.itertuples()}
with st.sidebar:
    zone = st.selectbox("上车区域", [None] + list(zone_labels),
                        format_func=lambda k: "全部区域" if k is None else zone_labels[k], key="zone")
    date_filter = st.checkbox("限定上车日期", key="date_filter")
    start, end = None, None
    if date_filter:
        selected_dates = st.date_input("日期范围", (first, last), min_value=first, max_value=last, key=f"dates_{month}")
        if len(selected_dates) != 2:
            st.info("请选择完整的开始和结束日期。")
            st.stop()
        start, end = selected_dates
    field = st.selectbox("分布查看字段", ["amount", "distance_miles", "duration_min"],
                         format_func=lambda x: {"amount": "总额（USD）", "distance_miles": "距离（英里）", "duration_min": "时长（分钟）"}[x], key="field")
    st.divider()
    st.caption(f"v{__version__} · 本地分析 · 规则评价与证据 + 下一小时预测 / 无 LLM")
    st.markdown(f"[数据来源]({SOURCE_PAGE}) · [官方字段字典]({DICTIONARY_URL})")

with st.spinner("正在查询完整数据快照…"):
    payload = aggregates(str(run), month, keys, view, start, end, zone, field)
raw, clean = payload["raw"], payload["clean"]
current = clean if view == "处理后" else raw
scope = f"{start} 至 {end}" if date_filter else "来源文件的全部记录（含待核查时间）"
st.caption(f"{month} Yellow Taxi · 全文件 {metadata['raw']['records']:,} 条 · 当前窗口：{scope} · {zone_labels.get(zone, '全部区域')}")
cols = st.columns(4)
cols[0].metric("当前视图记录数", f"{current['records']:,}", help="观测到的行程记录，不代表全部真实需求。")
cols[1].metric("总额均值 · USD", money(current["mean_amount"]), help=f"仅有限总额；分母 {current['amount_n']:,} 条。不含现金小费。")
cols[2].metric("总额中位数 · USD", money(current["median_amount"]), help="将有效费用排序后取中间位置；更不易受极端值影响。")
cols[3].metric("所选规则保留率", f"{clean['records']/raw['records']:.2%}" if raw["records"] else "—",
               help="处理后记录数 / 同一窗口原始记录数；不代表数据正确率。")

forecast_export = None
overview, quality_tab, dictionary_tab, lesson_tab, pattern_tab, forecast_tab = st.tabs(
    ["数据概览", "质量体检", "字段字典", "学习与复现", "模式发现", "需求预测"])

with overview:
    if not current["records"]:
        st.warning("当前视图没有记录。试试取消日期/区域筛选，或减少排除规则。")
    st.subheader("一天中，出行何时更集中？")
    st.caption("按目标月所选日期的工作日/周末日历天数归一化，避免两类天数不同造成误导；仅使用目标月有效上车时间。")
    hourly = payload["hourly"]
    chart(px.line(hourly, x="hour", y="daily_average", color="day_type", markers=True,
                  color_discrete_sequence=PALETTE, labels={"hour": "小时（TLC 本地时间）", "daily_average": "日均观测订单数（条/日）", "day_type": "日期类型", "calendar_days": "日历天数"},
                  hover_data=["calendar_days", "trips"]))
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("清洗前后，时间趋势是否变化？")
        daily = pd.concat([payload["daily_raw"].assign(version="原始"), payload["daily_clean"].assign(version="处理后")])
        chart(px.line(daily, x="date", y="trips", color="version", color_discrete_map={"原始": "#8b9dae", "处理后": "#087f8c"},
                      labels={"date": "日期", "trips": "观测订单数（条）", "version": "视图"}), 310)
    with right:
        st.subheader("均值之外，观察分布与长尾")
        labels = {"amount": "总额（USD）", "distance_miles": "距离（英里）", "duration_min": "时长（分钟）"}
        chart(px.bar(payload["histogram"], x="bin_start", y="trips", color_discrete_sequence=PALETTE,
                     labels={"bin_start": labels[field] + " · 分箱左端点", "trips": "记录数（条）"}), 310)
        tail = payload["tails"]
        st.caption(f"显示 [{tail['low']}, {tail['high']})，箱宽 {tail['width']}。范围外：低于下界 {tail['below']:,} 条；不低于上界 {tail['above']:,} 条；缺失/非有限 {tail['missing']:,} 条。仅限制显示，不改变统计。")
    st.subheader("上车区域 Top 15")
    rank = payload["ranking"].sort_values("trips")
    chart(px.bar(rank, x="trips", y="zone", orientation="h", color_discrete_sequence=PALETTE,
                 hover_data=["zone_id", "borough"], labels={"trips": "观测订单数（条）", "zone": "上车区域", "borough": "行政区", "zone_id": "区域 ID"}), 410)
    st.subheader("三条可复核观察")
    st.caption("以下始终比较同一筛选窗口的原始与处理后结果，由确定性代码计算，不是 AI 生成的因果解释。")
    for number, line in enumerate(payload["insights"], 1):
        st.markdown(f"**{number:02d}**　{line}")

with quality_tab:
    st.subheader("一条记录，可能命中多个问题")
    st.caption("此表基于当前筛选窗口的原始记录，不随“当前视图”切换而隐藏问题。各规则计数不能相加。")
    checks = payload["quality"].copy()
    checks["当前排除"] = checks["key"].isin(keys)
    st.dataframe(checks.drop(columns=["key", "默认排除"]), hide_index=True, width="stretch",
                 column_config={"占原始窗口比例": st.column_config.NumberColumn(format="percent")})
    st.info(f"同一窗口：原始 {raw['records']:,} → 所选规则保留 {clean['records']:,}。负费用可能是退款/冲正；高费用可能是真实长途。此处的标记不是欺诈标签。")
    issue = st.selectbox("查看命中某规则的原始记录", [None] + [r.key for r in RULES],
                         format_func=lambda x: "全部记录" if x is None else RULE_BY_KEY[x].label, key="issue")
    examples = example_rows(str(run), month, (), start, end, zone, issue)
    st.caption("最多展示 100 条，按原文件行号排序；这是排查示例，不是随机样本。原始记录号便于回溯。")
    st.dataframe(examples, hide_index=True, width="stretch")
    st.download_button("下载当前排查示例 CSV（最多 100 条）", examples.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"urbanpulse_{month}_inspection.csv", mime="text/csv")

with dictionary_tab:
    st.subheader("数据身份证")
    st.markdown("**记录单位**：一条出租车行程。**距离**：英里。**货币**：美元。**时间**：TLC 本地墙上时间。")
    st.caption("以下缺失比例基于来源文件的全部记录，与侧栏筛选无关。原始字段全部保留，派生列另行存储。")
    st.dataframe(pd.DataFrame(metadata["schema"]), hide_index=True, width="stretch",
                 column_config={"缺失比例": st.column_config.NumberColumn(format="percent")})
    st.warning("区域 ID、支付方式和费率都是类别编码。不能因为存储为整数就计算它们的均值，或把 ID 的差当作地理距离。")
    with st.expander("来源、校验值与当前版本"):
        st.json({"month": month, "built_at_utc": metadata["built_at_utc"], "source": metadata["source"], "versions": metadata["versions"]})

with lesson_tab:
    st.subheader("第一课练习：先提问，再操作，最后解释")
    for title, body in [
        ("01 · 数据理解", "选择三个字段，分别说明它们的语义类型、单位和是否允许缺失。解释区域 ID 为什么不适合取均值。"),
        ("02 · 清洗决策", "恢复默认规则，记录保留率；再排除零距离、负费用和高额费用。每次只改一项，比较均值、中位数与区域排名。"),
        ("03 · 描述与评价", "选择同一日期窗口比较工作日和周末，解释为什么要使用日均订单量。写出三条发现，并标明时间范围、样本量和局限。"),
        ("04 · 可复现性", "重新构建同一月份，验证原始 SHA-256、清洗后记录数和 zone_hour 订单总数一致。增加新月份时保持同一套规则。"),
    ]:
        with st.expander(title, expanded=title.startswith("01")):
            st.write(body)
    st.subheader("解释边界")
    for limit in metadata["limitations"]:
        st.markdown(f"- {limit}")
    st.subheader("复现与输出")
    st.code(f"python -m urbanpulse prepare --month {month}\npython -m streamlit run app.py\npython -m pytest -q", language="shell")
    st.caption("清洗后的完整行程与区域-小时特征由批处理生成。页面规则是临时探索，不覆盖磁盘上的默认规则产物。")
    st.code(str(run), language=None)
    st.markdown("输出包含清洗数据、区域小时特征、事务表、频繁项集、关联规则和可复核报告。")

with pattern_tab:
    st.subheader("哪些出行属性经常共同出现？")
    pattern_meta = metadata.get("patterns")
    if not pattern_meta:
        st.warning(f"当前快照没有模式结果。请运行 `python -m urbanpulse build --month {month}` 重新生成。")
    else:
        cfg = pattern_meta["config"]
        coverage = rule_coverage(pattern_meta)
        st.caption("模式预计算于整月默认清洗结果，不随侧栏日期、区域或临时清洗规则变化。每条行程表示为 8 个类别项；关联不代表因果。")
        summary = st.columns(4)
        summary[0].metric("行程事务", f"{pattern_meta['transaction_count']:,}")
        summary[1].metric("不同事务篮子", f"{pattern_meta['unique_baskets']:,}")
        summary[2].metric("频繁项集", f"{pattern_meta['frequent_itemsets']:,}")
        summary[3].metric("保存的方向规则", f"{pattern_meta['rules']:,}")
        st.caption(f"挖掘范围：联合支持度 ≥ {coverage['min_support']:.2%}，项集长度 ≤ {coverage['max_itemset_size']}。"
                   "规则筛选不改变全月分母，也不会恢复范围外的低频模式。")
        if not coverage["complete_within_itemsets"]:
            st.warning(f"旧快照已预过滤：置信度 ≥ {coverage['min_confidence']:.2f}、提升度 ≥ {coverage['min_lift']:.2f}。"
                       "未保存的规则不能在此恢复；重建快照后可降低展示阈值。双向置信度仍可由已有计数复算。")
        # A different immutable snapshot may have different coverage and defaults.
        if st.session_state.get("pattern_snapshot") != str(run):
            for field in ("support", "confidence", "lift"):
                st.session_state[f"pattern_{field}"] = float(cfg[f"min_{field}"])
            st.session_state["pattern_snapshot"] = str(run)
        filters = st.columns(3)
        min_support = filters[0].number_input("最小支持度（0.01 = 1%）", min_value=float(coverage["min_support"]),
                                              max_value=1.0, step=0.01, format="%.4f", key="pattern_support")
        min_confidence = filters[1].number_input("最小置信度", min_value=float(coverage["min_confidence"]),
                                                 max_value=1.0, step=0.05, format="%.2f", key="pattern_confidence")
        min_lift = filters[2].number_input("最小提升度", min_value=float(coverage["min_lift"]),
                                           step=0.05, format="%.2f", key="pattern_lift")
        required = st.columns(2)
        day_item = required[0].selectbox("规则必须包含日期类型", [None, "日期=工作日", "日期=周末"],
                                         format_func=lambda value: "不限" if value is None else value.split("=", 1)[1],
                                         key="pattern_day")
        borough_items = [f"上车行政区={name}" for name in sorted(
            set(zones["borough"].dropna()) - {"Unknown", "N/A"})]
        pickup_item = required[1].selectbox("规则必须包含上车行政区", [None] + borough_items,
                                            format_func=lambda value: "不限" if value is None else value.split("=", 1)[1],
                                            key="pattern_pickup")
        browsing = st.columns(2)
        sort_by = browsing[0].selectbox("规则排序", ["lift", "support", "kulc"],
                                        format_func={"lift": "提升度", "support": "支持度", "kulc": "Kulc 双向重合"}.get,
                                        key="pattern_sort")
        page_size = browsing[1].selectbox("每页属性组", [25, 50, 100, 200], index=1, key="pattern_page_size")
        rules_frame = pattern_rows(str(run), month, min_support, min_confidence, min_lift,
                                   day_item, pickup_item, sort_by)
        pairs = fold_rule_pairs(rules_frame)
        signature = (str(run), min_support, min_confidence, min_lift, day_item, pickup_item, sort_by, page_size)
        if st.session_state.get("pattern_filter_signature") != signature:
            st.session_state["pattern_page"] = 1
            st.session_state["pattern_filter_signature"] = signature
        st.caption(f"完整匹配：{len(rules_frame):,} 条方向规则，折叠为 {len(pairs):,} 组 A/B 关系。"
                   "每组保留排序最靠前的匹配方向；展开可核对两个方向的置信度。")
        export = rules_frame.assign(
            snapshot=run.name, month=month, data_scope="全月默认清洗事务",
            coverage_min_support=coverage["min_support"], coverage_max_itemset_size=coverage["max_itemset_size"],
            coverage_min_confidence=coverage["min_confidence"], coverage_min_lift=coverage["min_lift"],
            complete_within_itemsets=coverage["complete_within_itemsets"],
            filter_support=min_support, filter_confidence=min_confidence, filter_lift=min_lift,
            required_items="；".join(item for item in (day_item, pickup_item) if item))
        st.download_button("导出全部匹配方向规则（CSV）", export.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"urbanpulse-rules-{month}-{run.name}.csv", mime="text/csv", key="pattern_export")
        if rules_frame.empty:
            st.info("已保存范围内没有匹配规则；这不证明范围外不存在关联。可调整展示阈值或规则必须包含的项。")
        else:
            pages = (len(pairs) + page_size - 1) // page_size
            page = st.number_input("结果页", min_value=1, max_value=pages, step=1, key="pattern_page")
            shown = pairs.iloc[(page - 1) * page_size:page * page_size]
            st.caption(f"第 {page}/{pages} 页：展示 {len(shown)} / {len(pairs):,} 组关系；散点图与本页一致。CSV 包含全部匹配方向规则。")
            chart(px.scatter(shown, x="support", y="kulc", size="support_count", color="lift",
                             hover_data=["antecedent", "consequent"],
                             labels={"support": "支持度", "kulc": "Kulc 双向重合", "lift": "提升度",
                                     "support_count": "共同出现记录数", "antecedent": "前件", "consequent": "后件"}), 360)
            display = shown[["antecedent", "consequent", "matched_directions", "support_count", "support",
                             "confidence", "reverse_confidence", "lift", "kulc", "imbalance_ratio"]].rename(
                columns={"antecedent": "A组", "consequent": "B组", "matched_directions": "命中方向数",
                         "support_count": "共同记录数", "support": "支持度", "confidence": "A→B置信度",
                         "reverse_confidence": "B→A置信度", "lift": "提升度", "kulc": "Kulc", "imbalance_ratio": "IR"})
            st.dataframe(display, hide_index=True, width="stretch",
                         column_config={name: st.column_config.NumberColumn(format="percent")
                                        for name in ("支持度", "A→B置信度", "B→A置信度")})
            rule_options = list(shown[["antecedent", "consequent"]].itertuples(index=False, name=None))
            if st.session_state.get("pattern_rule") not in rule_options:
                st.session_state["pattern_rule"] = rule_options[0]
            selected = st.selectbox("选择一组关系查看双向规则与证据", rule_options,
                                    format_func=lambda rule: f"{rule[0]} ↔ {rule[1]}", key="pattern_rule")
            counts, evidence = pattern_examples(str(run), month, *selected)
            with st.expander("双向规则与计数证据", expanded=True):
                st.write(f"A：{selected[0]}　｜　B：{selected[1]}")
                st.caption(f"独立复算：{month} 整月默认清洗事务，N = {counts['transaction_count']:,}。"
                           "四格计数直接来自完整事务表，并已核对保存的规则计数。")
                st.dataframe(pd.DataFrame({
                    "A是否出现": ["包含 A", "不包含 A", "合计"],
                    "包含 B": [counts["n11"], counts["n01"], counts["consequent_count"]],
                    "不包含 B": [counts["n10"], counts["n00"], counts["transaction_count"] - counts["consequent_count"]],
                    "合计": [counts["antecedent_count"], counts["transaction_count"] - counts["antecedent_count"], counts["transaction_count"]],
                }), hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame({
                    "方向": ["A → B", "B → A"],
                    "置信度": [counts["confidence"], counts["reverse_confidence"]],
                    "后件基准比例": [counts["consequent_support"], counts["antecedent_support"]],
                    "通过当前数值阈值": [counts["support"] >= min_support and c >= min_confidence and counts["lift"] >= min_lift
                                       for c in (counts["confidence"], counts["reverse_confidence"])],
                }), hide_index=True, width="stretch",
                    column_config={name: st.column_config.NumberColumn(format="percent") for name in ("置信度", "后件基准比例")})
                evaluation = st.columns(4)
                for column, label, field in zip(evaluation, ["支持度", "提升度", "Kulc 双向重合", "IR 不平衡比"],
                                                ["support", "lift", "kulc", "imbalance_ratio"]):
                    value = counts[field]
                    column.metric(label, "不可计算" if value is None else (f"{value:.2%}" if field == "support" else f"{value:.4f}"))
                st.caption("两个方向的置信度都由计数计算；没有通过筛选的方向也会显示，但不会因此加入匹配列表。"
                           "Kulc 是双向置信度的均值；IR 越接近 1，A、B 的边际计数越不平衡。")
                st.caption(f"相对于 A/B 的空事务有 {counts['n00']:,} 条，即两者均未出现，不是缺失记录。"
                           "增加这类事务会影响 lift，但不改变 Kulc 和 IR。高 lift 不等于因果或欺诈证据。")
                if counts["metric_notes"]:
                    st.info(counts["metric_notes"])
                st.caption(f"以下是按 record_id 排序的前 {len(evidence)} 条共同出现行程（最多 100 条，非随机样本）；"
                           f"共同计数共 {counts['support_count']:,} 条。")
                st.dataframe(evidence, hide_index=True, width="stretch")
                with st.expander("公式与补充指标"):
                    st.code("support = nAB / N\nconfidence(A→B) = nAB / nA\nconfidence(B→A) = nAB / nB\n"
                            "lift = N × nAB / (nA × nB)\nKulc = (nAB/nA + nAB/nB) / 2\n"
                            "IR = |nA - nB| / (nA + nB - nAB)\nJaccard = nAB / (nA + nB - nAB)\n"
                            "cosine = nAB / √(nA × nB)", language="text")
                    st.write(f"Jaccard：{counts['jaccard']:.4f}；cosine：{counts['cosine']:.4f}。分母为零时返回不可计算，不用 0 代替。")
        with st.expander("算法与口径"):
            algorithms = pattern_meta["algorithms"]
            st.write(f"Apriori 与 FP-Growth 频繁项集一致：{'是' if algorithms['identical'] else '否'}。最大项集长度：{cfg['max_itemset_size']}。")
            st.write(f"本次构建用时：Apriori {algorithms['apriori_seconds']:.3f} 秒，FP-Growth {algorithms['fpgrowth_seconds']:.3f} 秒；单次用时仅作实现观察，不是严格基准测试。")
            st.write("事务字段：" + "、".join(pattern_meta["fields"]))
            for limitation in pattern_meta["limitations"]:
                st.markdown(f"- {limitation}")

with forecast_tab:
    st.subheader("下一小时，各区域会有多少已记录上车行程？")
    forecast_meta = metadata.get("forecast")
    if not forecast_meta:
        st.warning(f"当前快照没有预测结果。请运行 `python -m urbanpulse build --month {month}` 重新生成。")
    else:
        st.caption("样本以被预测的目标小时编号；所有统计特征止于前一小时。这里预测的是已记录行程数，不是未满足需求。侧栏日期和临时清洗规则不改变离线实验。")
        split = st.radio("评估分区", ["validation", "test"], horizontal=True,
                         format_func=lambda value: {"validation": "验证集", "test": "测试集"}[value],
                         key="forecast_split")
        forecast = forecast_rows(str(run), month, split, zone)
        metrics = forecast["metrics"]
        forecast_export = {
            "split": split, "pickup_zone_id": zone,
            "metrics": metrics.astype(object).where(pd.notna(metrics), None).to_dict(orient="records"),
        }
        model_row = metrics[metrics.model == "hist_gradient_boosting_poisson"].iloc[0]
        baseline_row = metrics[metrics.model == "lag_168"].iloc[0]
        improvement = ((baseline_row.wape - model_row.wape) / baseline_row.wape
                       if pd.notna(baseline_row.wape) and baseline_row.wape else None)
        cards = st.columns(4)
        cards[0].metric("模型 MAE", f"{model_row.mae:.2f}", help="平均每个区域小时相差的观测行程数。")
        cards[1].metric("模型 WAPE", f"{model_row.wape:.2%}" if pd.notna(model_row.wape) else "—",
                        help="绝对误差总和 / 实际行程总和；零分母时不计算。")
        cards[2].metric("模型 RMSE", f"{model_row.rmse:.2f}", help="对高峰时段的大误差惩罚更强。")
        cards[3].metric("相对 lag-168", f"{improvement:+.2%}" if improvement is not None else "—",
                        help="正值表示模型 WAPE 低于上周同小时基线。")
        model_names = {"lag_24": "昨日同小时", "lag_168": "上周同小时",
                       "hist_gradient_boosting_poisson": "Poisson 梯度提升"}
        comparison = metrics.copy()
        comparison["方法"] = comparison.model.map(model_names)
        st.dataframe(comparison[["方法", "rows", "actual_sum", "mae", "wape", "rmse"]].rename(
            columns={"rows": "评价行数", "actual_sum": "实际行程总数", "mae": "MAE",
                     "wape": "WAPE", "rmse": "RMSE"}), hide_index=True, width="stretch",
            column_config={"WAPE": st.column_config.NumberColumn(format="percent")})
        series = forecast["series"].rename(columns={"actual": "实际",
                                                     "lag_168_prediction": "上周同小时",
                                                     "model_prediction": "Poisson 梯度提升"})
        long_series = series.melt("target_hour", var_name="序列", value_name="观测行程数")
        scope_name = zone_labels.get(zone, "全部区域汇总")
        st.subheader(f"逐小时预测 · {scope_name}")
        chart(px.line(long_series, x="target_hour", y="观测行程数", color="序列",
                      color_discrete_sequence=["#183043", "#8b9dae", "#087f8c"],
                      labels={"target_hour": "目标小时（TLC 本地时间）"}), 380)
        st.subheader("模型绝对误差最大的 100 个区域小时")
        st.caption("按模型绝对误差排序用于排查，不是随机样本；切换侧栏区域可以查看单一区域。")
        st.dataframe(forecast["errors"], hide_index=True, width="stretch")
        with st.expander("实验设计与解释边界"):
            splits = forecast_meta["splits"]
            st.write(f"训练截止：{splits['train_end']}；验证截止：{splits['validation_end']}；测试截止：{splits['test_end']}。按时间切分，不打乱样本。")
            st.write("特征：" + "、".join(forecast_meta["feature_columns"]))
            selected_params = forecast_meta["model"]["selected_params"]
            st.write(f"模型参数只按验证集选择：{selected_params}；固定随机种子 {forecast_meta['model']['random_state']}。")
            st.write("采用滚动一步评价：模型在每个分区内保持固定，但预测下一小时可以使用此前已经发生的真实观测更新滞后特征。")
            for limitation in forecast_meta["limitations"]:
                st.markdown(f"- {limitation}")

report_payload = {"month": month, "view": view, "rules": list(keys),
                  "start_date": str(start) if start else None, "end_date": str(end) if end else None,
                  "pickup_zone_id": zone, "raw": raw, "clean": clean,
                  "insights": payload["insights"], "source": metadata["source"],
                  "forecast": forecast_export, "limitations": metadata["limitations"]}
st.divider()
st.download_button("导出当前窗口分析报告 JSON", json.dumps(report_payload, ensure_ascii=False, indent=2, allow_nan=False),
                   file_name=f"urbanpulse_{month}_analysis.json", mime="application/json")
st.caption(f"UrbanPulse v{__version__} · 规则评价可回溯计数证据；预测坚持严格时间切分。")
