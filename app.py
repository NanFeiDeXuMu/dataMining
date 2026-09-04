"""UrbanPulse v0.2: data quality, exploration and association patterns.

Run: python -m streamlit run app.py
"""

from datetime import timedelta
from pathlib import Path
import json

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from urbanpulse.analytics import Explorer, active_run, available_months, insights
from urbanpulse.config import DEFAULT_RULES, DICTIONARY_URL, RULES, RULE_BY_KEY, SOURCE_PAGE, data_root, month_bounds

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
def pattern_rows(run_dir, month, support, confidence, lift, day_item, pickup_item):
    with Explorer(Path(run_dir) / "urbanpulse.duckdb", month) as ex:
        return ex.pattern_rules(support, confidence, lift, (day_item, pickup_item))


@st.cache_data(max_entries=32, show_spinner=False)
def pattern_examples(run_dir, month, antecedent, consequent):
    with Explorer(Path(run_dir) / "urbanpulse.duckdb", month) as ex:
        return ex.pattern_sample(antecedent, consequent)


st.markdown('<div class="eyebrow">URBANPULSE / DATA LAB 02</div>', unsafe_allow_html=True)
st.title("城市出行数据实验室")
st.caption("先理解数据，再发现模式。数据质量 → 描述统计 → 频繁项集 → 关联规则 → 证据核查")

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
    st.caption("v0.2 · 本地分析 · 关联规则 / 无 LLM")
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

overview, quality_tab, dictionary_tab, lesson_tab, pattern_tab = st.tabs(
    ["数据概览", "质量体检", "字段字典", "学习与复现", "模式发现"])

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
        st.warning(f"当前是 v0.1 快照。请运行 `python -m urbanpulse build --month {month}` 生成模式结果。")
    else:
        cfg = pattern_meta["config"]
        st.caption("模式预计算于整月默认清洗结果，不随侧栏日期、区域或临时清洗规则变化。每条行程表示为 8 个类别项；关联不代表因果。")
        summary = st.columns(4)
        summary[0].metric("行程事务", f"{pattern_meta['transaction_count']:,}")
        summary[1].metric("不同事务篮子", f"{pattern_meta['unique_baskets']:,}")
        summary[2].metric("频繁项集", f"{pattern_meta['frequent_itemsets']:,}")
        summary[3].metric("候选规则", f"{pattern_meta['rules']:,}")
        filters = st.columns(3)
        min_support = filters[0].slider("最小支持度", float(cfg["min_support"]), 0.20,
                                        float(cfg["min_support"]), 0.01, format="%.2f", key="pattern_support")
        min_confidence = filters[1].slider("最小置信度", float(cfg["min_confidence"]), 1.0,
                                           float(cfg["min_confidence"]), 0.05, format="%.2f", key="pattern_confidence")
        min_lift = filters[2].slider("最小提升度", float(cfg["min_lift"]), 3.0,
                                     float(cfg["min_lift"]), 0.05, format="%.2f", key="pattern_lift")
        required = st.columns(2)
        day_item = required[0].selectbox("规则必须包含日期类型", [None, "日期=工作日", "日期=周末"],
                                         format_func=lambda value: "不限" if value is None else value.split("=", 1)[1],
                                         key="pattern_day")
        borough_items = [f"上车行政区={name}" for name in sorted(
            set(zones["borough"].dropna()) - {"Unknown", "N/A"})]
        pickup_item = required[1].selectbox("规则必须包含上车行政区", [None] + borough_items,
                                            format_func=lambda value: "不限" if value is None else value.split("=", 1)[1],
                                            key="pattern_pickup")
        rules_frame = pattern_rows(str(run), month, min_support, min_confidence, min_lift,
                                   day_item, pickup_item)
        if rules_frame.empty:
            st.info("当前阈值下没有规则，请降低筛选阈值。页面最低支持度不能低于快照构建阈值。")
        else:
            plot = rules_frame.head(100).copy()
            chart(px.scatter(plot, x="support", y="confidence", size="support_count", color="lift",
                             hover_data=["antecedent", "consequent"],
                             labels={"support": "支持度", "confidence": "置信度", "lift": "提升度",
                                     "support_count": "共同出现记录数", "antecedent": "前件", "consequent": "后件"}), 360)
            display = rules_frame.rename(columns={"antecedent": "前件", "consequent": "后件",
                                                   "support_count": "共同记录数", "support": "支持度",
                                                   "confidence": "置信度", "lift": "提升度"})
            st.dataframe(display, hide_index=True, width="stretch",
                         column_config={name: st.column_config.NumberColumn(format="percent")
                                        for name in ("支持度", "置信度")})
            rule_options = list(rules_frame[["antecedent", "consequent"]].itertuples(index=False, name=None))
            selected = st.selectbox("选择一条规则查看匹配行程", rule_options,
                                    format_func=lambda rule: f"{rule[0]} → {rule[1]}", key="pattern_rule")
            chosen = rules_frame[(rules_frame.antecedent == selected[0]) &
                                 (rules_frame.consequent == selected[1])].iloc[0]
            evidence = pattern_examples(str(run), month, chosen.antecedent, chosen.consequent)
            st.caption(f"最多展示 100 条共同包含前件和后件的行程；规则基于 {chosen.support_count:,} 条记录。")
            st.dataframe(evidence, hide_index=True, width="stretch")
        with st.expander("算法与口径"):
            algorithms = pattern_meta["algorithms"]
            st.write(f"Apriori 与 FP-Growth 频繁项集一致：{'是' if algorithms['identical'] else '否'}。最大项集长度：{cfg['max_itemset_size']}。")
            st.write(f"本次构建用时：Apriori {algorithms['apriori_seconds']:.3f} 秒，FP-Growth {algorithms['fpgrowth_seconds']:.3f} 秒；单次用时仅作实现观察，不是严格基准测试。")
            st.write("事务字段：" + "、".join(pattern_meta["fields"]))
            for limitation in pattern_meta["limitations"]:
                st.markdown(f"- {limitation}")

report_payload = {"month": month, "view": view, "rules": list(keys),
                  "start_date": str(start) if start else None, "end_date": str(end) if end else None,
                  "pickup_zone_id": zone, "raw": raw, "clean": clean,
                  "insights": payload["insights"], "source": metadata["source"],
                  "limitations": metadata["limitations"]}
st.divider()
st.download_button("导出当前窗口分析报告 JSON", json.dumps(report_payload, ensure_ascii=False, indent=2, allow_nan=False),
                   file_name=f"urbanpulse_{month}_analysis.json", mime="application/json")
st.caption("UrbanPulse v0.2 · 可信预处理是模式挖掘的前提，关联规则不是因果证明。")
