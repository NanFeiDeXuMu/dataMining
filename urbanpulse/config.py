"""Keep the data contract and cleaning choices explicit and teachable."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import os
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAGE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
DICTIONARY_URL = "https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf"


def data_root() -> Path:
    return Path(os.environ.get("URBANPULSE_DATA_DIR", ROOT / "data")).resolve()


def month_bounds(month: str) -> tuple[date, date]:
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("月份必须为 YYYY-MM，例如 2025-01。")
    year, number = map(int, month.split("-"))
    return date(year, number, 1), date(year + (number == 12), number % 12 + 1, 1)


def trip_url(month: str) -> str:
    month_bounds(month)
    return f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"


@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    reason: str
    default: bool = False


@dataclass(frozen=True)
class PatternConfig:
    """Fixed mining policy recorded with every snapshot."""

    min_support: float = 0.01
    min_confidence: float = 0.35
    min_lift: float = 1.05
    max_itemset_size: int = 3


RULES = (
    Rule("missing_time", "时间缺失/无法解析", "无法计算行程时长或定位到小时。", True),
    Rule("invalid_duration", "时长不大于 0", "下车时间不晚于上车时间。", True),
    Rule("outside_month", "上车时间不在目标月", "与本次分析月份不一致；原始记录仍保留。", True),
    Rule("invalid_distance", "距离缺失/非有限/负值", "里程应为有限非负数，单位为英里。", True),
    Rule("missing_amount", "费用缺失/非有限", "计价费用或总额不是有效有限数值。", True),
    Rule("unknown_pickup", "上车区域未知", "区域无法关联、Unknown 或 N/A，无法做区域分析。", True),
    Rule("unknown_dropoff", "下车区域未知", "区域无法关联、Unknown 或 N/A。", True),
    Rule("zero_distance", "零距离（待核查）", "可能是短途、取消或记录问题，不默认排除。"),
    Rule("negative_amount", "负费用（待核查）", "可能为退款、冲正或数据问题，不等同于欺诈。"),
    Rule("high_amount", "总额超过 $300（待核查）", "教学探索阈值，不是官方错误判定。"),
    Rule("long_duration", "时长超过 240 分钟（待核查）", "教学探索阈值，不是官方错误判定。"),
    Rule("high_speed", "均速超过 100 mph（待核查）", "距离/时长计算的探索阈值，不是轨迹测量速度。"),
    Rule("missing_passengers", "乘客数缺失（待核查）", "不凭空填补；不影响按记录统计订单量。"),
)
RULE_BY_KEY = {rule.key: rule for rule in RULES}
DEFAULT_RULES = tuple(rule.key for rule in RULES if rule.default)
PATTERN_CONFIG = PatternConfig()
PATTERN_FIELDS = (
    "日期类型", "上车时段", "上车行政区", "下车行政区",
    "距离区间", "费用区间", "支付方式", "乘客数区间",
)
REQUIRED_COLUMNS = (
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
    "PULocationID", "DOLocationID", "fare_amount", "total_amount",
)

# Numeric storage does not imply a continuous or ordinal measurement.
FIELD_INFO = {
    "VendorID": ("类别（名义）", "技术供应商编码；不能平均或解释大小"),
    "tpep_pickup_datetime": ("时间", "计价器启动时间；保留 TLC 本地墙上时间，不作 UTC 转换"),
    "tpep_dropoff_datetime": ("时间", "计价器停止时间"),
    "passenger_count": ("离散数值", "司机报告的乘客数；缺失不自动当作 1"),
    "trip_distance": ("连续数值", "计价器记录距离，单位英里，不是公里"),
    "RatecodeID": ("类别（名义）", "费率类别编码，不表示有序等级"),
    "store_and_fwd_flag": ("二元类别", "是否缓存后转发记录"),
    "PULocationID": ("类别（名义）", "上车区域 ID；不是经纬度或连续变量"),
    "DOLocationID": ("类别（名义）", "下车区域 ID；不是经纬度或连续变量"),
    "payment_type": ("类别（名义）", "支付方式编码，0 也有定义，不能一律作缺失"),
    "fare_amount": ("连续数值", "计价器计算的车费，美元；与总额不同"),
    "total_amount": ("连续数值", "向乘客收取的总额，美元；不含现金小费"),
    "tip_amount": ("连续数值", "自动记录信用卡小费，不含现金小费"),
    "cbd_congestion_fee": ("连续数值", "2025 年起新增的拥堵区域收费，美元"),
}


def keep_expression(keys: tuple[str, ...] | list[str]) -> str:
    unknown = set(keys) - RULE_BY_KEY.keys()
    if unknown:
        raise ValueError(f"未知清洗规则: {sorted(unknown)}")
    return " AND ".join(f"NOT q_{key}" for key in sorted(set(keys))) or "TRUE"
