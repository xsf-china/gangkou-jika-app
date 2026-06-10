# 港区集卡预约排队与通行效率分析工具
# 先做成本地和网页都能跑的版本，后面如果有真实闸口数据，再把阈值和到达分布调细一点。

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# 1.基础配置

SOFTWARE_NAME = "港区集卡预约排队预测与通行效率分析软件"
VERSION = "V1.0"


@dataclass
class PortScenario:
    """港区运行场景模板。"""

    name: str
    description: str
    total_vehicles: int
    gate_num: int
    service_time: float
    peak_ratio: float
    fluctuation: float
    early_ratio: float
    late_ratio: float
    broken_gate_count: int
    broken_start_index: int
    broken_end_index: int


SCENARIOS: Dict[str, PortScenario] = {
    "普通工作日": PortScenario(
        name="普通工作日",
        description="车辆分布相对平稳，上午和下午存在常规进港小高峰。",
        total_vehicles=800,
        gate_num=4,
        service_time=3.0,
        peak_ratio=1.2,
        fluctuation=0.15,
        early_ratio=0.05,
        late_ratio=0.08,
        broken_gate_count=0,
        broken_start_index=0,
        broken_end_index=0,
    ),
    "上午集中进港": PortScenario(
        name="上午集中进港",
        description="上午预约车辆集中，适合测试闸口高峰排队压力。",
        total_vehicles=1100,
        gate_num=4,
        service_time=3.0,
        peak_ratio=2.0,
        fluctuation=0.20,
        early_ratio=0.05,
        late_ratio=0.10,
        broken_gate_count=0,
        broken_start_index=0,
        broken_end_index=0,
    ),
    "节假日前高峰": PortScenario(
        name="节假日前高峰",
        description="节前车辆集中到港，车辆波动更明显，整体服务压力较高。",
        total_vehicles=1500,
        gate_num=5,
        service_time=3.2,
        peak_ratio=2.6,
        fluctuation=0.25,
        early_ratio=0.08,
        late_ratio=0.12,
        broken_gate_count=0,
        broken_start_index=0,
        broken_end_index=0,
    ),
    "闸口临时故障": PortScenario(
        name="闸口临时故障",
        description="中间时段有一个闸口临时不可用，用于分析故障对排队的影响。",
        total_vehicles=900,
        gate_num=4,
        service_time=3.0,
        peak_ratio=1.5,
        fluctuation=0.18,
        early_ratio=0.05,
        late_ratio=0.08,
        broken_gate_count=1,
        broken_start_index=6,
        broken_end_index=10,
    ),
    "大船集中到港": PortScenario(
        name="大船集中到港",
        description="模拟靠泊船舶集中作业后，集装箱提箱车辆短时集中进港。",
        total_vehicles=1300,
        gate_num=5,
        service_time=2.8,
        peak_ratio=2.2,
        fluctuation=0.22,
        early_ratio=0.06,
        late_ratio=0.10,
        broken_gate_count=0,
        broken_start_index=0,
        broken_end_index=0,
    ),
}


DEFAULT_USERS = {
    "admin": {"password": "123456", "role": "管理员", "name": "港区调度员"},
    "student": {"password": "2026", "role": "演示用户", "name": "交通管理学生"},
}


WARNING_RULES = {
    "green_wait": 10,
    "yellow_wait": 20,
    "orange_wait": 40,
    "green_queue": 15,
    "yellow_queue": 30,
    "orange_queue": 60,
    "util_high": 95,
}

# 2.一些通用小函数


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-9:
        return default
    return a / b


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def generate_time_periods(start_hour: int, end_hour: int, period_minutes: int) -> List[str]:
    """生成时段标签，例如 08:00-08:30。"""

    if end_hour <= start_hour:
        raise ValueError("结束时间必须大于开始时间。")

    total_minutes = (end_hour - start_hour) * 60
    period_count = total_minutes // period_minutes

    if period_count <= 0:
        raise ValueError("分析时段数量不能为0。")

    periods = []
    for i in range(period_count):
        start_total = start_hour * 60 + i * period_minutes
        end_total = start_total + period_minutes
        s_h, s_m = divmod(start_total, 60)
        e_h, e_m = divmod(end_total, 60)
        periods.append(f"{s_h:02d}:{s_m:02d}-{e_h:02d}:{e_m:02d}")

    return periods


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8-sig")


def make_run_log(title: str, details: List[str]) -> str:

    lines = [
        f"【{title}】",
        f"生成时间：{now_text()}",
        "-" * 48,
    ]
    lines.extend(details)
    return "\n".join(lines)

#3.关于登录

def init_login_state() -> None:

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_role" not in st.session_state:
        st.session_state.user_role = ""


def login_panel() -> bool:

    init_login_state()

    if st.session_state.logged_in:
        return True

    st.title(f"{SOFTWARE_NAME} {VERSION}")
    st.caption("用户登录")

    with st.container(border=True):
        st.write("请输入账号信息进入系统。")
        account = st.text_input("账号", value="admin")
        password = st.text_input("密码", value="123456", type="password")

        col1, col2 = st.columns([1, 3])
        with col1:
            do_login = st.button("登录系统", use_container_width=True)
        with col2:
            st.info("演示账号：admin，密码：123456")

        if do_login:
            user = DEFAULT_USERS.get(account)
            if user and user["password"] == password:
                st.session_state.logged_in = True
                st.session_state.user_name = user["name"]
                st.session_state.user_role = user["role"]
                st.rerun()
            else:
                st.error("账号或密码错误，请重新输入。")

    st.markdown(
        """
        **软件说明：**  
        本软件面向港区集卡进出港预约管理、闸口排队预测、拥堵预警和通行效率评价场景，
        可运用于课程设计、港口集疏运组织分析等场景。
        """
    )

    return False


def logout_button() -> None:

    with st.sidebar:
        st.divider()
        st.write(f"当前用户：{st.session_state.user_name}")
        st.write(f"用户角色：{st.session_state.user_role}")
        if st.button("退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.user_role = ""
            st.rerun()


# 4. 数据生成与导入

def build_period_weight(period_count: int, peak_ratio: float, pattern: str = "双峰") -> np.ndarray:

    if period_count <= 0:
        return np.array([], dtype=float)

    weights = np.ones(period_count, dtype=float)

    if pattern == "单峰":
        center = period_count * 0.35
        for i in range(period_count):
            distance = abs(i - center)
            weights[i] += peak_ratio * math.exp(-(distance ** 2) / max(period_count, 1))

    elif pattern == "双峰":
        morning_center = period_count * 0.28
        afternoon_center = period_count * 0.68
        for i in range(period_count):
            d1 = abs(i - morning_center)
            d2 = abs(i - afternoon_center)
            weights[i] += peak_ratio * math.exp(-(d1 ** 2) / max(period_count, 1))
            weights[i] += peak_ratio * 0.75 * math.exp(-(d2 ** 2) / max(period_count, 1))

    elif pattern == "均匀":
        weights[:] = 1.0

    elif pattern == "尾峰":
        for i in range(period_count):
            weights[i] += peak_ratio * (i / max(period_count - 1, 1)) ** 2

    else:
        # 正常情况下界面不会传到这里，留个兜底。
        return build_period_weight(period_count, peak_ratio, "双峰")

    weights = weights / weights.sum()
    return weights


def generate_appointment_counts(
    periods: List[str],
    total_vehicles: int,
    peak_ratio: float,
    pattern: str,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """生成各时段预约车辆数。"""

    if random_seed is not None:
        np.random.seed(random_seed)

    weights = build_period_weight(len(periods), peak_ratio, pattern)
    counts = np.round(weights * total_vehicles).astype(int)

    # round完可能差一两辆，直接补到最后一个时段。
    diff = int(total_vehicles - counts.sum())
    if len(counts) > 0:
        counts[-1] += diff

    counts[counts < 0] = 0
    return counts


def simulate_actual_arrival(
    appointment: np.ndarray,
    fluctuation: float,
    early_ratio: float,
    late_ratio: float,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据预约量生成实际到达量，同时考虑早到、迟到和随机波动。"""

    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(appointment)
    actual = np.zeros(n, dtype=int)
    early_shift = np.zeros(n, dtype=int)
    late_shift = np.zeros(n, dtype=int)

    for i, count in enumerate(appointment):
        count = max(0, int(count))

        early = int(round(count * early_ratio))
        late = int(round(count * late_ratio))
        normal = max(0, count - early - late)

        actual[i] += normal

        if i > 0:
            actual[i - 1] += early
            early_shift[i - 1] += early
        else:
            actual[i] += early
            early_shift[i] += early

        if i < n - 1:
            actual[i + 1] += late
            late_shift[i + 1] += late
        else:
            actual[i] += late
            late_shift[i] += late

    for i in range(n):
        factor = np.random.uniform(1 - fluctuation, 1 + fluctuation)
        actual[i] = max(0, int(round(actual[i] * factor)))

    return actual, early_shift, late_shift


def make_sample_dataframe(
    periods: List[str],
    total_vehicles: int,
    peak_ratio: float,
    fluctuation: float,
    early_ratio: float,
    late_ratio: float,
    pattern: str,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """生成一张完整模拟数据表。"""

    appointment = generate_appointment_counts(
        periods=periods,
        total_vehicles=total_vehicles,
        peak_ratio=peak_ratio,
        pattern=pattern,
        random_seed=random_seed,
    )
    actual, early_shift, late_shift = simulate_actual_arrival(
        appointment=appointment,
        fluctuation=fluctuation,
        early_ratio=early_ratio,
        late_ratio=late_ratio,
        random_seed=random_seed,
    )

    df = pd.DataFrame(
        {
            "时段": periods,
            "预约车辆数": appointment,
            "实际到达车辆数": actual,
            "早到转入车辆数": early_shift,
            "迟到转入车辆数": late_shift,
        }
    )
    return df


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """读取上传文件，支持 CSV 和 Excel。"""

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        # 如果环境没有 openpyxl，Excel 可能读取失败。
        # 所以为了网页部署更稳，建议用户上传 CSV。
        df = pd.read_excel(uploaded_file)

    required = {"时段", "预约车辆数"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"上传文件缺少字段：{', '.join(missing)}")

    if "实际到达车辆数" not in df.columns:
        df["实际到达车辆数"] = df["预约车辆数"]

    return clean_input_dataframe(df)


def clean_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["时段"] = df["时段"].astype(str)
    df["预约车辆数"] = df["预约车辆数"].fillna(0).astype(int)
    df["实际到达车辆数"] = df["实际到达车辆数"].fillna(0).astype(int)

    df.loc[df["预约车辆数"] < 0, "预约车辆数"] = 0
    df.loc[df["实际到达车辆数"] < 0, "实际到达车辆数"] = 0

    return df

#5.闸口排队预测

def effective_gate_number(
    gate_num: int,
    period_index: int,
    broken_gate_count: int,
    broken_start_index: int,
    broken_end_index: int,
) -> int:
    """计算某个时段有效闸口数。"""

    effective = gate_num
    if broken_gate_count > 0 and broken_start_index <= period_index <= broken_end_index:
        effective = max(1, gate_num - broken_gate_count)
    return effective


def service_capacity_by_period(
    gate_num: int,
    service_time: float,
    period_minutes: int,
    period_count: int,
    broken_gate_count: int = 0,
    broken_start_index: int = 0,
    broken_end_index: int = 0,
    service_variation: float = 0.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:

    if random_seed is not None:
        np.random.seed(random_seed)

    records = []

    for i in range(period_count):
        effective_gate = effective_gate_number(
            gate_num=gate_num,
            period_index=i,
            broken_gate_count=broken_gate_count,
            broken_start_index=broken_start_index,
            broken_end_index=broken_end_index,
        )

        if service_variation > 0:
            service_factor = np.random.uniform(1 - service_variation, 1 + service_variation)
        else:
            service_factor = 1.0

        real_service_time = max(0.5, service_time * service_factor)
        capacity = int(effective_gate * period_minutes / real_service_time)

        records.append(
            {
                "时段序号": i,
                "有效闸口数": effective_gate,
                "实际平均服务时间/分钟": round(real_service_time, 2),
                "单时段服务能力": max(0, capacity),
            }
        )

    return pd.DataFrame(records)


def predict_queue(
    input_df: pd.DataFrame,
    gate_num: int,
    service_time: float,
    period_minutes: int,
    broken_gate_count: int = 0,
    broken_start_index: int = 0,
    broken_end_index: int = 0,
    service_variation: float = 0.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:

    df = input_df.copy().reset_index(drop=True)
    period_count = len(df)

    capacity_df = service_capacity_by_period(
        gate_num=gate_num,
        service_time=service_time,
        period_minutes=period_minutes,
        period_count=period_count,
        broken_gate_count=broken_gate_count,
        broken_start_index=broken_start_index,
        broken_end_index=broken_end_index,
        service_variation=service_variation,
        random_seed=random_seed,
    )

    queue_values = []
    served_values = []
    wait_values = []
    utilization_values = []
    punctuality_values = []

    last_queue = 0

    for i, row in df.iterrows():
        arrival = int(row["实际到达车辆数"])
        appointment = int(row["预约车辆数"])
        capacity = int(capacity_df.loc[i, "单时段服务能力"])
        effective_gate = int(capacity_df.loc[i, "有效闸口数"])

        demand = last_queue + arrival
        served = min(demand, capacity)
        queue = max(0, demand - capacity)

        service_rate_per_minute = effective_gate / max(service_time, 0.5)
        wait_time = queue / max(service_rate_per_minute, 0.01)

        utilization = safe_divide(demand, max(capacity, 1), 0.0) * 100
        utilization = min(utilization, 130.0)

        if appointment <= 0:
            punctuality = 100.0
        else:
            deviation = abs(arrival - appointment)
            punctuality = max(0.0, 100.0 - deviation / appointment * 100.0)

        queue_values.append(int(queue))
        served_values.append(int(served))
        wait_values.append(round(wait_time, 2))
        utilization_values.append(round(utilization, 2))
        punctuality_values.append(round(punctuality, 2))

        last_queue = queue

    df["有效闸口数"] = capacity_df["有效闸口数"]
    df["实际平均服务时间/分钟"] = capacity_df["实际平均服务时间/分钟"]
    df["单时段服务能力"] = capacity_df["单时段服务能力"]
    df["完成服务车辆数"] = served_values
    df["剩余排队车辆数"] = queue_values
    df["平均等待时间/分钟"] = wait_values
    df["闸口利用率/%"] = utilization_values
    df["准点率/%"] = punctuality_values

    return df


def summarize_queue_result(df: pd.DataFrame) -> Dict[str, float]:
    """汇总排队预测结果。"""

    summary = {
        "总预约车辆数": int(df["预约车辆数"].sum()),
        "总实际到达车辆数": int(df["实际到达车辆数"].sum()),
        "总完成服务车辆数": int(df["完成服务车辆数"].sum()),
        "最大排队长度": int(df["剩余排队车辆数"].max()),
        "平均排队长度": round(float(df["剩余排队车辆数"].mean()), 2),
        "平均等待时间": round(float(df["平均等待时间/分钟"].mean()), 2),
        "最大等待时间": round(float(df["平均等待时间/分钟"].max()), 2),
        "平均闸口利用率": round(float(df["闸口利用率/%"].mean()), 2),
        "平均准点率": round(float(df["准点率/%"].mean()), 2),
    }
    return summary


# 6.简单错峰调整

def calculate_balance_index(values: np.ndarray) -> float:
    """计算预约均衡度，100分代表非常均衡。"""

    values = np.array(values, dtype=float)

    if values.sum() <= 0:
        return 100.0

    mean_value = values.mean()
    if mean_value <= 0:
        return 100.0

    cv = values.std() / mean_value
    score = max(0.0, 100.0 - cv * 60.0)
    return round(score, 2)


def transfer_to_nearby_period(
    values: np.ndarray,
    source_index: int,
    overflow: int,
    target_capacity: int,
    max_search_radius: int = 4,
) -> Tuple[np.ndarray, List[str]]:
    """把高峰时段的超出车辆转移到附近低负荷时段。"""

    result = values.copy()
    logs = []

    if overflow <= 0:
        return result, logs

    for radius in range(1, max_search_radius + 1):
        candidate_indices = [source_index - radius, source_index + radius]

        for idx in candidate_indices:
            if idx < 0 or idx >= len(result):
                continue

            spare = int(target_capacity * 0.85 - result[idx])
            if spare <= 0:
                continue

            move = min(overflow, spare)
            result[source_index] -= move
            result[idx] += move
            overflow -= move

            logs.append(f"将第{source_index + 1}时段 {move} 辆车调整至第{idx + 1}时段。")

            if overflow <= 0:
                return result, logs

    return result, logs


def optimize_by_capacity(
    appointment: np.ndarray,
    service_capacity: np.ndarray,
    max_transfer_ratio: float = 0.35,
    max_search_radius: int = 4,
) -> Tuple[np.ndarray, List[str]]:
    """根据服务能力进行预约错峰优化。"""

    optimized = appointment.copy().astype(int)
    logs = []

    for i in range(len(optimized)):
        capacity = int(service_capacity[i])
        if capacity <= 0:
            continue

        limit = int(capacity * 0.90)

        if optimized[i] > limit:
            raw_overflow = optimized[i] - limit
            max_transfer = int(optimized[i] * max_transfer_ratio)
            overflow = min(raw_overflow, max_transfer)

            if overflow > 0:
                optimized, item_logs = transfer_to_nearby_period(
                    values=optimized,
                    source_index=i,
                    overflow=overflow,
                    target_capacity=capacity,
                    max_search_radius=max_search_radius,
                )
                logs.extend(item_logs)

    if not logs:
        logs.append("当前预约分布未触发明显错峰调整，保持原预约方案。")

    return optimized, logs


def make_optimized_dataframe(df: pd.DataFrame, optimized: np.ndarray) -> pd.DataFrame:
    """生成优化方案表。"""

    opt_df = df[["时段", "预约车辆数"]].copy()
    opt_df["优化后预约车辆数"] = optimized
    opt_df["调整车辆数"] = opt_df["优化后预约车辆数"] - opt_df["预约车辆数"]

    suggestions = []
    for value in opt_df["调整车辆数"]:
        if value > 0:
            suggestions.append(f"建议增加预约 {int(value)} 辆")
        elif value < 0:
            suggestions.append(f"建议减少预约 {abs(int(value))} 辆")
        else:
            suggestions.append("保持原预约量")

    opt_df["预约调整建议"] = suggestions
    return opt_df


def build_strategy_text(logs: List[str], before_balance: float, after_balance: float) -> str:
    """把优化日志整理成人能看懂的文字。"""

    lines = [
        "预约错峰优化说明：",
        f"优化前预约均衡度：{before_balance} 分",
        f"优化后预约均衡度：{after_balance} 分",
        "",
        "主要调整记录：",
    ]

    for item in logs[:12]:
        lines.append(f"- {item}")

    if len(logs) > 12:
        lines.append(f"- 其余 {len(logs) - 12} 条调整记录已省略，可在导出结果中查看。")

    return "\n".join(lines)


def compare_before_after(
    before_df: pd.DataFrame,
    after_arrival: np.ndarray,
    gate_num: int,
    service_time: float,
    period_minutes: int,
    broken_gate_count: int = 0,
    broken_start_index: int = 0,
    broken_end_index: int = 0,
    service_variation: float = 0.0,
) -> pd.DataFrame:
    """基于优化后的预约量重新计算排队结果，并生成对比表。"""

    after_input = before_df[["时段", "预约车辆数"]].copy()
    after_input["实际到达车辆数"] = after_arrival

    after_df = predict_queue(
        input_df=after_input,
        gate_num=gate_num,
        service_time=service_time,
        period_minutes=period_minutes,
        broken_gate_count=broken_gate_count,
        broken_start_index=broken_start_index,
        broken_end_index=broken_end_index,
        service_variation=service_variation,
    )

    compare = pd.DataFrame(
        {
            "时段": before_df["时段"],
            "优化前实际到达": before_df["实际到达车辆数"],
            "优化后实际到达": after_arrival,
            "优化前排队": before_df["剩余排队车辆数"],
            "优化后排队": after_df["剩余排队车辆数"],
            "优化前等待/分钟": before_df["平均等待时间/分钟"],
            "优化后等待/分钟": after_df["平均等待时间/分钟"],
            "排队减少量": before_df["剩余排队车辆数"] - after_df["剩余排队车辆数"],
        }
    )
    return compare


# 7. 运行评价

def score_wait_time(avg_wait: float) -> float:
    """等待时间评分。"""

    if avg_wait <= 5:
        return 100.0
    if avg_wait <= 10:
        return 95.0 - (avg_wait - 5) * 2
    if avg_wait <= 20:
        return 85.0 - (avg_wait - 10) * 2.5
    if avg_wait <= 40:
        return 60.0 - (avg_wait - 20) * 1.5

    return max(0.0, 30.0 - (avg_wait - 40) * 0.8)


def score_queue_length(max_queue: int) -> float:
    """最大排队长度评分。"""

    if max_queue <= 10:
        return 100.0
    if max_queue <= 30:
        return 95.0 - (max_queue - 10) * 1.2
    if max_queue <= 60:
        return 70.0 - (max_queue - 30) * 1.0

    return max(0.0, 40.0 - (max_queue - 60) * 0.5)


def score_utilization(avg_util: float) -> float:
    """闸口利用率评分。利用率太低或者太高都不理想。"""

    if 70 <= avg_util <= 92:
        return 100.0
    if avg_util < 70:
        return clamp(60 + avg_util * 0.5, 0, 95)

    return clamp(100 - (avg_util - 92) * 2.5, 0, 100)


def score_punctuality(avg_punctuality: float) -> float:
    """准点率评分。"""

    return clamp(avg_punctuality, 0, 100)


def calculate_efficiency(df: pd.DataFrame) -> Dict[str, float | str]:
    """计算综合通行效率评分。"""

    avg_wait = float(df["平均等待时间/分钟"].mean())
    max_queue = int(df["剩余排队车辆数"].max())
    avg_util = float(df["闸口利用率/%"].mean())
    avg_punctuality = float(df["准点率/%"].mean())
    balance = calculate_balance_index(df["预约车辆数"].values)

    wait_s = score_wait_time(avg_wait)
    queue_s = score_queue_length(max_queue)
    util_s = score_utilization(avg_util)
    punctual_s = score_punctuality(avg_punctuality)
    balance_s = balance

    total_score = (
        wait_s * 0.30
        + queue_s * 0.25
        + util_s * 0.20
        + punctual_s * 0.15
        + balance_s * 0.10
    )

    if total_score >= 90:
        status = "运行顺畅"
    elif total_score >= 75:
        status = "基本稳定"
    elif total_score >= 60:
        status = "轻度拥堵"
    elif total_score >= 40:
        status = "中度拥堵"
    else:
        status = "严重拥堵"

    return {
        "综合评分": round(total_score, 2),
        "运行状态": status,
        "等待时间评分": round(wait_s, 2),
        "排队长度评分": round(queue_s, 2),
        "闸口利用率评分": round(util_s, 2),
        "准点率评分": round(punctual_s, 2),
        "预约均衡度评分": round(balance_s, 2),
        "平均等待时间": round(avg_wait, 2),
        "最大排队长度": max_queue,
        "平均闸口利用率": round(avg_util, 2),
        "平均准点率": round(avg_punctuality, 2),
    }


def build_evaluation_table(score_dict: Dict[str, float | str]) -> pd.DataFrame:
    """把评价结果整理成表格。"""

    rows = [
        ("综合评分", score_dict["综合评分"], "分"),
        ("运行状态", score_dict["运行状态"], ""),
        ("等待时间评分", score_dict["等待时间评分"], "分"),
        ("排队长度评分", score_dict["排队长度评分"], "分"),
        ("闸口利用率评分", score_dict["闸口利用率评分"], "分"),
        ("准点率评分", score_dict["准点率评分"], "分"),
        ("预约均衡度评分", score_dict["预约均衡度评分"], "分"),
        ("平均等待时间", score_dict["平均等待时间"], "分钟"),
        ("最大排队长度", score_dict["最大排队长度"], "辆"),
        ("平均闸口利用率", score_dict["平均闸口利用率"], "%"),
        ("平均准点率", score_dict["平均准点率"], "%"),
    ]

    return pd.DataFrame(rows, columns=["评价指标", "指标值", "单位"])


def build_advice(score_dict: Dict[str, float | str]) -> List[str]:
    """根据评价结果生成管理建议。"""

    advice = []

    if float(score_dict["平均等待时间"]) > 20:
        advice.append("平均等待时间偏高，建议对高峰时段预约车辆进行错峰分流。")
    if int(score_dict["最大排队长度"]) > 50:
        advice.append("最大排队长度较大，建议临时增开闸口或提前释放低峰预约额度。")
    if float(score_dict["平均闸口利用率"]) > 95:
        advice.append("闸口利用率接近饱和，应预留应急服务能力，避免突发车辆造成拥堵扩散。")
    if float(score_dict["平均准点率"]) < 80:
        advice.append("预约车辆实际到达波动较大，建议加强预约时段提醒和车辆到港引导。")
    if float(score_dict["预约均衡度评分"]) < 70:
        advice.append("预约时段分布不均衡，建议优化预约额度配置。")

    if not advice:
        advice.append("当前通行状态整体较稳定，可保持现有预约计划，并持续关注高峰时段车辆波动。")

    return advice


# 8. 拥堵预警判断

def judge_warning_level(wait_time: float, queue_length: int, utilization: float) -> str:
    """根据等待时间、排队长度、利用率判断拥堵等级。"""

    if (
        wait_time <= WARNING_RULES["green_wait"]
        and queue_length <= WARNING_RULES["green_queue"]
        and utilization <= WARNING_RULES["util_high"]
    ):
        return "绿色-正常"

    if wait_time <= WARNING_RULES["yellow_wait"] and queue_length <= WARNING_RULES["yellow_queue"]:
        return "黄色-轻度拥堵"

    if wait_time <= WARNING_RULES["orange_wait"] and queue_length <= WARNING_RULES["orange_queue"]:
        return "橙色-中度拥堵"

    return "红色-严重拥堵"


def warning_color(level: str) -> str:
    """预警等级表格颜色。"""

    if "绿色" in level:
        return "#E4F7E7"
    if "黄色" in level:
        return "#FFF3C4"
    if "橙色" in level:
        return "#FFE0B8"
    return "#FFD1D1"


def warning_weight(level: str) -> int:
    """预警等级权重。"""

    if "绿色" in level:
        return 1
    if "黄色" in level:
        return 2
    if "橙色" in level:
        return 3
    return 4


def make_dispatch_suggestion(level: str) -> str:
    """生成调度建议。"""

    if "绿色" in level:
        return "维持当前预约与闸口配置。"
    if "黄色" in level:
        return "关注车辆集中到达情况，必要时引导部分车辆错峰。"
    if "橙色" in level:
        return "建议启动错峰分流，并优先保障已预约车辆通行。"
    return "建议临时增开闸口、限制集中到达，并发布红色拥堵提示。"


def add_warning_columns(df: pd.DataFrame) -> pd.DataFrame:
    """给预测结果添加预警等级和调度建议。"""

    df = df.copy()

    levels = []
    suggestions = []

    for _, row in df.iterrows():
        level = judge_warning_level(
            wait_time=float(row["平均等待时间/分钟"]),
            queue_length=int(row["剩余排队车辆数"]),
            utilization=float(row["闸口利用率/%"]),
        )
        levels.append(level)
        suggestions.append(make_dispatch_suggestion(level))

    df["拥堵预警等级"] = levels
    df["预警权重"] = [warning_weight(item) for item in levels]
    df["调度建议"] = suggestions

    return df


def summarize_warning(df: pd.DataFrame) -> pd.DataFrame:
    """统计各预警等级数量。"""

    order = ["绿色-正常", "黄色-轻度拥堵", "橙色-中度拥堵", "红色-严重拥堵"]
    counts = df["拥堵预警等级"].value_counts().to_dict()

    rows = []
    for item in order:
        rows.append(
            {
                "预警等级": item,
                "时段数量": int(counts.get(item, 0)),
                "占比/%": round(counts.get(item, 0) / max(len(df), 1) * 100, 2),
            }
        )

    return pd.DataFrame(rows)


def most_serious_periods(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """找出最严重时段。"""

    sort_cols = ["预警权重", "平均等待时间/分钟", "剩余排队车辆数"]
    return df.sort_values(sort_cols, ascending=False).head(top_n)


# 9. 导出报告

def build_text_report(
    evaluation: Dict[str, float | str],
    advice: List[str],
    warning_df: pd.DataFrame,
    scenario_name: str,
) -> str:
    """生成文字报告。"""

    lines = [
        "港区集卡预约排队预测与通行效率分析报告",
        f"生成时间：{now_text()}",
        f"分析场景：{scenario_name}",
        "",
        "一、综合评价",
        f"综合评分：{evaluation['综合评分']} 分",
        f"运行状态：{evaluation['运行状态']}",
        f"平均等待时间：{evaluation['平均等待时间']} 分钟",
        f"最大排队长度：{evaluation['最大排队长度']} 辆",
        f"平均闸口利用率：{evaluation['平均闸口利用率']}%",
        f"平均准点率：{evaluation['平均准点率']}%",
        "",
        "二、预警统计",
    ]

    for _, row in warning_df.iterrows():
        lines.append(f"{row['预警等级']}：{row['时段数量']} 个时段，占比 {row['占比/%']}%")

    lines.append("")
    lines.append("三、调度建议")
    for idx, item in enumerate(advice, start=1):
        lines.append(f"（{idx}）{item}")

    lines.append("")
    lines.append("四、说明")
    lines.append("本报告由软件根据输入参数与模拟数据自动生成，仅作为港区集卡预约组织和闸口通行效率分析参考。")

    return "\n".join(lines)


def merge_result_text(
    result_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    warning_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    serious_df: pd.DataFrame,
) -> str:
    """把多个表合并成一个文本，方便一个文件下载。"""

    parts = [
        "【预测结果】",
        result_df.to_csv(index=False),
        "\n【预约优化方案】",
        opt_df.to_csv(index=False),
        "\n【预警统计】",
        warning_df.to_csv(index=False),
        "\n【效率评价】",
        evaluation_df.to_csv(index=False),
        "\n【重点关注时段】",
        serious_df.to_csv(index=False),
    ]

    return "\n".join(parts)


# 10. 页面输入与主流程

def page_header() -> None:
    """页面标题。"""

    st.title(f"{SOFTWARE_NAME} {VERSION}")
    st.caption("面向港区集卡进出港预约管理、闸口排队预测、拥堵预警和通行效率评价的网页分析软件")

def check_params(params: Dict[str, object]) -> List[str]:
        tips = []

        if int(params["end_hour"]) <= int(params["start_hour"]):
            tips.append("结束时间要大于开始时间。")

        if float(params["service_time"]) <= 0:
            tips.append("单车服务时间不能小于等于0。")

        if int(params["broken_gate_count"]) >= int(params["gate_num"]):
            tips.append("故障闸口数量不能大于等于总闸口数量。")

        if int(params["broken_start_index"]) > int(params["broken_end_index"]):
            tips.append("故障开始时段不能晚于故障结束时段。")

        return tips

def scenario_select_panel() -> Dict[str, object]:
    st.sidebar.header("一、场景选择")
    scenario_name = st.sidebar.selectbox("选择港区运行场景", list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_name]
    st.sidebar.caption(scenario.description)

    st.sidebar.header("二、基础参数")
    start_hour = st.sidebar.number_input("开始时间", min_value=0, max_value=23, value=8)
    end_hour = st.sidebar.number_input("结束时间", min_value=1, max_value=24, value=18)
    period_minutes = st.sidebar.selectbox("预约时段长度/分钟", [15, 30, 60], index=1)

    total_vehicles = st.sidebar.number_input(
        "计划进港集卡总数/辆",
        min_value=50,
        max_value=6000,
        value=scenario.total_vehicles,
        step=50,
    )

    gate_num = st.sidebar.number_input(
        "闸口数量/个",
        min_value=1,
        max_value=20,
        value=scenario.gate_num,
    )

    service_time = st.sidebar.number_input(
        "单车平均服务时间/分钟",
        min_value=0.5,
        max_value=20.0,
        value=scenario.service_time,
        step=0.1,
    )

    st.sidebar.header("三、到达规律参数")
    pattern = st.sidebar.selectbox("车辆到达分布模式", ["双峰", "单峰", "均匀", "尾峰"], index=0)
    peak_ratio = st.sidebar.slider("高峰集中程度", 0.0, 3.5, scenario.peak_ratio, 0.1)
    fluctuation = st.sidebar.slider("车辆到达波动系数", 0.0, 0.6, scenario.fluctuation, 0.01)
    early_ratio = st.sidebar.slider("早到车辆比例", 0.0, 0.3, scenario.early_ratio, 0.01)
    late_ratio = st.sidebar.slider("迟到车辆比例", 0.0, 0.3, scenario.late_ratio, 0.01)

    st.sidebar.header("四、闸口异常设置")
    broken_gate_count = st.sidebar.number_input(
        "临时故障闸口数量/个",
        min_value=0,
        max_value=max(gate_num - 1, 0),
        value=min(scenario.broken_gate_count, max(gate_num - 1, 0)),
    )
    broken_start_index = st.sidebar.number_input(
        "故障开始时段序号",
        min_value=0,
        max_value=40,
        value=scenario.broken_start_index,
    )
    broken_end_index = st.sidebar.number_input(
        "故障结束时段序号",
        min_value=0,
        max_value=40,
        value=scenario.broken_end_index,
    )
    service_variation = st.sidebar.slider("服务时间波动系数", 0.0, 0.4, 0.05, 0.01)

    st.sidebar.header("五、运行控制")
    random_seed = st.sidebar.number_input("随机种子", min_value=0, max_value=99999, value=2026, step=1)
    run_note = st.sidebar.text_area(
        "本次分析备注",
        value="先按默认参数试算，后续再结合实际闸口记录调整。",
        height=70,
    )

    return {
        "scenario_name": scenario_name,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "period_minutes": period_minutes,
        "total_vehicles": total_vehicles,
        "gate_num": gate_num,
        "service_time": service_time,
        "pattern": pattern,
        "peak_ratio": peak_ratio,
        "fluctuation": fluctuation,
        "early_ratio": early_ratio,
        "late_ratio": late_ratio,
        "broken_gate_count": broken_gate_count,
        "broken_start_index": broken_start_index,
        "broken_end_index": broken_end_index,
        "service_variation": service_variation,
        "random_seed": random_seed,
        "run_note": run_note,
    }


def upload_panel():
    """可选上传数据。"""

    with st.expander("可选：上传自定义预约数据"):
        st.write("上传文件建议使用 CSV，字段至少包含：时段、预约车辆数。若包含“实际到达车辆数”，软件会优先使用。")
        uploaded = st.file_uploader("上传 CSV 或 Excel", type=["csv", "xlsx", "xls"])
        return uploaded


def run_analysis(params: Dict[str, object], uploaded_file=None) -> Dict[str, object]:
    """执行完整分析流程。"""

    periods = generate_time_periods(
        start_hour=int(params["start_hour"]),
        end_hour=int(params["end_hour"]),
        period_minutes=int(params["period_minutes"]),
    )

    if uploaded_file is not None:
        input_df = read_uploaded_file(uploaded_file)
    else:
        input_df = make_sample_dataframe(
            periods=periods,
            total_vehicles=int(params["total_vehicles"]),
            peak_ratio=float(params["peak_ratio"]),
            fluctuation=float(params["fluctuation"]),
            early_ratio=float(params["early_ratio"]),
            late_ratio=float(params["late_ratio"]),
            pattern=str(params["pattern"]),
            random_seed=int(params["random_seed"]),
        )

    result_df = predict_queue(
        input_df=input_df,
        gate_num=int(params["gate_num"]),
        service_time=float(params["service_time"]),
        period_minutes=int(params["period_minutes"]),
        broken_gate_count=int(params["broken_gate_count"]),
        broken_start_index=int(params["broken_start_index"]),
        broken_end_index=int(params["broken_end_index"]),
        service_variation=float(params["service_variation"]),
        random_seed=int(params["random_seed"]),
    )
    result_df = add_warning_columns(result_df)

    service_capacity = result_df["单时段服务能力"].values
    optimized, opt_logs = optimize_by_capacity(
        appointment=result_df["预约车辆数"].values,
        service_capacity=service_capacity,
    )

    opt_df = make_optimized_dataframe(result_df, optimized)

    before_balance = calculate_balance_index(result_df["预约车辆数"].values)
    after_balance = calculate_balance_index(optimized)
    strategy_text = build_strategy_text(opt_logs, before_balance, after_balance)

    compare_df = compare_before_after(
        before_df=result_df,
        after_arrival=optimized,
        gate_num=int(params["gate_num"]),
        service_time=float(params["service_time"]),
        period_minutes=int(params["period_minutes"]),
        broken_gate_count=int(params["broken_gate_count"]),
        broken_start_index=int(params["broken_start_index"]),
        broken_end_index=int(params["broken_end_index"]),
        service_variation=float(params["service_variation"]),
    )

    evaluation = calculate_efficiency(result_df)
    evaluation_df = build_evaluation_table(evaluation)
    advice = build_advice(evaluation)
    warning_df = summarize_warning(result_df)
    serious_df = most_serious_periods(result_df)
    queue_summary = summarize_queue_result(result_df)

    run_log = make_run_log(
        "港区集卡排队预测运行日志",
        [
            f"分析场景：{params['scenario_name']}",
            f"分析时段数量：{len(result_df)}",
            f"闸口数量：{params['gate_num']}",
            f"单车平均服务时间：{params['service_time']} 分钟",
            f"计划车辆总数：{params['total_vehicles']} 辆",
            f"平均等待时间：{evaluation['平均等待时间']} 分钟",
            f"最大排队长度：{evaluation['最大排队长度']} 辆",
            f"综合评分：{evaluation['综合评分']} 分",
            f"运行状态：{evaluation['运行状态']}",
        ],
    )

    text_report = build_text_report(
        evaluation=evaluation,
        advice=advice,
        warning_df=warning_df,
        scenario_name=str(params["scenario_name"]),
    )

    return {
        "result_df": result_df,
        "opt_df": opt_df,
        "compare_df": compare_df,
        "evaluation": evaluation,
        "evaluation_df": evaluation_df,
        "advice": advice,
        "warning_df": warning_df,
        "serious_df": serious_df,
        "strategy_text": strategy_text,
        "queue_summary": queue_summary,
        "run_log": run_log,
        "text_report": text_report,
    }


# 11. 图表展示

def show_native_line_chart(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """使用 Streamlit 原生折线图。"""

    chart_df = df[[x_col, y_col]].copy()
    chart_df = chart_df.set_index(x_col)
    st.line_chart(chart_df)


def show_native_bar_chart(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """使用 Streamlit 原生柱状图。"""

    chart_df = df[[x_col, y_col]].copy()
    chart_df = chart_df.set_index(x_col)
    st.bar_chart(chart_df)


def show_compare_line_chart(df: pd.DataFrame, x_col: str, y_cols: List[str]) -> None:
    """多指标对比折线图。"""

    chart_df = df[[x_col] + y_cols].copy()
    chart_df = chart_df.set_index(x_col)
    st.line_chart(chart_df)


def highlight_warning(row):
    """预警表格高亮。"""

    color = warning_color(row["拥堵预警等级"])
    return [f"background-color: {color}"] * len(row)


def show_metrics(result: Dict[str, object]) -> None:
    """显示核心指标。"""

    evaluation = result["evaluation"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("综合评分", f"{evaluation['综合评分']} 分", evaluation["运行状态"])
    c2.metric("平均等待时间", f"{evaluation['平均等待时间']} 分钟")
    c3.metric("最大排队长度", f"{evaluation['最大排队长度']} 辆")
    c4.metric("平均闸口利用率", f"{evaluation['平均闸口利用率']}%")


def show_result_tabs(result: Dict[str, object]) -> None:
    """展示结果标签页。"""

    result_df = result["result_df"]
    opt_df = result["opt_df"]
    compare_df = result["compare_df"]
    warning_df = result["warning_df"]
    evaluation_df = result["evaluation_df"]
    serious_df = result["serious_df"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["预测数据", "图表分析", "优化方案", "预警统计", "报告导出"])

    with tab1:
        st.subheader("预测结果数据表")
        st.dataframe(result_df.style.apply(highlight_warning, axis=1), use_container_width=True)

        st.subheader("通行效率评价表")
        st.dataframe(evaluation_df, use_container_width=True)

        st.subheader("排队结果汇总")
        st.json(result["queue_summary"])

    with tab2:
        st.subheader("各时段实际到达车辆数")
        show_native_bar_chart(result_df, "时段", "实际到达车辆数")

        st.subheader("预约车辆数与实际到达车辆数对比")
        show_compare_line_chart(result_df, "时段", ["预约车辆数", "实际到达车辆数"])

        st.subheader("各时段剩余排队车辆数")
        show_native_line_chart(result_df, "时段", "剩余排队车辆数")

        st.subheader("各时段平均等待时间")
        show_native_line_chart(result_df, "时段", "平均等待时间/分钟")

        st.subheader("各时段闸口利用率")
        show_native_line_chart(result_df, "时段", "闸口利用率/%")

    with tab3:
        st.subheader("预约错峰优化说明")
        st.text(result["strategy_text"])

        st.subheader("预约优化方案")
        st.dataframe(opt_df, use_container_width=True)

        st.subheader("优化前后预约车辆数对比")
        show_compare_line_chart(opt_df, "时段", ["预约车辆数", "优化后预约车辆数"])

        st.subheader("优化前后排队长度对比")
        show_compare_line_chart(compare_df, "时段", ["优化前排队", "优化后排队"])

        st.subheader("优化前后等待时间对比")
        show_compare_line_chart(compare_df, "时段", ["优化前等待/分钟", "优化后等待/分钟"])

    with tab4:
        st.subheader("拥堵预警统计")
        st.dataframe(warning_df, use_container_width=True)
        show_native_bar_chart(warning_df, "预警等级", "时段数量")

        st.subheader("重点关注时段")
        st.dataframe(serious_df, use_container_width=True)

    with tab5:
        st.subheader("管理建议")
        for idx, item in enumerate(result["advice"], start=1):
            st.write(f"{idx}. {item}")

        st.subheader("文字版分析报告")
        st.text_area("报告内容", value=result["text_report"], height=260)

        all_text = merge_result_text(
            result_df=result_df,
            opt_df=opt_df,
            warning_df=warning_df,
            evaluation_df=evaluation_df,
            serious_df=serious_df,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.download_button(
                label="下载预测结果CSV",
                data=dataframe_to_csv_bytes(result_df),
                file_name="港区集卡预测结果.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c2:
            st.download_button(
                label="下载文字版报告",
                data=text_to_bytes(result["text_report"]),
                file_name="港区集卡通行效率分析报告.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with c3:
            st.download_button(
                label="下载综合结果文本",
                data=text_to_bytes(all_text),
                file_name="港区集卡综合分析结果.txt",
                mime="text/plain",
                use_container_width=True,
            )


# 12. 程序入口

def main() -> None:
    """主程序。"""

    st.set_page_config(
        page_title=f"{SOFTWARE_NAME} {VERSION}",
        page_icon="🚛",
        layout="wide",
    )

    if not login_panel():
        return

    page_header()
    logout_button()

    params = scenario_select_panel()
    uploaded_file = upload_panel()

    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("开始预测分析", type="primary", use_container_width=True)
    with col2:
        st.info("第一次可以先不改参数，直接跑一遍看看结果。")

    if run_btn:
        tips = check_params(params)
        if tips:
            for tip in tips:
                st.warning(tip)
            return

        try:
            result = run_analysis(params, uploaded_file)
            st.session_state["last_result"] = result
            st.success("已算完，可以查看下面的表格和图。")
        except Exception as exc:
            st.error(f"运行失败：{exc}")

    if "last_result" in st.session_state:
        show_metrics(st.session_state["last_result"])
        show_result_tabs(st.session_state["last_result"])
    else:
        st.info("请在左侧设置参数，然后点击“开始预测分析”。")


if __name__ == "__main__":
    main()
