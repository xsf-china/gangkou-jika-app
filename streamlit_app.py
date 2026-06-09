import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="港区集卡预约排队预测与通行效率分析软件 V1.0",
    page_icon="🚛",
    layout="wide"
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def generate_time_periods(start_hour, end_hour, period_minutes):
    periods = []
    total_minutes = (end_hour - start_hour) * 60
    count = total_minutes // period_minutes
    for i in range(count):
        start_total = start_hour * 60 + i * period_minutes
        end_total = start_total + period_minutes
        s_h, s_m = divmod(start_total, 60)
        e_h, e_m = divmod(end_total, 60)
        periods.append(f"{s_h:02d}:{s_m:02d}-{e_h:02d}:{e_m:02d}")
    return periods


def simulate_arrival(periods, total_vehicles, peak_ratio, fluctuation):
    n = len(periods)
    base = np.ones(n)

    for i in range(n):
        if 2 <= i <= 5:
            base[i] += peak_ratio
        if 10 <= i <= 13:
            base[i] += peak_ratio * 0.8

    base = base / base.sum()
    appointment = np.round(base * total_vehicles).astype(int)

    diff = total_vehicles - appointment.sum()
    appointment[-1] += diff

    actual = []
    for value in appointment:
        factor = np.random.uniform(1 - fluctuation, 1 + fluctuation)
        actual.append(max(0, int(value * factor)))

    return appointment, np.array(actual)


def queue_prediction(actual_arrival, gate_num, service_time, period_minutes):
    service_capacity = int(gate_num * period_minutes / service_time)

    queue_list = []
    served_list = []
    wait_list = []
    utilization_list = []

    last_queue = 0

    for arrival in actual_arrival:
        demand = last_queue + arrival
        served = min(demand, service_capacity)
        queue = max(0, demand - service_capacity)

        if gate_num > 0:
            service_rate = gate_num / service_time
            wait_time = queue / max(service_rate, 0.01)
        else:
            wait_time = 0

        utilization = min(demand / max(service_capacity, 1), 1.2) * 100

        queue_list.append(queue)
        served_list.append(served)
        wait_list.append(round(wait_time, 2))
        utilization_list.append(round(utilization, 2))

        last_queue = queue

    return service_capacity, queue_list, served_list, wait_list, utilization_list


def warning_level(wait_time, queue_length):
    if wait_time <= 10 and queue_length <= 15:
        return "绿色-正常"
    elif wait_time <= 20 and queue_length <= 30:
        return "黄色-轻度拥堵"
    elif wait_time <= 40 and queue_length <= 60:
        return "橙色-中度拥堵"
    else:
        return "红色-严重拥堵"


def warning_color(level):
    if "绿色" in level:
        return "#DFF5E1"
    if "黄色" in level:
        return "#FFF4CC"
    if "橙色" in level:
        return "#FFE3C2"
    return "#FFD6D6"


def efficiency_score(avg_wait, max_queue, avg_utilization):
    wait_score = max(0, 100 - avg_wait * 2.2)
    queue_score = max(0, 100 - max_queue * 0.9)

    if 70 <= avg_utilization <= 95:
        util_score = 95
    elif avg_utilization < 70:
        util_score = max(60, avg_utilization + 15)
    else:
        util_score = max(40, 100 - (avg_utilization - 95) * 2)

    score = wait_score * 0.4 + queue_score * 0.35 + util_score * 0.25
    return round(score, 2)


def optimize_appointment(appointment, service_capacity):
    optimized = appointment.copy().astype(int)

    for i in range(len(optimized)):
        if optimized[i] > service_capacity:
            overflow = optimized[i] - service_capacity
            optimized[i] = service_capacity

            for j in range(len(optimized)):
                if optimized[j] < service_capacity * 0.8:
                    can_add = int(service_capacity * 0.8 - optimized[j])
                    add = min(overflow, can_add)
                    optimized[j] += add
                    overflow -= add
                    if overflow <= 0:
                        break

            if overflow > 0:
                optimized[i] += overflow

    return optimized


def make_chart(df, x_col, y_col, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[x_col], df[y_col], marker="o")
    ax.set_title(title)
    ax.set_xlabel("时段")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def make_bar_chart(df, x_col, y_col, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel("时段")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="预测结果")
    return output.getvalue()


st.title("港区集卡预约排队预测与通行效率分析软件 V1.0")
st.caption("面向港区集卡进出港预约管理、闸口排队预测、拥堵预警和通行效率评价的网页分析软件")

with st.sidebar:
    st.header("参数配置")

    start_hour = st.number_input("开始时间", min_value=0, max_value=23, value=8)
    end_hour = st.number_input("结束时间", min_value=1, max_value=24, value=18)
    period_minutes = st.selectbox("预约时段长度/分钟", [15, 30, 60], index=1)

    total_vehicles = st.number_input("计划进港集卡总数/辆", min_value=50, max_value=5000, value=800)
    gate_num = st.number_input("闸口数量/个", min_value=1, max_value=20, value=4)
    service_time = st.number_input("单车平均服务时间/分钟", min_value=0.5, max_value=20.0, value=3.0)

    peak_ratio = st.slider("高峰集中程度", 0.0, 3.0, 1.2, 0.1)
    fluctuation = st.slider("车辆到达波动系数", 0.0, 0.5, 0.15, 0.01)

    run_button = st.button("开始预测分析", use_container_width=True)

if run_button:
    periods = generate_time_periods(start_hour, end_hour, period_minutes)

    appointment, actual = simulate_arrival(
        periods=periods,
        total_vehicles=total_vehicles,
        peak_ratio=peak_ratio,
        fluctuation=fluctuation
    )

    service_capacity, queue_list, served_list, wait_list, utilization_list = queue_prediction(
        actual_arrival=actual,
        gate_num=gate_num,
        service_time=service_time,
        period_minutes=period_minutes
    )

    optimized = optimize_appointment(appointment, service_capacity)

    df = pd.DataFrame({
        "时段": periods,
        "预约车辆数": appointment,
        "优化后预约车辆数": optimized,
        "实际到达车辆数": actual,
        "单时段服务能力": service_capacity,
        "完成服务车辆数": served_list,
        "剩余排队车辆数": queue_list,
        "平均等待时间/分钟": wait_list,
        "闸口利用率/%": utilization_list
    })

    df["拥堵预警等级"] = [
        warning_level(w, q) for w, q in zip(df["平均等待时间/分钟"], df["剩余排队车辆数"])
    ]

    avg_wait = round(df["平均等待时间/分钟"].mean(), 2)
    max_queue = int(df["剩余排队车辆数"].max())
    avg_utilization = round(df["闸口利用率/%"].mean(), 2)
    score = efficiency_score(avg_wait, max_queue, avg_utilization)

    if score >= 90:
        status = "运行顺畅"
    elif score >= 75:
        status = "基本稳定"
    elif score >= 60:
        status = "轻度拥堵"
    elif score >= 40:
        status = "中度拥堵"
    else:
        status = "严重拥堵"

    st.subheader("一、核心分析结果")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("平均等待时间", f"{avg_wait} 分钟")
    c2.metric("最大排队长度", f"{max_queue} 辆")
    c3.metric("平均闸口利用率", f"{avg_utilization}%")
    c4.metric("通行效率评分", f"{score} 分", status)

    st.subheader("二、拥堵预警结果")

    warning_counts = df["拥堵预警等级"].value_counts().to_dict()
    w1, w2, w3, w4 = st.columns(4)
    w1.info(f"绿色正常：{warning_counts.get('绿色-正常', 0)} 个时段")
    w2.warning(f"黄色轻度：{warning_counts.get('黄色-轻度拥堵', 0)} 个时段")
    w3.warning(f"橙色中度：{warning_counts.get('橙色-中度拥堵', 0)} 个时段")
    w4.error(f"红色严重：{warning_counts.get('红色-严重拥堵', 0)} 个时段")

    def highlight_warning(row):
        color = warning_color(row["拥堵预警等级"])
        return [f"background-color: {color}"] * len(row)

    st.subheader("三、预测数据表")
    st.dataframe(df.style.apply(highlight_warning, axis=1), use_container_width=True)

    st.subheader("四、图表分析")

    tab1, tab2, tab3, tab4 = st.tabs(["到达量", "排队长度", "等待时间", "优化对比"])

    with tab1:
        st.pyplot(make_bar_chart(df, "时段", "实际到达车辆数", "各时段实际到达车辆数", "车辆数/辆"))

    with tab2:
        st.pyplot(make_chart(df, "时段", "剩余排队车辆数", "各时段剩余排队车辆数变化", "排队车辆数/辆"))

    with tab3:
        st.pyplot(make_chart(df, "时段", "平均等待时间/分钟", "各时段平均等待时间变化", "等待时间/分钟"))

    with tab4:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["时段"], df["预约车辆数"], marker="o", label="原预约车辆数")
        ax.plot(df["时段"], df["优化后预约车辆数"], marker="s", label="优化后预约车辆数")
        ax.set_title("预约车辆错峰优化前后对比")
        ax.set_xlabel("时段")
        ax.set_ylabel("车辆数/辆")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("五、结果导出")

    excel_data = to_excel_bytes(df)
    st.download_button(
        label="下载 Excel 分析结果",
        data=excel_data,
        file_name="港区集卡预约排队预测结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("请在左侧设置参数，然后点击“开始预测分析”。")