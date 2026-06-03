import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_schedule():
    static = pd.read_csv(
        os.path.join(BASE_DIR, "..", "data", "raw", "2026_공휴일_KBO_일정_v2.csv")
    )
    upcoming_path = os.path.join(
        BASE_DIR, "..", "data", "processed", "upcoming_kbo_holiday.csv"
    )
    if not os.path.exists(upcoming_path):
        return static
    upcoming = pd.read_csv(upcoming_path, encoding="utf-8-sig")
    covered = set(upcoming["Date"].unique())
    static = static[~static["Date"].isin(covered)]
    return pd.concat([static, upcoming], ignore_index=True)


def _load_coex():
    df = pd.read_excel(
        os.path.join(BASE_DIR, "..", "data", "raw", "coex_event_data.xlsx")
    )
    df = df.rename(columns={"날짜|역명": "date", "삼성역": "coex_value"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    upcoming_path = os.path.join(
        BASE_DIR, "..", "data", "processed", "upcoming_coex.csv"
    )
    if not os.path.exists(upcoming_path):
        return df
    upcoming = pd.read_csv(upcoming_path).rename(columns={"coex_event": "coex_value"})
    upcoming["date"] = pd.to_datetime(upcoming["date"]).dt.strftime("%Y-%m-%d")
    covered = set(upcoming["date"].unique())
    df = df[~df["date"].isin(covered)]
    return pd.concat([df, upcoming], ignore_index=True)


_df_schedule = _load_schedule()
_df_weights = pd.read_csv(
    os.path.join(BASE_DIR, "..", "data", "processed", "event_weights.csv"),
    encoding="utf-8-sig",
)
_df_coex = _load_coex()

SCHEDULE_STATIONS = {"강남", "잠실", "종합운동장", "홍대입구"}
TIME_COLS = [c for c in _df_schedule.columns if c not in ("Date", "Station")]


def get_weekday(date_str):
    dow = datetime.strptime(date_str, "%Y-%m-%d").weekday()
    if dow == 5:
        return "토요일"
    if dow == 6:
        return "일요일"
    return "평일"


def _has_event_at(date_str, station_key, hour_col):
    row = _df_schedule[
        (_df_schedule["Date"] == date_str) & (_df_schedule["Station"] == station_key)
    ]
    if row.empty:
        return False
    if hour_col not in row.columns:
        return False
    return int(row[hour_col].values[0]) == 1


def _is_kbo_date(date_str):
    # 종합운동장역 이벤트가 있는 날 = KBO 경기일
    row = _df_schedule[
        (_df_schedule["Date"] == date_str) & (_df_schedule["Station"] == "종합운동장역")
    ]
    if row.empty:
        return False
    return bool(row[TIME_COLS].values.max())


def check_event(date, station, minute):
    # date: "2026-03-01", station: "잠실" (역 없음), minute: 1020 (=17:00)
    events = []
    hour = minute // 60
    hour_col = f"{hour:02d}-{hour + 1:02d}"

    # KBO / 공휴일
    if station in SCHEDULE_STATIONS:
        if _has_event_at(date, station + "역", hour_col):
            if station == "종합운동장" or (station == "잠실" and _is_kbo_date(date)):
                event_type = "KBO"
            else:
                event_type = "공휴일"

            weekday = get_weekday(date)
            weight_row = _df_weights[
                (_df_weights["이벤트종류"] == event_type)
                & (_df_weights["역명"] == station)
                & (_df_weights["요일"] == weekday)
                & (_df_weights["시간대"] == minute)
            ]
            if not weight_row.empty:
                mean = float(weight_row["mean"].values[0])
                percent = round((mean - 1) * 100, 1)
                direction = "혼잡" if percent >= 0 else "한산"
                label = "KBO 경기" if event_type == "KBO" else "공휴일"
                events.append({
                    "event_type": event_type,
                    "percentage": percent,
                    "direction": direction,
                    "message": f"오늘은 {label}로 평소보다 {abs(percent):.0f}% 더 {direction} 예상",
                })

    # COEX
    if station == "삼성":
        coex_row = _df_coex[_df_coex["date"] == date]
        if not coex_row.empty and float(coex_row["coex_value"].values[0]) > 0:
            events.append({
                "event_type": "COEX",
                "percentage": None,
                "direction": None,
                "message": "오늘은 COEX 행사로 삼성역이 평소보다 혼잡할 수 있습니다",
            })

    return {"event_active": bool(events), "events": events}


if __name__ == "__main__":
    test_cases = [
        ("2026-03-01", "종합운동장", 1020),
        ("2026-03-01", "잠실",       1020),
        ("2026-05-05", "강남",        840),
        ("2026-04-23", "삼성",        900),
        ("2026-03-02", "강남",        840),
    ]

    for date, station, minute in test_cases:
        result = check_event(date, station, minute)
        print(f"\n[{date}] {station}  {minute // 60:02d}:{minute % 60:02d}")
        print(f"  event_active: {result['event_active']}")
        for e in result["events"]:
            print(f"  {e}")
