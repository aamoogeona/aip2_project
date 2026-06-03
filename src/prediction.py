import os
import joblib
import pandas as pd
from datetime import datetime
from event_detection import check_event

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2호선 외선 방향 역 순서 (지선역 제외)
LINE2_STATIONS = [
    "시청", "을지로 입구", "을지로 3가", "을지로 4가", "동대문역사문화공원",
    "신당", "상왕십리", "왕십리", "한양대", "뚝섬", "성수",
    "건대입구", "구의", "강변", "잠실나루", "잠실", "잠실새내",
    "종합운동장", "삼성", "선릉", "역삼", "강남", "교대", "서초",
    "방배", "사당", "낙성대", "서울대입구", "봉천", "신림", "신대방",
    "구로디지털단지", "대림", "신도림", "문래", "영등포구청", "당산",
    "합정", "홍대입구", "신촌", "이대", "아현", "충정로",
]
station_idx = {s: i for i, s in enumerate(LINE2_STATIONS)}
N_STATIONS = len(LINE2_STATIONS)


def get_direction(departure, arrival):
    # 출발->도착 중 더 짧은 방향을 외선/내선으로 반환
    if departure not in station_idx or arrival not in station_idx:
        return "외선"
    dep = station_idx[departure]
    arr = station_idx[arrival]
    forward = (arr - dep) % N_STATIONS
    return "외선" if forward <= N_STATIONS - forward else "내선"


# 모델 로딩
model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'randomforest_final.pkl'))
encoders = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'encoders.pkl'))
feature_columns = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'feature_columns.pkl'))


def get_weekday(date_str):
    dow = datetime.strptime(date_str, "%Y-%m-%d").weekday()
    if dow == 5:
        return "토요일"
    if dow == 6:
        return "일요일"
    return "평일"


def congestion_grade(congestion):
    if congestion <= 33:
        return "여유"
    if congestion <= 50:
        return "보통"
    if congestion <= 80:
        return "혼잡"
    if congestion <= 130:
        return "매우 혼잡"
    return "극혼잡"


def predict_congestion(date, departure, arrival, minute):
    direction = get_direction(departure, arrival)
    weekday = get_weekday(date)

    event_result = check_event(date, departure, minute)
    kbo, holiday, coex = 0, 0, 0
    for e in event_result["events"]:
        if e["event_type"] == "KBO":
            kbo = 1
        elif e["event_type"] == "공휴일":
            holiday = 1
        elif e["event_type"] == "COEX":
            coex = 1

    station_code = encoders['역명'].transform([departure])[0]
    weekday_code = encoders['요일'].transform([weekday])[0]
    direction_code = encoders['상하구분'].transform([direction])[0]

    row = {
        '시간대': minute,
        'KBO': kbo,
        '공휴일': holiday,
        'COEX': coex,
        f'역명_{station_code}': 1,
        f'요일_{weekday_code}': 1,
        f'상하구분_{direction_code}': 1,
    }
    df = pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)

    congestion = round(model.predict(df)[0])
    return {
        "congestion": congestion,
        "grade": congestion_grade(congestion),
        "direction": direction,
    }


def recommend_time(date, departure, arrival, minute):
    time_slots = [minute - 60, minute - 30, minute, minute + 30, minute + 60]

    slots = []
    for t in time_slots:
        result = predict_congestion(date, departure, arrival, t)
        slots.append({
            "minute": t,
            "time": f"{(t % 1440) // 60:02d}:{t % 60:02d}",
            "congestion": result["congestion"],
            "grade": result["grade"],
        })

    input_congestion = 0
    for s in slots:
        if s["minute"] == minute:
            input_congestion = s["congestion"]
            break

    best = slots[0]
    for s in slots:
        if s["congestion"] < best["congestion"]:
            best = s

    if best["congestion"] < input_congestion:
        diff = best["minute"] - minute
        if diff < 0:
            timing = f"{abs(diff)}분 일찍 출발하면 더 쾌적합니다"
        else:
            timing = f"{diff}분 늦게 출발하면 더 쾌적합니다"
        message = f"{timing}. {best['time']} 출발 추천"
    else:
        message = "현재 시간대가 가장 적합합니다"

    return {
        "slots": slots,
        "recommendation": {
            "minute": best["minute"],
            "time": best["time"],
            "message": message,
        },
    }


if __name__ == "__main__":
    result = predict_congestion("2026-03-01", "잠실", "강남", 1020)
    print(result)
