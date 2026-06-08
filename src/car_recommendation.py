import os
import pandas as pd
from prediction import LINE2_STATIONS, station_idx, N_STATIONS, get_weekday

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

THRESHOLD = 51  # 혼잡 이상부터 칸 추천

_df_cars = pd.read_excel(
    os.path.join(BASE_DIR, '..', 'data', 'raw', '환승역_칸_v2.xlsx')
).drop_duplicates(subset=["역명", "종착역명", "환승선", "칸"])

_df_transfer_count = pd.read_excel(
    os.path.join(BASE_DIR, '..', 'data', 'raw', '환승역_환승인원.xlsx')
)

# 환승역_칸_v2 역명 정규화 ("교대(법원.검찰청)" -> "교대")
def _normalize_station(name):
    name = str(name).split('(')[0].strip()
    name = name.replace("을지로3가", "을지로 3가")
    name = name.replace("을지로4가", "을지로 4가")
    name = name.replace("을지로입구", "을지로 입구")
    return name

_df_cars['역명'] = _df_cars['역명'].apply(_normalize_station)
_df_cars = _df_cars[_df_cars['역명'].isin(LINE2_STATIONS)].reset_index(drop=True)

TRANSFER_STATIONS = set(_df_cars['역명'].unique())


def _get_route(departure, arrival, direction):
    # 출발역부터 도착역까지 지나는 역 목록 반환 (출발역 제외, 도착역 포함)
    if departure not in station_idx or arrival not in station_idx:
        return []

    dep_idx = station_idx[departure]
    arr_idx = station_idx[arrival]
    stations = []

    if direction == "외선":
        idx = (dep_idx + 1) % N_STATIONS
        while True:
            stations.append(LINE2_STATIONS[idx])
            if idx == arr_idx:
                break
            idx = (idx + 1) % N_STATIONS
    else:
        idx = (dep_idx - 1) % N_STATIONS
        while True:
            stations.append(LINE2_STATIONS[idx])
            if idx == arr_idx:
                break
            idx = (idx - 1) % N_STATIONS

    return stations


def recommend_car(date, departure, arrival, direction, congestion):
    if congestion < THRESHOLD:
        return {
            "active": False,
            "message": "현재 혼잡도가 낮아 어느 칸이든 자리가 있습니다",
        }

    route = _get_route(departure, arrival, direction)
    if not route:
        return {"active": False, "message": "경로를 계산할 수 없습니다"}

    N = len(route) // 2
    if N == 0:
        return {"active": False, "message": "경로가 짧아 칸 추천이 어렵습니다"}

    # 앞 N개 역 중 첫 번째 환승역 찾기
    first_transfer = None
    for s in route[:N]:
        if s in TRANSFER_STATIONS:
            first_transfer = s
            break

    if first_transfer is None:
        return {"active": False, "message": "경로 내 환승역이 없어 칸 추천이 어렵습니다"}

    rows = _df_cars[
        (_df_cars['역명'] == first_transfer) &
        (_df_cars['종착역명'] == direction)
    ]
    if rows.empty:
        return {"active": False, "message": "해당 역 칸 정보가 없습니다"}

    # 요일별 환승 인원 10만 이상이면 높은 확률
    weekday = get_weekday(date)
    count_row = _df_transfer_count[_df_transfer_count['출발역명'] == first_transfer]
    if not count_row.empty and int(count_row[weekday].values[0]) >= 100000:
        confidence = "높습니다"
    else:
        confidence = "있습니다"

    recommendations = []
    for _, row in rows.iterrows():
        transfer_line = row["환승선"]
        recommendations.append({
            "car": int(row["칸"]),
            "door": int(row["차량출입문번호"]),
            "transfer_line": str(transfer_line) if pd.notna(transfer_line) else None,
        })

    if len(recommendations) == 1:
        r = recommendations[0]
        message = (
            f"{first_transfer}역 환승객이 많아 자리가 날 확률이 {confidence}. "
            f"{r['car']}번 칸 추천"
        )
    else:
        car_parts_list = []
        for r in recommendations:
            car_parts_list.append(f"{r['transfer_line']} 방향 {r['car']}번 칸")
        car_parts = ", ".join(car_parts_list)
        message = (
            f"{first_transfer}역 환승객이 많아 자리가 날 확률이 {confidence}. "
            f"{car_parts} 추천"
        )

    return {
        "active": True,
        "transfer_station": first_transfer,
        "recommendations": recommendations,
        "message": message,
    }


if __name__ == "__main__":
    test_cases = [
        ("2026-06-04", "강남",     "신촌",   "외선", 65),
        ("2026-06-04", "잠실",     "강남",   "외선", 70),
        ("2026-06-04", "홍대입구", "강남",   "내선", 35),
        ("2026-06-04", "강남",     "역삼",   "외선", 80),
    ]
    for date, dep, arr, direc, cong in test_cases:
        result = recommend_car(date, dep, arr, direc, cong)
        print(f"\n{dep}→{arr} ({direc}, 혼잡도={cong})")
        print(f"  active: {result['active']}")
        print(f"  {result['message']}")
        if result.get("recommendations"):
            for r in result["recommendations"]:
                print(f"    {r['transfer_line']} → {r['car']}번 칸 {r['door']}번 문")
