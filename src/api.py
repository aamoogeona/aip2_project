"""
2호선 혼잡도 예측 API 서버
- 팀원이 구현한 src/ 모듈(prediction, event_detection, car_recommendation)을 통합
- 프론트엔드와 JSON으로 통신

실행 위치: src/ 폴더 안에서 실행
    cd src
    uvicorn api:app --reload --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from datetime import date as date_type, date, timedelta

# ── 팀원 구현 모듈 import ──────────────────────────────
# api.py를 src/ 폴더 안에 두면 아래 import가 그대로 동작
from prediction import (
    predict_congestion,
    recommend_time,
    get_direction,
    get_weekday,
    LINE2_STATIONS,
)
from event_detection import check_event
from car_recommendation import recommend_car


app = FastAPI(title="2호선 혼잡도 예측 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 배포 시 실제 프론트 주소로 변경
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청 ──────────────────────────────────────────
class RecommendRequest(BaseModel):
    departure: str    # 출발역 (예: "강남")
    arrival: str      # 도착역 (예: "홍대입구")
    date: str         # 날짜 (예: "2026-06-05")
    time: str         # 출발 시간 (예: "18:30")

    @field_validator('time')
    @classmethod
    def validate_time(cls, v):
        try:
            h, m = v.split(':')
            h, m = int(h), int(m)
            assert 0 <= h <= 23 and m in (0, 30)
        except Exception:
            raise ValueError("시간은 'HH:MM' 형식, 30분 단위여야 합니다 (예: 18:00)")
        return v

    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        try:
            date_type.fromisoformat(v)
        except Exception:
            raise ValueError("날짜는 'YYYY-MM-DD' 형식이어야 합니다")
        return v


# ── 유틸 ─────────────────────────────────────────────────
def time_to_minute(time_str: str) -> int:
    """'HH:MM' → 분 단위 정수. 새벽(05:30 이전)은 +1440 보정."""
    h, m = time_str.split(':')
    total = int(h) * 60 + int(m)
    if total < 330:           # 학습 데이터와 동일한 새벽 보정
        total += 1440
    return total


def minute_to_time(minute: int) -> str:
    """분 단위 → 'HH:MM'."""
    actual = minute % 1440
    return f"{actual // 60:02d}:{actual % 60:02d}"


# ── 메인 엔드포인트 ──────────────────────────────────────
@app.post("/recommend")
def recommend(req: RecommendRequest):
    # 1. 역 유효성 검증
    if req.departure not in LINE2_STATIONS:
        raise HTTPException(400, f"'{req.departure}'은(는) 2호선 역이 아닙니다.")
    if req.arrival not in LINE2_STATIONS:
        raise HTTPException(400, f"'{req.arrival}'은(는) 2호선 역이 아닙니다.")
    if req.departure == req.arrival:
        raise HTTPException(400, "출발역과 도착역이 같습니다.")

    # 2. 날짜 범위 검증 (오늘 ~ 7일 이내)
    target = date_type.fromisoformat(req.date)
    today = date.today()
    if not (today <= target <= today + timedelta(days=7)):
        raise HTTPException(400, "날짜는 오늘부터 7일 이내만 선택 가능합니다.")

    minute = time_to_minute(req.time)

    try:
        # 3. 혼잡도 예측 (방향·등급 포함)
        pred = predict_congestion(req.date, req.departure, req.arrival, minute)
        congestion = pred["congestion"]
        grade = pred["grade"]
        direction = pred["direction"]

        # 4. 시간대 추천 (±60분)
        time_result = recommend_time(req.date, req.departure, req.arrival, minute)

        # 5. 칸 추천 (등급 3~5에서만 활성화)
        car_result = recommend_car(
            req.date, req.departure, req.arrival, direction, congestion
        )

        # 6. 이벤트 감지 (출발역 기준)
        event_result = check_event(req.date, req.departure, minute)

    except Exception as e:
        raise HTTPException(500, f"예측 처리 중 오류: {e}")

    # 7. 응답 조립
    return {
        "input": {
            "departure": req.departure,
            "arrival": req.arrival,
            "date": req.date,
            "time": req.time,
            "weekday": get_weekday(req.date),
        },
        "congestion": congestion,
        "grade": grade,
        "direction": direction,
        "time_recommendation": time_result,
        "car_recommendation": car_result,
        "event": event_result,
    }


# ── 이벤트만 조회 (날짜 선택 시 프론트에서 호출) ─────────
class EventRequest(BaseModel):
    date: str
    station: str
    time: str


@app.post("/check_event")
def check_event_api(req: EventRequest):
    if req.station not in LINE2_STATIONS:
        raise HTTPException(400, f"'{req.station}'은(는) 2호선 역이 아닙니다.")
    minute = time_to_minute(req.time)
    return check_event(req.date, req.station, minute)


# ── 역 목록 ──────────────────────────────────────────────
@app.get("/stations")
def get_stations():
    return {"stations": LINE2_STATIONS}


@app.get("/health")
def health():
    return {"status": "ok"}
