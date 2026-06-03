import csv
import os

from playwright.sync_api import sync_playwright


def crawl_kbo_schedule(season: int, month: int) -> list[dict[str, str]]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_default_timeout(15000)

            print("KBO 일정 페이지에 접속합니다.")
            page.goto(
                "https://www.koreabaseball.com/Schedule/Schedule.aspx",
                wait_until="domcontentloaded",
            )

            print(f"{season}년 {month}월을 선택합니다.")
            page.locator("#ddlYear").select_option(str(season))
            page.locator("#ddlMonth").select_option(f"{month:02d}")
            page.wait_for_load_state("networkidle")
            page.wait_for_selector("#tblScheduleList > tbody > tr")

            rows = page.locator("#tblScheduleList > tbody > tr").all()
            schedules = []
            current_date = ""

            for row in rows:
                cells = [" ".join(cell.inner_text().split()) for cell in row.locator("td").all()]

                if not cells:
                    continue

                if len(cells) == 9:
                    current_date = cells[0]
                    cells = cells[1:]

                if len(cells) < 8:
                    continue

                schedules.append(
                    {
                        "date": current_date,
                        "time": cells[0],
                        "game": cells[1],
                        "game_center": cells[2],
                        "highlight": cells[3],
                        "tv": cells[4],
                        "radio": cells[5],
                        "stadium": cells[6],
                        "note": cells[7],
                    }
                )

            return schedules
        finally:
            browser.close()


def save_schedule_csv(schedules: list[dict[str, str]], filename: str, season: int) -> None:
    with open(filename, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["year", "date", "time", "game", "stadium"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for schedule in schedules:
            writer.writerow(
                {
                    "year": season,
                    "date": schedule["date"],
                    "time": schedule["time"],
                    "game": schedule["game"],
                    "stadium": schedule["stadium"],
                }
            )


if __name__ == "__main__":
    season = 2026
    month = 5
    csv_filename = "kbo_schedule.csv"

    schedules = crawl_kbo_schedule(season, month)
    save_schedule_csv(schedules, csv_filename, season)

    print(f"{season}년 {month}월 KBO 일정 {len(schedules)}개 행을 저장했습니다.")
    print(f"저장 파일: {csv_filename}")










"""
코엑스 행사 일정 스크랩 코드
- 날짜별로 코엑스 행사 여부를 0/1로 반환
- 코엑스 사이트: https://www.coex.co.kr/event/full-schedules/
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import time
import re


COEX_BASE_URL = "https://www.coex.co.kr/event/full-schedules/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://www.coex.co.kr/",
}


# 1. 행사 목록 파싱

def fetch_events_for_period(start_date: str, end_date: str) -> list[dict]:
    """
    코엑스 행사 일정 페이지에서 특정 기간의 행사를 가져옴.

    Parameters
    ----------
    start_date : str  "YYYY.MM.DD"
    end_date   : str  "YYYY.MM.DD"

    Returns
    -------
    list of dict  [{"name": ..., "start": date, "end": date}, ...]
    """
    params = {
        "search_start_date": start_date,
        "search_end_date": end_date,
        "list_type": "LIST",
    }

    try:
        resp = requests.get(COEX_BASE_URL, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] 요청 실패: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # 행사 카드 파싱 (li 안에 날짜 텍스트 포함)
    # 패턴 예시: "2026.05.20 - 2026.05.22"
    date_pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2})\s*[-–]\s*(\d{4}\.\d{2}\.\d{2})")

    events = []
    # 행사 링크가 담긴 li 요소들 수집
    list_items = soup.select("ul li a[href*='/exhibitions/']")

    seen = set()
    for a_tag in list_items:
        text = a_tag.get_text(" ", strip=True)
        match = date_pattern.search(text)
        if match:
            raw_start, raw_end = match.group(1), match.group(2)
            key = (text[:30], raw_start, raw_end)
            if key in seen:
                continue
            seen.add(key)

            event_start = datetime.strptime(raw_start, "%Y.%m.%d").date()
            event_end   = datetime.strptime(raw_end,   "%Y.%m.%d").date()

            # 행사명 추출 (날짜 이전 텍스트)
            name_part = date_pattern.sub("", text).strip()
            # 중복된 타입 접두어 제거 (Exhibition, Convention 등)
            name_part = re.sub(r"^(Exhibition|Convention|Pop-up/Event)\s*", "", name_part, flags=re.IGNORECASE).strip()

            events.append({
                "name": name_part[:60],
                "start": event_start,
                "end": event_end,
            })

    print(f"[INFO] {start_date} ~ {end_date}: {len(events)}개 행사 수집")
    return events


# 2. 날짜별 0/1 생성

def build_event_flag(
    start_date: str,
    end_date: str,
    chunk_months: int = 1,
) -> pd.DataFrame:
    """
    start_date ~ end_date 범위의 날짜별 코엑스 행사 여부를 DataFrame으로 반환.

    Parameters
    ----------
    start_date   : "YYYY-MM-DD"
    end_date     : "YYYY-MM-DD"
    chunk_months : 한 번에 요청할 개월 수 (기본 1 → 부하 분산)

    Returns
    -------
    pd.DataFrame  columns: ["date", "coex_event"]
                coex_event: 0 or 1
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    all_events: list[dict] = []

    # 기간을 chunk_months 단위로 분할 요청 (서버 부하 방지)
    cursor = start_dt
    while cursor <= end_dt:
        chunk_end = min(
            end_dt,
            (cursor.replace(day=1) + timedelta(days=32 * chunk_months)).replace(day=1) - timedelta(days=1),
        )
        all_events.extend(
            fetch_events_for_period(
                cursor.strftime("%Y.%m.%d"),
                chunk_end.strftime("%Y.%m.%d"),
            )
        )
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # 요청 간격 (과도한 요청 방지)

    # 날짜 범위 생성
    all_dates = pd.date_range(start=start_dt, end=end_dt, freq="D").date

    # 각 날짜에 행사 여부 표시
    event_flag = {d: 0 for d in all_dates}

    for ev in all_events:
        day = ev["start"]
        while day <= ev["end"]:
            if day in event_flag:
                event_flag[day] = 1
            day += timedelta(days=1)

    df = pd.DataFrame(
        {"date": list(event_flag.keys()), "coex_event": list(event_flag.values())}
    ).sort_values("date").reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"])

    return df


# 3. 단일 날짜 조회 유틸

def is_coex_event_day(date: str) -> int:
    """
    특정 날짜에 코엑스 행사가 있으면 1, 없으면 0 반환.

    Parameters
    ----------
    date : "YYYY-MM-DD"

    Returns
    -------
    int  0 or 1
    """
    df = build_event_flag(date, date)
    return int(df["coex_event"].iloc[0])


# 4. CSV 저장

def save_event_flag_csv(start_date: str, end_date: str, output_path: str = "coex_event_flag.csv"):
    """
    날짜별 코엑스 행사 여부를 CSV로 저장.

    Parameters
    ----------
    start_date  : "YYYY-MM-DD"
    end_date    : "YYYY-MM-DD"
    output_path : 저장할 파일 경로
    """
    df = build_event_flag(start_date, end_date)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[✓] 저장 완료: {output_path} ({len(df)}일)")
    print(df.tail(10).to_string(index=False))
    return df


# 5. 실행 예시

if __name__ == "__main__":
    # 예시: 2026년 5~6월 데이터 수집
    df = save_event_flag_csv(
        start_date="2026-05-01",
        end_date="2026-06-30",
        output_path="coex_event_flag.csv",
    )

    # 행사 있는 날만 출력
    print("\n[행사 있는 날]")
    print(df[df["coex_event"] == 1].to_string(index=False))

    # 단일 날짜 조회 예시
    target = "2026-05-22"
    flag = is_coex_event_day(target)
    print(f"\n{target} 코엑스 행사 여부: {flag}")


# 크롤링 이벤트 업데이트 

from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JAMSIL_STADIUM = "잠실"
JAMSIL_STATIONS = ["잠실역", "종합운동장역"]
ALL_HOUR_SLOTS = [f"{h:02d}-{h+1:02d}" for h in range(3, 24)] + ["00-01", "01-02", "02-03"]


def _parse_kbo_date(date_str: str, year: int) -> str:
    """'03.01(일)' → '2026-03-01'. 파싱 실패 시 None."""
    m = re.match(r"(\d{1,2})\.(\d{1,2})", date_str)
    if not m:
        return None
    return f"{year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"


def kbo_to_station_slots(
    schedules: list[dict], start_date: str, end_date: str, year: int
) -> pd.DataFrame:
    """KBO raw 일정 → 잠실역·종합운동장역 시간대 슬롯 DataFrame."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    all_dates = [
        (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end_dt - start_dt).days + 1)
    ]

    rows = [
        {"Date": d, "Station": s, **{slot: 0 for slot in ALL_HOUR_SLOTS}}
        for d in all_dates
        for s in JAMSIL_STATIONS
    ]
    df = pd.DataFrame(rows)

    for game in schedules:
        parsed = _parse_kbo_date(game.get("date", ""), year)
        if parsed not in all_dates:
            continue
        if JAMSIL_STADIUM not in game.get("stadium", ""):
            continue
        try:
            game_hour = int(game["time"].split(":")[0])
        except (ValueError, IndexError, KeyError):
            continue

        for h in (game_hour - 1, game_hour, game_hour + 3):
            if not 0 <= h <= 23:
                continue
            slot = f"{h:02d}-{h+1:02d}"
            df.loc[df["Date"] == parsed, slot] = 1

    return df


def update_event_schedule(days_ahead: int = 7) -> None:
    """
    오늘부터 days_ahead일치 이벤트 파일 갱신.
    data/processed/upcoming_kbo_holiday.csv  (2026_공휴일_KBO 동일 포맷)
    data/processed/upcoming_coex.csv         (date, coex_event)
    """
    today = date.today()
    end_dt = today + timedelta(days=days_ahead)
    start_str = today.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # KBO 크롤링 (월 경계 대응)
    all_games: list[dict] = []
    for m in sorted({today.month, end_dt.month}):
        try:
            all_games.extend(crawl_kbo_schedule(today.year, m))
        except Exception as e:
            print(f"[WARN] KBO {m}월 크롤링 실패: {e}")
    df_kbo = kbo_to_station_slots(all_games, start_str, end_str, today.year)

    # 공휴일: 정적 2026 파일에서 해당 기간 슬라이싱
    df_h = pd.read_csv(
        os.path.join(BASE_DIR, "..", "data", "raw", "2026_공휴일_KBO_일정_v2.csv")
    )
    df_holiday = df_h[
        df_h["Date"].between(start_str, end_str) &
        df_h["Station"].isin(["강남역", "잠실역", "홍대입구역"])
    ].copy()

    # KBO + 공휴일 병합: 같은 (Date, Station) 슬롯은 OR(max)
    df_schedule = (
        pd.concat([df_kbo, df_holiday], ignore_index=True)
        .groupby(["Date", "Station"], as_index=False)[ALL_HOUR_SLOTS]
        .max()
    )
    out_sch = os.path.join(BASE_DIR, "..", "data", "processed", "upcoming_kbo_holiday.csv")
    df_schedule.to_csv(out_sch, index=False, encoding="utf-8-sig")
    print(f"[✓] KBO+공휴일: {out_sch} ({len(df_schedule)}행)")

    # COEX: 기존 build_event_flag 활용 (사이트 크롤링)
    df_coex = build_event_flag(start_str, end_str)
    out_coex = os.path.join(BASE_DIR, "..", "data", "processed", "upcoming_coex.csv")
    df_coex.to_csv(out_coex, index=False, encoding="utf-8-sig")
    print(f"[✓] COEX: {out_coex} ({len(df_coex)}행)")