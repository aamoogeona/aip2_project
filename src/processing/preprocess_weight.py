import pandas as pd
import os
from preprocess_event import map_day_of_week

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', '..', 'data', 'raw', '2023_2호선_일별승하차량.csv')
df_passenger_raw = pd.read_csv(csv_path, encoding="cp949")

kbo_2023_path = os.path.join(BASE_DIR, '..', '..', 'data', 'raw', '2023_공휴일_KBO_일정_v2.csv')
df_kbo_2023 = pd.read_csv(kbo_2023_path, encoding='utf-8-sig')

def load_weight_data():
    #승하차 데이터 전처리
    df_passenger = df_passenger_raw.rename(columns={'수송일자': '날짜'})
    df_passenger = df_passenger.drop(columns=['승하차구분'])
    df_passenger = df_passenger.groupby(['날짜', '역명'], as_index=False).sum()
    df_passenger['역명'] = df_passenger['역명'].str.replace(r'\s*\(.*?\)', '', regex=True)

    #wide->long 변환
    df_passenger_long = df_passenger.melt(id_vars=['날짜', '역명'], var_name='시간대', value_name='인원수')

    #시간대 문자열 -> 분 단위 숫자 변환(30분 단위 확장)
    def convert_time(time_str):
        if time_str == '06시이전':
            return 0
        elif time_str == '24시이후':
            return 1440
        else:
            return int(time_str.split('-')[0]) * 60

    df_passenger_long['시간대'] = df_passenger_long['시간대'].apply(convert_time)
    df_passenger_long_30 = df_passenger_long.copy()
    df_passenger_long_30['시간대'] = df_passenger_long_30['시간대'] + 30
    df_passenger_long = pd.concat([df_passenger_long, df_passenger_long_30], ignore_index=True)
    df_passenger_long['시간대'] = df_passenger_long['시간대'].apply(lambda x: x + 1440 if x < 330 else x)

    #KBO/공휴일 이벤트 데이터 전처리
    df_kbo = df_kbo_2023.rename(columns={'Date': '날짜', 'Station': '역명'})
    df_kbo['역명'] = df_kbo['역명'].str.replace('역', '', regex=False)
    time_cols = [col for col in df_kbo.columns if '-' in col]

    #wide->long 변환
    df_event_long = df_kbo.melt(
        id_vars=['날짜', '역명'],
        value_vars=time_cols,
        var_name='시간대',
        value_name='이벤트'
    )

    #시간대 문자열 -> 분 단위 숫자 변환(30분 단위 확장)
    df_event_long['시간대'] = df_event_long['시간대'].str.split('-').str[0].astype(int) * 60
    df_event_long_30 = df_event_long.copy()
    df_event_long_30['시간대'] = df_event_long_30['시간대'] + 30
    df_event_long = pd.concat([df_event_long, df_event_long_30], ignore_index=True)
    df_event_long['시간대'] = df_event_long['시간대'].apply(lambda x: x + 1440 if x < 330 else x)

    #요일 구분 매핑
    df_event_long['요일'] = pd.to_datetime(df_event_long['날짜']).dt.dayofweek.apply(map_day_of_week)
    df_event_long['날짜'] = pd.to_datetime(df_event_long['날짜'])
    df_passenger_long['날짜'] = pd.to_datetime(df_passenger_long['날짜'])

    df_merged = pd.merge(df_passenger_long, df_event_long, on=['날짜', '역명', '시간대'], how='left')
    df_merged['이벤트'] = df_merged['이벤트'].fillna(0)
    df_merged['요일'] = df_merged['날짜'].dt.dayofweek.apply(map_day_of_week)


    # 이벤트 종류 컬럼 추가 (잠실-KBO, 나머지-공휴일)
    kbo_dates = set(df_event_long[
        (df_event_long['역명'] == '종합운동장') & 
        (df_event_long['이벤트'] == 1)
        ]['날짜'].unique())

    def classify_event(row):
        if row['이벤트'] == 0:
            return 'none'
        elif row['역명'] == '종합운동장':
            return 'KBO'
        elif row['역명'] == '잠실' and row['날짜'] in kbo_dates:
            return 'KBO'
        else:
            return '공휴일'


    df_merged['이벤트종류'] = df_merged.apply(classify_event, axis=1)

    overlap_dates = [
        # KBO + 공휴일 겹친 날
        '2023-06-06', '2023-08-15', '2023-10-02', '2023-10-03', '2023-10-09',
        # 명절/특수 연휴 (baseline 오염 방지)
        '2023-01-01',  # 신정
        '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',  # 설 연휴
        '2023-03-01',  # 삼일절
        '2023-09-28', '2023-09-29', '2023-09-30', '2023-10-01',  # 추석 연휴
        ]

    df_for_ratio = df_merged[
        ~df_merged['날짜'].isin(pd.to_datetime(overlap_dates))
        ].copy()

    return df_for_ratio

df_for_ratio = load_weight_data()

#평소 데이터 확인
baseline = df_for_ratio[df_for_ratio['이벤트'] == 0].groupby(
    ['역명', '요일', '시간대']
    )['인원수'].mean().reset_index()
baseline.columns = ['역명', '요일', '시간대', 'baseline_인원']

#이벤트 데이터에 baseline 인원수 병합
df_event = df_for_ratio[df_for_ratio['이벤트'] == 1].copy()
df_event = df_event.merge(baseline, on=['역명', '요일', '시간대'], how='left')

#이벤트별 승하차량 ratio 계산
df_event['ratio'] = df_event['인원수'] / df_event['baseline_인원']

ratio_summary = df_event.groupby(
    ['이벤트종류', '역명', '요일', '시간대']
    )['ratio'].agg(['mean', 'std', 'count']).reset_index()

#0~1 정규화(이벤트종류별로 max로 나눔)
ratio_summary['가중치'] = ratio_summary.groupby('이벤트종류')['mean'].transform(
    lambda x: x / x.max()
)
output_path = os.path.join(BASE_DIR, '..', '..', 'data', 'processed', 'event_weights.csv')
ratio_summary.to_csv(output_path, index=False, encoding='utf-8-sig')


