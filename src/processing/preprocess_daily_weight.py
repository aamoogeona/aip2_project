import pandas as pd
import os
from preprocess_event import map_day_of_week

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', '..', 'data', 'raw', '2023_2호선_일별승하차량.csv')
df_passenger_raw = pd.read_csv(csv_path, encoding="cp949")

# 승차 + 하차 합산, 승하차 구분 컬럼 제거
df_passenger_all = df_passenger_raw.rename(columns={'수송일자': '날짜'})
df_passenger_all = df_passenger_all.drop(columns=['승하차구분'])
df_passenger_all = df_passenger_all.groupby(['날짜', '역명'], as_index=False).sum()
df_passenger_all['역명'] = df_passenger_all['역명'].str.replace(r'\s*\(.*?\)', '', regex=True)


time_cols = [col for col in df_passenger_all.columns if col not in ['날짜', '역명']]
df_passenger_all = df_passenger_all.melt(
    id_vars=['날짜', '역명'],
    value_vars=time_cols,
    var_name='시간대',
    value_name='인원수'
)

def convert_time(time_str):
    if time_str == '06시이전':
        return 0
    elif time_str == '24시이후':
        return 1440
    else:
        return int(time_str.split('-')[0]) * 60

df_passenger_all['시간대'] = df_passenger_all['시간대'].apply(convert_time)

df_passenger_all_30 = df_passenger_all.copy()
df_passenger_all_30['시간대'] = df_passenger_all_30['시간대'] + 30
df_passenger_all = pd.concat([df_passenger_all, df_passenger_all_30], ignore_index=True)
df_passenger_all['시간대'] = df_passenger_all['시간대'].apply(lambda x: x + 1440 if x < 330 else x)


df_passenger_all['날짜'] = pd.to_datetime(df_passenger_all['날짜'])
df_passenger_all['요일'] = df_passenger_all['날짜'].dt.dayofweek.apply(map_day_of_week)

# baseline: 이벤트 없는 평일
kbo_2023_path = os.path.join(BASE_DIR, '..', '..', 'data', 'raw', '2023_공휴일_KBO_일정_v2.csv')
df_kbo_2023 = pd.read_csv(kbo_2023_path, encoding='utf-8-sig')


special_holidays = [
    '2023-01-01',
    '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',
    '2023-03-01',
    '2023-09-28', '2023-09-29', '2023-09-30', '2023-10-01',
]

time_cols_kbo = [col for col in df_kbo_2023.columns if col not in ['Date', 'Station']]
event_dates = df_kbo_2023[df_kbo_2023[time_cols_kbo].max(axis=1) == 1]['Date'].unique()

#kbo, 공휴일 겹친 날은 제외. event_dates에 포함 
exclude_dates = pd.to_datetime(list(special_holidays) + list(event_dates))

baseline = df_passenger_all[
    ~df_passenger_all['날짜'].isin(exclude_dates)
].groupby(['역명', '요일', '시간대'])['인원수'].mean().reset_index()
baseline.columns = ['역명', '요일', '시간대', 'baseline_인원']

df_merged = df_passenger_all.merge(baseline, on=['역명', '요일', '시간대'], how='left')
df_merged['ratio'] = (df_merged['인원수'] / df_merged['baseline_인원']).fillna(1.0)
df_merged['ratio'] = df_merged['ratio'].clip(upper=3.0)


df_merged['가중치'] = df_merged.groupby(['역명', '요일'])['ratio'].transform(lambda x: x / x.max())

output_path = os.path.join(BASE_DIR, '..', '..', 'data', 'processed', 'daily_weights.csv')
df_merged[['날짜', '역명', '시간대', '가중치']].to_csv(output_path, index=False, encoding='utf-8-sig')

result = pd.read_csv(output_path, encoding='utf-8-sig')
print('shape:', result.shape)
print('날짜 범위:', result['날짜'].min(), '~', result['날짜'].max())
print('가중치 범위:', result['가중치'].min(), '~', result['가중치'].max())
print('역 수:', result['역명'].nunique())