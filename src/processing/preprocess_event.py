import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', '..', 'data', 'raw', '2026_공휴일_KBO_일정_v2.csv')
df_kbo = pd.read_csv(csv_path, encoding="utf-8-sig")

def map_day_of_week(day):
        if day <= 4: 
            return '평일'
        elif day == 5: 
            return '토요일'
        else:
            return '일요일'

def load_event_data():
    df = df_kbo.rename(columns={'Date': '날짜', 'Station': '역명'})
    df['역명'] = df['역명'].str.replace('역', '', regex=False)
    time_cols = [col for col in df.columns if '-' in col]

    #melt로 wide -> long 변환
    df_event = df.melt(
        id_vars=['날짜', '역명'],
        value_vars=time_cols,
        var_name='시간대',
        value_name='이벤트'
    )

    #시간대 문자열 -> 분 단위 숫자 변환(30분 단위 확장)
    df_event['시간대'] = df_event['시간대'].str.split('-').str[0].astype(int) * 60
    df_event_30 = df_event.copy()
    df_event_30['시간대'] = df_event_30['시간대'] + 30
    df_event = pd.concat([df_event, df_event_30], ignore_index=True)
    df_event['시간대'] = df_event['시간대'].apply(lambda x: x + 1440 if x < 330 else x)

    #요일 구분 매핑
    df_event['요일'] = pd.to_datetime(df_event['날짜'])
    df_event['요일'] = df_event['요일'].dt.dayofweek.apply(map_day_of_week)

    df_event['날짜'] = pd.to_datetime(df_event['날짜'])
    return df_event


#print(df_event.head(10))
#print('shape:', df_event.shape)