import pandas as pd
import os
import datetime
from sklearn.preprocessing import LabelEncoder
from preprocess_event import map_day_of_week, load_event_data

def load_data(filepath="혼잡도_정리본2.xlsx"):
    df = pd.read_excel(filepath)
    
    # 컬럼명 정리
    df = df.rename(columns={'역명 |시간대': '역명', '요일구분': '요일'})

    # 시간대 컬럼 이름을 "HH:MM" 문자열로 변환
    col_rename = {}
    for col in df.columns:
        if isinstance(col, datetime.time):
            col_rename[col] = col.strftime('%H:%M')
        elif isinstance(col, datetime.timedelta):
            total_minutes = int(col.total_seconds() // 60)
            col_rename[col] = f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"
    df = df.rename(columns=col_rename)

    # 병합 셀로 인한 역명, 요일 결측치를 앞 값으로 채우기
    df[['역명', '요일']] = df[['역명', '요일']].ffill()

    # melt로 wide -> long 변환
    import re
    id_vars = ['역명', '요일', '상하구분']
    value_vars = [col for col in df.columns if re.match(r'^\d{2}:\d{2}$', str(col))]
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='시간대',
        value_name='혼잡도'
    )

    # 공백 문자열을 NaN으로 변환 후 제거
    df_melted['혼잡도'] = pd.to_numeric(df_melted['혼잡도'], errors='coerce')
    df_melted = df_melted.dropna(subset=['혼잡도'])

    # 시간대 "HH:MM" -> 분 단위 숫자로 변환
    df_melted['시간대'] = df_melted['시간대'].apply(
    lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    )
    df_melted['시간대'] = df_melted['시간대'].apply(
    lambda x: x + 1440 if x < 330 else x
    )

    #2026년 날짜 테이블 만들기 
    dates = pd.date_range(start='2026-01-01', end='2026-12-31')
    df_dates = pd.DataFrame({'날짜': dates})
    df_dates['요일'] = df_dates['날짜'].dt.dayofweek.apply(map_day_of_week)

    #요일 정보 병합
    df_melted = pd.merge(df_melted, df_dates, on='요일')

    #공휴일 kbo 데이터 병합
    df_event = load_event_data()
    df_melted = pd.merge(df_melted, df_event[['날짜', '역명', '시간대', '이벤트']],
                        on=['날짜', '역명', '시간대'],
                        how='left')
    df_melted['이벤트'] = df_melted['이벤트'].fillna(0)

    kbo_dates = set(df_event[
        (df_event['역명'] == '종합운동장') & 
        (df_event['이벤트'] == 1)
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
        
    df_melted['이벤트종류'] = df_melted.apply(classify_event, axis=1)
    df_melted['KBO'] = df_melted['이벤트종류'].apply(lambda x: 1 if x == 'KBO' else 0)
    df_melted['공휴일'] = df_melted['이벤트종류'].apply(lambda x: 1 if x == '공휴일' else 0)

    #혼잡도 가중치 보정
    df_weights = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'event_weights.csv'))
    df_melted = pd.merge(df_melted, df_weights[['역명', '요일', '시간대', '가중치', '이벤트종류']], on=['역명', '요일', '시간대', '이벤트종류'], how='left')
    df_melted['가중치'] = df_melted['가중치'].fillna(0)
    #df_melted['혼잡도'] = df_melted['혼잡도'] * (1 + df_melted['가중치'])

    df_melted = df_melted.drop(columns=['날짜', '이벤트종류', '이벤트'])

    # 문자열 숫자로 인코딩
    encoders = {}
    for col in ['역명', '요일', '상하구분']:
        le = LabelEncoder()
        df_melted[col] = le.fit_transform(df_melted[col])
        encoders[col] = le

    # 역명 코드 매핑 테이블 저장
    station_map = pd.DataFrame({
        '역_코드': range(len(encoders['역명'].classes_)),
        '역명': encoders['역명'].classes_
    })

    return df_melted, encoders, station_map

