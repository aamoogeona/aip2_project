import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'catboost_model.pkl'))
encoders = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'encoders.pkl'))

df_weights = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'processed', 'event_weights.csv'))
df_melted = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'processed', 'df_processed.csv'))



#== 입력 변수 고정(초기 테스트용) ==
station = '잠실'
day = '평일'
direction = '내선'
time = 18 * 60
event_type = 'KBO'


station_enc = encoders['역명'].transform(['잠실'])[0]
day_enc = encoders['요일'].transform(['평일'])[0]
dir_enc = encoders['상하구분'].transform(['내선'])[0]

# == 이상 탐지 == (데이터 처리 시도 후 재구현)
'''
#가중치 조회
weight_row = df_weights[
    (df_weights['역명'] == station) &
    (df_weights['요일'] == day) &
    (df_weights['시간대'] == time) &
    (df_weights['이벤트종류'] == event_type)
]

# 이벤트 변수 세팅
if event_type == 'KBO' and len(weight_row) > 0:
    kbo = 1
    holiday = 0
    coex = 0
    weight = weight_row['가중치'].values[0]
    coex_weight = 0
elif event_type == '공휴일' and len(weight_row) > 0:
    kbo = 0
    holiday = 1
    coex = 0
    weight = weight_row['가중치'].values[0]
    coex_weight = 0
elif event_type == 'COEX':
    # COEX는 별도 파일에서 조회 (날짜별 가중치)
    # 추후 파이프라인에서 날짜 기반으로 조회
    kbo = 0
    holiday = 0
    coex = 1
    weight = 0
    coex_weight = 0  # 임시 - 파이프라인에서 실제 값으로 대체
else:  # none
    kbo = 0
    holiday = 0
    coex = 0
    weight = 0
    coex_weight = 0

# 평소 입력 (이벤트 없음)
input_normal = pd.DataFrame([{
    '역명': station_enc, '요일': day_enc, '상하구분': dir_enc,
    '시간대': time,
    'KBO': 0, '공휴일': 0, 'COEX': 0, '가중치': 0, 'COEX_가중치': 0
}])

# 오늘 입력 (이벤트 반영)
input_today = pd.DataFrame([{
    '역명': station_enc, '요일': day_enc, '상하구분': dir_enc,
    '시간대': time,
    'KBO': kbo, '공휴일': holiday, 'COEX': coex,
    '가중치': weight, 'COEX_가중치': coex_weight
}])

# 예측
pred_normal = model.predict(input_normal)[0]
pred_today = model.predict(input_today)[0]

# 차이 계산
diff = pred_today - pred_normal
diff_pct = (diff / pred_normal) * 100

# 출력
print(f"역명: {station}, 요일: {day}, 시간대: {time//60}시, 방향: {direction}")
print(f"이벤트: {event_type}")
print(f"평소 예측 혼잡도: {pred_normal:.1f}")
print(f"오늘 예측 혼잡도: {pred_today:.1f}")
print(f"차이: {diff:+.1f} ({diff_pct:+.1f}%)")
'''

# == 시간대 추천 ==

target_time = 18 * 60
search_range = 60           # 앞뒤로 몇 분까지 살펴볼지 (90분 = 1시간 30분)
step = 30                    # 시간대 간격 (데이터가 30분 단위)

time_list = list(range(target_time - search_range,
                    target_time + search_range + 1,
                    step))


results = []
for t in time_list:
    input_df = pd.DataFrame([{
        '역명': station_enc, '요일': day_enc, '상하구분': dir_enc,
        '시간대': t,
        'KBO': 0, '공휴일': 0, 'COEX': 0, '가중치': 0, 'COEX_가중치': 0
    }])
    pred = model.predict(input_df)[0]
    results.append({'시간대': t, '예측혼잡도': pred})

# 결과 정리
df_result = pd.DataFrame(results)
df_result['시간대_표시'] = df_result['시간대'].apply(
    lambda x: f"{(x % 1440) // 60:02d}:{x % 60:02d}"
)

# 출력
target_display = f"{(target_time % 1440) // 60:02d}:{target_time % 60:02d}"
print(f"역명: {station}, 요일: {day}, 방향: {direction}")
print(f"기준 시간: {target_display}\n")

print("=== 시간대별 혼잡도 순위 (한산한 순) ===")
for i, row in df_result.iterrows():
    mark = "  ← 기준 시간" if row['시간대'] == target_time else ""
    print(f"{i+1}위  {row['시간대_표시']}  (혼잡도 {row['예측혼잡도']:.1f}){mark}")

best = df_result.iloc[0]
print(f"\n추천: {best['시간대_표시']}가 가장 한산")

'''
print(df_melted[
    (df_melted['역명'] == station_enc) &
    (df_melted['요일'] == day_enc) &
    (df_melted['상하구분'] == dir_enc) &
    (df_melted['시간대'].isin(time_list))
][['시간대', '혼잡도']].drop_duplicates().sort_values('시간대'))
'''