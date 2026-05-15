import sys
sys.path.append('preprocess')
from preprocess2 import load_data
from sklearn.model_selection import train_test_split

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_melted, encoders, station_map = load_data(os.path.join(BASE_DIR, "data", "raw", "혼잡도_정리본.xlsx"))

# feature, label 분리
X = df_melted[['역명', '요일', '상하구분', '시간대']]
y = df_melted['혼잡도']

# 학습, 검증 데이터 8:2 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

#모델 학습
from xgboost import XGBRegressor

model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

print("학습 완료")

from sklearn.metrics import mean_absolute_error

#모델 평가(MAE)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

'''

# 역별 오류값 확인
df_test = X_test.copy()
df_test['실제값'] = y_test.values
df_test['예측값'] = y_pred
df_test['오차'] = abs(df_test['실제값'] - df_test['예측값'])
df_test['역명'] = encoders['역명'].inverse_transform(df_test['역명'])

groupby_station = df_test.groupby('역명')['오차'].mean().reset_index()
groupby_station.columns = ['역명', '평균오차']
print(groupby_station.sort_values('평균오차', ascending=False))


print(df_melted['혼잡도'].min())
print(df_melted['혼잡도'].max())

'''


'''
# 오차 계산
df_test = X_test.copy()
df_test['실제값'] = y_test.values
df_test['예측값'] = y_pred
df_test['오차'] = abs(df_test['실제값'] - df_test['예측값'])

# 시간대별 오차
groupby_time = df_test.groupby('시간대')['오차'].mean().reset_index()
groupby_time.columns = ['시간대(분)', '평균오차']
groupby_time['시간대'] = groupby_time['시간대(분)'].apply(
    lambda x: f"{(x % 1440) // 60:02d}:{x % 60:02d}"
)
print("=== 시간대별 평균오차 ===")
print(groupby_time[['시간대', '평균오차']].sort_values('평균오차', ascending=False).to_string())

print()

# 요일별 오차
df_test['요일명'] = encoders['요일'].inverse_transform(df_test['요일'])
groupby_day = df_test.groupby('요일명')['오차'].mean().reset_index()
groupby_day.columns = ['요일', '평균오차']
print("=== 요일별 평균오차 ===")
print(groupby_day.sort_values('평균오차', ascending=False).to_string())

'''
