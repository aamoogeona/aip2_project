import sys
sys.path.append('preprocess')
from preprocess import load_data
from sklearn.model_selection import train_test_split

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_melted, encoders, station_map = load_data(os.path.join(BASE_DIR, "data", "역별지하철_혼잡도_정리본-1 (1).xlsx"))

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

# 역별 오류값 확인
df_test = X_test.copy()
df_test['실제값'] = y_test.values
df_test['예측값'] = y_pred
df_test['오차'] = abs(df_test['실제값'] - df_test['예측값'])
df_test['역명'] = encoders['역명'].inverse_transform(df_test['역명'])

groupby_station = df_test.groupby('역명')['오차'].mean().reset_index()
groupby_station.columns = ['역명', '평균오차']
print(groupby_station.sort_values('평균오차', ascending=False))

#print(station_map)