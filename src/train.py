import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processing'))
from preprocess import load_data
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_melted, encoders, station_map = load_data(os.path.join(BASE_DIR, '..', 'data', 'raw', '혼잡도_정리본_2.xlsx'))

# feature, label 분리
X = df_melted[['역명', '요일', '상하구분', '시간대', 'KBO', '공휴일', '가중치']]
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


# 오차 계산
df_test = X_test.copy()
df_test['실제값'] = y_test.values
df_test['예측값'] = y_pred
df_test['오차'] = abs(df_test['실제값'] - df_test['예측값'])


#이벤트 역별 오차 반영 확인
# 인코딩 풀어서 비교
df_test['역명_원본'] = encoders['역명'].inverse_transform(df_test['역명'])

'''
target_stations = ['잠실', '종합운동장', '홍대입구', '강남']
df_target = df_test[df_test['역명_원본'].isin(target_stations)]
print("=== 역별 실제값 vs 예측값 ===")
print(df_target.groupby(['역명_원본', 'KBO', '공휴일'])[['실제값', '예측값']].mean())

#피처 중요도 확인
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
print("=== 피처 중요도 ===")
print(feat_imp.sort_values(ascending=False))

print("이벤트 값 분포:")
print(df_melted['KBO'].value_counts())
print(df_melted['공휴일'].value_counts())
'''

# 이벤트일/비이벤트일 MAE 분리
df_test['이벤트있음'] = df_test['KBO']==1
print("=== 이벤트일/비이벤트일 MAE ===")
print(df_test.groupby('이벤트있음')['오차'].agg(['mean', 'count']))

