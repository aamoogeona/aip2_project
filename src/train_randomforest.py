import sys
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rc('axes', unicode_minus=False)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'processing'))
from preprocess import load_data

# =============================================
# Step 1: estimated_daily_usage 로드
# =============================================
estimated_path = os.path.join(BASE_DIR, '..', 'data', 'processed', 'estimated_daily_usage.csv')
df_est = pd.read_csv(estimated_path, encoding='utf-8-sig')
df_est['날짜'] = pd.to_datetime(df_est['날짜'])
df_est = df_est.dropna(subset=['시간대_y'])
df_est['시간대_y'] = df_est['시간대_y'].astype(int)

print(f"[Step 1] estimated 행 수: {len(df_est)}")
print(f"         Estimated 범위: {df_est['Estimated'].min():.1f} ~ {df_est['Estimated'].max():.1f}")

# =============================================
# Step 2: 이벤트 컬럼 추출
# =============================================
df_melted, encoders, station_map = load_data(
    os.path.join(BASE_DIR, '..', 'data', 'raw', '혼잡도_정리본_2.xlsx')
)
df_melted['날짜'] = pd.to_datetime(df_melted['날짜'])
df_melted['역명_원본'] = encoders['역명'].inverse_transform(df_melted['역명'])
df_melted['요일_원본'] = encoders['요일'].inverse_transform(df_melted['요일'])
df_melted['상하구분_원본'] = encoders['상하구분'].inverse_transform(df_melted['상하구분'])

df_events = df_melted[['역명_원본', '요일_원본', '상하구분_원본', '시간대', '날짜',
                    'KBO', '공휴일', 'COEX']].copy()

# =============================================
# Step 3: estimated에 이벤트 컬럼 병합
# =============================================
df_est_renamed = df_est.rename(columns={
    '요일구분': '요일_원본',
    '역명': '역명_원본',
    '상하구분': '상하구분_원본',
    '시간대_y': '시간대',
})

df_data = pd.merge(
    df_est_renamed[['요일_원본', '역명_원본', '상하구분_원본', '시간대', '날짜', 'Estimated']],
    df_events,
    on=['요일_원본', '역명_원본', '상하구분_원본', '시간대', '날짜'],
    how='left'
)

df_data = df_data.dropna(subset=['Estimated'])
for col in ['KBO', '공휴일', 'COEX']:
    df_data[col] = df_data[col].fillna(0).astype(int)

for col, orig_col in [('역명', '역명_원본'), ('요일', '요일_원본'), ('상하구분', '상하구분_원본')]:
    df_data[orig_col] = df_data[orig_col].astype(str).str.strip()
    df_data[col] = encoders[col].transform(df_data[orig_col])

# =============================================
# Step 4: 날짜 기준 train/test 분할 (월별로)
# =============================================
df_data = df_data.drop(columns=['역명_원본', '요일_원본', '상하구분_원본'])

dates = df_data['날짜'].drop_duplicates().reset_index(drop=True)
months = pd.to_datetime(dates).dt.month
train_dates, test_dates = train_test_split(
    dates, test_size=0.2, random_state=42, stratify=months
)
train_mask = df_data['날짜'].isin(set(train_dates))
df_split_train = df_data[train_mask]
df_split_test = df_data[~train_mask]

# =============================================
# Step 5: 피처 구성, 원핫 인코딩
# =============================================
feature_cols = ['역명', '요일', '상하구분', '시간대', 'KBO', '공휴일', 'COEX']

X_train = pd.get_dummies(
    df_split_train[feature_cols],
    columns=['역명', '요일', '상하구분'], dtype=int
)
X_test = pd.get_dummies(
    df_split_test[feature_cols],
    columns=['역명', '요일', '상하구분'], dtype=int
)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
y_train = df_split_train['Estimated']
y_test = df_split_test['Estimated']

print(f"\n[Step 5] X_train: {X_train.shape}, X_test: {X_test.shape}")

# =============================================
# Step 6: RandomForest 학습 및 평가
# =============================================
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"\n[Step 6] 전체 MAE: {mae:.4f}")

df_result = df_split_test.copy()
df_result['예측값'] = y_pred
df_result['오차'] = abs(df_result['Estimated'] - y_pred)
df_result['이벤트있음'] = (df_result['KBO'] == 1) | (df_result['COEX'] == 1)

event_mae = df_result.groupby('이벤트있음')['오차'].mean()
print(f"         비이벤트일 MAE: {event_mae.get(False, float('nan')):.4f}")
print(f"         이벤트일 MAE  : {event_mae.get(True, float('nan')):.4f}")

# =============================================
# Step 7: 모델 저장
# =============================================
import joblib
model_dir = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, 'randomforest_final.pkl'))
joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, 'feature_columns.pkl'))
joblib.dump(encoders, os.path.join(model_dir, 'encoders.pkl'))
print(f"\n[Step 7] 모델 저장 완료: {model_dir}")