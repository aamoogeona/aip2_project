import sys
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rc('axes', unicode_minus=False)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'processing'))
from preprocess import load_data

# =============================================
# Step 1: estimated_daily_usage 로드
# =============================================
estimated_path = os.path.join(BASE_DIR, '..', 'data', 'processed', 'estimated_daily_usage.csv')
df_est = pd.read_csv(estimated_path, encoding='utf-8-sig')
df_est['날짜'] = pd.to_datetime(df_est['날짜'])
n_before = len(df_est)
df_est = df_est.dropna(subset=['시간대_y'])
n_dropped = n_before - len(df_est)
if n_dropped > 0:
    print(f"[경고] 시간대_y NaN 행 {n_dropped}개 제거됨")
df_est['시간대_y'] = df_est['시간대_y'].astype(int)

print(f"[Step 1] estimated 행 수: {len(df_est)}")
print(f"         시간대 범위: {df_est['시간대_y'].min()} ~ {df_est['시간대_y'].max()}")
print(f"         Estimated 범위: {df_est['Estimated'].min():.1f} ~ {df_est['Estimated'].max():.1f}")

# =============================================
# Step 2: df_melted에서 이벤트 컬럼 추출
# =============================================
df_melted, encoders, station_map = load_data(
    os.path.join(BASE_DIR, '..', 'data', 'raw', '혼잡도_정리본_2.xlsx')
)
df_melted['날짜'] = pd.to_datetime(df_melted['날짜'])

df_melted['역명_원본'] = encoders['역명'].inverse_transform(df_melted['역명'])
df_melted['요일_원본'] = encoders['요일'].inverse_transform(df_melted['요일'])
df_melted['상하구분_원본'] = encoders['상하구분'].inverse_transform(df_melted['상하구분'])

df_events = df_melted[['역명_원본', '요일_원본', '상하구분_원본', '시간대', '날짜',
                        'KBO', '공휴일', 'COEX', '가중치', 'COEX_가중치']].copy()

# =============================================
# Step 3: estimated를 베이스로 이벤트 컬럼 병합
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

n_total = len(df_data)
n_missing = df_data['KBO'].isna().sum()
print(f"\n[Step 2-3] 병합 결과")
print(f"           전체 행 수: {n_total}")
print(f"           이벤트 컬럼 결측: {n_missing} ({n_missing/n_total*100:.1f}%)")

n_before = len(df_data)
df_data = df_data.dropna(subset=['Estimated'])
n_dropped = n_before - len(df_data)
if n_dropped > 0:
    print(f"[경고] Estimated NaN 행 {n_dropped}개 제거됨")

for col in ['KBO', '공휴일', 'COEX', '가중치', 'COEX_가중치']:
    df_data[col] = df_data[col].fillna(0)
for col in ['KBO', '공휴일', 'COEX']:
    df_data[col] = df_data[col].astype(int)

for col, orig_col in [('역명', '역명_원본'), ('요일', '요일_원본'), ('상하구분', '상하구분_원본')]:
    df_data[orig_col] = df_data[orig_col].astype(str).str.strip()
    df_data[col] = encoders[col].transform(df_data[orig_col])

# estimated에서 KBO=1 vs KBO=0 실제값 비교
# (df_data는 train_catboost_estimated.py의 Step 3까지 실행한 결과를 사용)

# 종합운동장, 잠실 한정
target_stations = ['잠실', '종합운동장']
df_check = df_data[df_data['역명_원본'].isin(target_stations)].copy()

print("=== 역별 KBO 0/1 Estimated 평균 ===")
print(df_check.groupby(['역명_원본', 'KBO'])['Estimated'].agg(['mean', 'count']).to_string())

# 시간대별로도 보기 (KBO 경기 시간대인 18~22시 위주로 확인)
df_check['시간대_h'] = df_check['시간대'].apply(lambda x: f"{(x % 1440)//60:02d}:{x%60:02d}")
print("\n=== 종합운동장 시간대별 KBO 0/1 Estimated 평균 (요일별 분리) ===")
df_stadium = df_check[df_check['역명_원본'] == '종합운동장']
pivot = df_stadium.pivot_table(
    index=['요일_원본', '시간대_h'],
    columns='KBO',
    values='Estimated',
    aggfunc='mean'
)
pivot.columns = [f'KBO={c}' for c in pivot.columns]
pivot['차이'] = pivot.get('KBO=1', 0) - pivot.get('KBO=0', 0)
pivot['비율'] = pivot.get('KBO=1', 0) / pivot.get('KBO=0', 1)
print(pivot.to_string())



'''
# =============================================
# Step 4: 그래프 확인 (학습 전)
# =============================================
output_dir = os.path.join(BASE_DIR, '..', 'outputs')
os.makedirs(output_dir, exist_ok=True)

df_data['이벤트있음'] = ((df_data['KBO'] == 1) | (df_data['COEX'] == 1)).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Estimated y값 분포 확인 (학습 전 검증)', fontsize=14)

# 그래프 1: Estimated 전체 분포 vs 기존 혼잡도 평균
axes[0].hist(df_melted['혼잡도'], bins=50, alpha=0.5, label='기존 혼잡도(평균)', color='steelblue')
axes[0].hist(df_data['Estimated'], bins=50, alpha=0.5, label='Estimated', color='coral')
axes[0].set_title('전체 분포 비교\n(이벤트 신호가 생겼으면 오른쪽으로 퍼져야 함)')
axes[0].set_xlabel('혼잡도')
axes[0].set_ylabel('빈도')
axes[0].legend()

# 그래프 2: 이벤트일/비이벤트일 Estimated 분포
for flag, label, color in [(0, '비이벤트일', 'steelblue'), (1, '이벤트일', 'coral')]:
    vals = df_data[df_data['이벤트있음'] == flag]['Estimated']
    axes[1].hist(vals, bins=50, alpha=0.6, label=f'{label} (n={len(vals)})', color=color)
axes[1].set_title('이벤트일/비이벤트일 Estimated 분포\n(이벤트일이 더 높아야 신호 존재)')
axes[1].set_xlabel('Estimated 혼잡도')
axes[1].set_ylabel('빈도')
axes[1].legend()

# 그래프 3: 잠실/종합운동장 KBO일 vs 비KBO일 시간대별 Estimated 평균
target = df_data[df_data['역명_원본'].isin(['잠실', '종합운동장'])].copy()
target['시간대_h'] = target['시간대'].apply(lambda x: f"{(x % 1440)//60:02d}:{x%60:02d}")
kbo_mean = target.groupby(['역명_원본', 'KBO', '시간대_h'])['Estimated'].mean().reset_index()

for station, colors in [('종합운동장', ('steelblue', 'coral')), ('잠실', ('seagreen', 'orange'))]:
    sub = kbo_mean[kbo_mean['역명_원본'] == station]
    for kbo_val, color in [(0, colors[0]), (1, colors[1])]:
        row = sub[sub['KBO'] == kbo_val].sort_values('시간대_h')
        if len(row) > 0:
            label = f'{station} {"KBO일" if kbo_val == 1 else "비KBO일"}'
            axes[2].plot(row['시간대_h'], row['Estimated'], label=label, color=color,
                        linestyle='--' if kbo_val == 0 else '-', marker='o', markersize=3)

axes[2].set_title('KBO일 vs 비KBO일 시간대별 Estimated\n(KBO일 선이 위에 있어야 신호 존재)')
axes[2].set_xlabel('시간대')
axes[2].set_ylabel('Estimated 평균')
axes[2].legend(fontsize=8)
axes[2].tick_params(axis='x', rotation=60)

plt.tight_layout()
plt.show()
'''

# =============================================
# Step 5: 날짜 기준 train/test 분할
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

feature_cols = ['역명', '요일', '상하구분', '시간대', 'KBO', '공휴일', 'COEX', '가중치', 'COEX_가중치']

X_train = df_split_train[feature_cols]
y_train = df_split_train['Estimated']
X_test = df_split_test[feature_cols]
y_test = df_split_test['Estimated']

print(f"\n[Step 5] X_train: {X_train.shape}, X_test: {X_test.shape}")

'''
# =============================================
# Step 6: Ablation - 이벤트 변수 있음 vs 없음
# =============================================
print("\n" + "="*50)
print("Step 6: Ablation - 이벤트 변수 있음 vs 없음")
print("="*50)

cat_features_ablation = ['역명', '요일', '상하구분', 'KBO', '공휴일', 'COEX']
feature_cols_no_event = ['역명', '요일', '상하구분', '시간대']

# --- 이벤트 변수 있음 ---
X_train_with = df_split_train[feature_cols].copy()
for col in ['KBO', '공휴일', 'COEX']:
    X_train_with[col] = X_train_with[col].astype(int).astype(str)

X_test_with = X_test.copy()
for col in ['KBO', '공휴일', 'COEX']:
    X_test_with[col] = X_test_with[col].astype(int).astype(str)

model_with = CatBoostRegressor(random_state=42, verbose=False)
model_with.fit(X_train_with, df_split_train['Estimated'], cat_features=cat_features_ablation)

y_pred_with = model_with.predict(X_test_with)

df_with = X_test.copy()
df_with['실제값'] = y_test.values
df_with['예측값'] = y_pred_with
df_with['오차'] = abs(df_with['실제값'] - df_with['예측값'])
df_with['역명_원본'] = encoders['역명'].inverse_transform(df_with['역명'].astype(int))
df_with['이벤트있음'] = (df_with['KBO'].astype(int) == 1) | (df_with['COEX'].astype(int) == 1)

mae_with = mean_absolute_error(y_test, y_pred_with)
event_mae_with = df_with.groupby('이벤트있음')['오차'].mean()
feat_imp_with = pd.Series(model_with.feature_importances_, index=feature_cols)

df_kbo_with = df_with[df_with['역명_원본'].isin(['잠실', '종합운동장'])]
kbo_pred_with = df_kbo_with.groupby(['역명_원본', 'KBO'])['예측값'].mean()

print("\n[이벤트 변수 있음]")
print(f"  전체 MAE       : {mae_with:.4f}")
print(f"  비이벤트일 MAE : {event_mae_with.get(False, float('nan')):.4f}")
print(f"  이벤트일 MAE   : {event_mae_with.get(True, float('nan')):.4f}")
print(f"  피처 중요도    :")
for col in feature_cols:
    print(f"    {col:12s}: {feat_imp_with[col]:.4f}")
print(f"  KBO 0/1 예측값 :")
print(kbo_pred_with.to_string())

# --- 이벤트 변수 없음 ---
cat_features_no_event = ['역명', '요일', '상하구분']

X_train_without = df_split_train[feature_cols_no_event].copy()
X_test_without = X_test[feature_cols_no_event].copy()

model_without = CatBoostRegressor(random_state=42, verbose=False)
model_without.fit(X_train_without, df_split_train['Estimated'], cat_features=cat_features_no_event)

y_pred_without = model_without.predict(X_test_without)

df_without = X_test.copy()
df_without['실제값'] = y_test.values
df_without['예측값'] = y_pred_without
df_without['오차'] = abs(df_without['실제값'] - df_without['예측값'])
df_without['역명_원본'] = encoders['역명'].inverse_transform(df_without['역명'].astype(int))
df_without['이벤트있음'] = (df_without['KBO'].astype(int) == 1) | (df_without['COEX'].astype(int) == 1)

mae_without = mean_absolute_error(y_test, y_pred_without)
event_mae_without = df_without.groupby('이벤트있음')['오차'].mean()

df_kbo_without = df_without[df_without['역명_원본'].isin(['잠실', '종합운동장'])]
kbo_pred_without = df_kbo_without.groupby(['역명_원본', 'KBO'])['예측값'].mean()

print("\n[이벤트 변수 없음]")
print(f"  전체 MAE       : {mae_without:.4f}")
print(f"  비이벤트일 MAE : {event_mae_without.get(False, float('nan')):.4f}")
print(f"  이벤트일 MAE   : {event_mae_without.get(True, float('nan')):.4f}")
print(f"  피처 중요도    :")
for col in feature_cols_no_event:
    imp = pd.Series(model_without.feature_importances_, index=feature_cols_no_event)
    print(f"    {col:12s}: {imp[col]:.4f}")
print(f"  KBO 0/1 예측값 :")
print(kbo_pred_without.to_string())

# --- 요약 ---
print("\n[Ablation 요약]")
print(f"  {'':20s} | {'이벤트 있음':>12} | {'이벤트 없음':>12} | {'차이':>8}")
print(f"  {'-'*60}")
print(f"  {'전체 MAE':20s} | {mae_with:>12.4f} | {mae_without:>12.4f} | {mae_with - mae_without:>+8.4f}")
print(f"  {'이벤트일 MAE':20s} | {event_mae_with.get(True, float('nan')):>12.4f} | {event_mae_without.get(True, float('nan')):>12.4f} | {event_mae_with.get(True, 0) - event_mae_without.get(True, 0):>+8.4f}")
print(f"  {'비이벤트일 MAE':20s} | {event_mae_with.get(False, float('nan')):>12.4f} | {event_mae_without.get(False, float('nan')):>12.4f} | {event_mae_with.get(False, 0) - event_mae_without.get(False, 0):>+8.4f}")



# =============================================
# Step 5.5: train 데이터 다운샘플링 + 배수별 학습 비교
# =============================================
import numpy as np

cat_features = ['역명', '요일', '상하구분', 'KBO', '공휴일', 'COEX']  # 추가
ratios_to_try = [30, 50, 100, 200]
results = {}


# 이벤트일/비이벤트일 train 분리
train_event_mask = (df_split_train['KBO'] == 1) | (df_split_train['COEX'] == 1)
df_train_event = df_split_train[train_event_mask]
df_train_nonevent = df_split_train[~train_event_mask]

print(f"\n[Step 5.5] train 이벤트일: {len(df_train_event)}, 비이벤트일: {len(df_train_nonevent)}")

for ratio in ratios_to_try:
    n_sample = min(len(df_train_event) * ratio, len(df_train_nonevent))
    df_nonevent_sampled = df_train_nonevent.sample(n=n_sample, random_state=42)
    df_train_balanced = pd.concat([df_train_event, df_nonevent_sampled], ignore_index=True)

    X_tr = df_train_balanced[feature_cols]
    y_tr = df_train_balanced['Estimated']

    print(f"\n--- 비율 1:{ratio} (train 행 수: {len(df_train_balanced)}) ---")

    model = CatBoostRegressor(random_state=42, verbose=False)
    model.fit(X_tr, y_tr, cat_features=cat_features)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    df_r = X_test.copy()
    df_r['실제값'] = y_test.values
    df_r['예측값'] = y_pred
    df_r['오차'] = abs(df_r['실제값'] - df_r['예측값'])
    df_r['역명_원본'] = encoders['역명'].inverse_transform(df_r['역명'].astype(int))
    df_r['이벤트있음'] = (df_r['KBO'] == 1) | (df_r['COEX'] == 1)

    feat_imp = pd.Series(model.feature_importances_, index=feature_cols)
    event_mae = df_r.groupby('이벤트있음')['오차'].mean()

    # KBO=1 vs KBO=0 예측값 차이
    df_kbo = df_r[df_r['역명_원본'].isin(['잠실', '종합운동장'])]
    kbo_pred_diff = df_kbo.groupby(['역명_원본', 'KBO'])['예측값'].mean()

    print(f"전체 MAE: {mae:.4f}")
    print(f"비이벤트일 MAE: {event_mae.get(False, 0):.4f}")
    print(f"이벤트일 MAE: {event_mae.get(True, 0):.4f}")
    print(f"KBO importance: {feat_imp.get('KBO', 0):.4f}")
    print(f"COEX importance: {feat_imp.get('COEX', 0):.4f}")
    print(f"가중치 importance: {feat_imp.get('가중치', 0):.4f}")
    print(f"COEX_가중치 importance: {feat_imp.get('COEX_가중치', 0):.4f}")
    print(f"KBO 0/1 예측값:\n{kbo_pred_diff.to_string()}")

    results[ratio] = {
        'mae': mae,
        'event_mae': event_mae.get(True, 0),
        'nonevent_mae': event_mae.get(False, 0),
        'kbo_imp': feat_imp.get('KBO', 0),
        'coex_imp': feat_imp.get('COEX', 0),
    }

# 다운샘플링 없음 (원본) baseline
print(f"\n--- 다운샘플링 없음 (원본, 약 1:228) ---")
X_tr = df_split_train[feature_cols].copy()
y_tr = df_split_train['Estimated']
for col in ['KBO', '공휴일', 'COEX']:
    X_tr[col] = X_tr[col].astype(int).astype(str)

model = CatBoostRegressor(random_state=42, verbose=False)
model.fit(X_tr, y_tr, cat_features=cat_features)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

df_r = X_test.copy()
df_r['실제값'] = y_test.values
df_r['예측값'] = y_pred
df_r['오차'] = abs(df_r['실제값'] - df_r['예측값'])
df_r['역명_원본'] = encoders['역명'].inverse_transform(df_r['역명'].astype(int))
df_r['이벤트있음'] = (df_r['KBO'].astype(int) == 1) | (df_r['COEX'].astype(int) == 1)

feat_imp = pd.Series(model.feature_importances_, index=feature_cols)
event_mae = df_r.groupby('이벤트있음')['오차'].mean()
df_kbo = df_r[df_r['역명_원본'].isin(['잠실', '종합운동장'])]
kbo_pred_diff = df_kbo.groupby(['역명_원본', 'KBO'])['예측값'].mean()

print(f"전체 MAE: {mae:.4f}")
print(f"비이벤트일 MAE: {event_mae.get(False, 0):.4f}")
print(f"이벤트일 MAE: {event_mae.get(True, 0):.4f}")
print(f"KBO importance: {feat_imp.get('KBO', 0):.4f}")
print(f"COEX importance: {feat_imp.get('COEX', 0):.4f}")
print(f"KBO 0/1 예측값:\n{kbo_pred_diff.to_string()}")

print("\n=== 배수별 요약 ===")
for ratio, r in results.items():
    print(f"1:{ratio:>2} | 전체MAE={r['mae']:.3f} | 이벤트MAE={r['event_mae']:.3f} | "
        f"비이벤트MAE={r['nonevent_mae']:.3f} | KBO_imp={r['kbo_imp']:.4f} | COEX_imp={r['coex_imp']:.4f}")
'''

'''
# =============================================
# Step 7: Ablation - 가중치 제거 (0/1만 사용)
# =============================================
print("\n" + "="*50)
print("Step 7: Ablation - 가중치 제거 (0/1만)")
print("="*50)

feature_cols_no_weight = ['역명', '요일', '상하구분', '시간대', 'KBO', '공휴일', 'COEX']
cat_features_no_weight = ['역명', '요일', '상하구분', 'KBO', '공휴일', 'COEX']

X_train_nw = df_split_train[feature_cols_no_weight].copy()
X_test_nw = X_test[feature_cols_no_weight].copy()
for col in ['KBO', '공휴일', 'COEX']:
    X_train_nw[col] = X_train_nw[col].astype(int).astype(str)
    X_test_nw[col] = X_test_nw[col].astype(int).astype(str)

model_nw = CatBoostRegressor(random_state=42, verbose=False)
model_nw.fit(X_train_nw, df_split_train['Estimated'], cat_features=cat_features_no_weight)
y_pred_nw = model_nw.predict(X_test_nw)

df_nw = X_test.copy()
df_nw['실제값'] = y_test.values
df_nw['예측값'] = y_pred_nw
df_nw['오차'] = abs(df_nw['실제값'] - df_nw['예측값'])
df_nw['역명_원본'] = encoders['역명'].inverse_transform(df_nw['역명'].astype(int))
df_nw['이벤트있음'] = (df_nw['KBO'].astype(int) == 1) | (df_nw['COEX'].astype(int) == 1)

mae_nw = mean_absolute_error(y_test, y_pred_nw)
event_mae_nw = df_nw.groupby('이벤트있음')['오차'].mean()
feat_imp_nw = pd.Series(model_nw.feature_importances_, index=feature_cols_no_weight)
df_kbo_nw = df_nw[df_nw['역명_원본'].isin(['잠실', '종합운동장'])]
kbo_pred_nw = df_kbo_nw.groupby(['역명_원본', 'KBO'])['예측값'].mean()

print(f"  전체 MAE       : {mae_nw:.4f}")
print(f"  비이벤트일 MAE : {event_mae_nw.get(False, float('nan')):.4f}")
print(f"  이벤트일 MAE   : {event_mae_nw.get(True, float('nan')):.4f}")
print(f"  피처 중요도    :")
for col in feature_cols_no_weight:
    print(f"    {col:12s}: {feat_imp_nw[col]:.4f}")
print(f"  KBO 0/1 예측값 :")
print(kbo_pred_nw.to_string())


# =============================================
# Step 8: Ablation - 0/1 제거 (가중치만 사용)
# =============================================
print("\n" + "="*50)
print("Step 8: Ablation - 0/1 제거 (가중치만)")
print("="*50)

feature_cols_wo = ['역명', '요일', '상하구분', '시간대', '가중치', 'COEX_가중치']
cat_features_wo = ['역명', '요일', '상하구분']

X_train_wo = df_split_train[feature_cols_wo].copy()
X_test_wo = X_test[feature_cols_wo].copy()

model_wo = CatBoostRegressor(random_state=42, verbose=False)
model_wo.fit(X_train_wo, df_split_train['Estimated'], cat_features=cat_features_wo)
y_pred_wo = model_wo.predict(X_test_wo)

df_wo = X_test.copy()
df_wo['실제값'] = y_test.values
df_wo['예측값'] = y_pred_wo
df_wo['오차'] = abs(df_wo['실제값'] - df_wo['예측값'])
df_wo['역명_원본'] = encoders['역명'].inverse_transform(df_wo['역명'].astype(int))
df_wo['이벤트있음'] = (df_wo['KBO'].astype(int) == 1) | (df_wo['COEX'].astype(int) == 1)

mae_wo = mean_absolute_error(y_test, y_pred_wo)
event_mae_wo = df_wo.groupby('이벤트있음')['오차'].mean()
feat_imp_wo = pd.Series(model_wo.feature_importances_, index=feature_cols_wo)
df_kbo_wo = df_wo[df_wo['역명_원본'].isin(['잠실', '종합운동장'])]
kbo_pred_wo = df_kbo_wo.groupby(['역명_원본', 'KBO'])['예측값'].mean()

print(f"  전체 MAE       : {mae_wo:.4f}")
print(f"  비이벤트일 MAE : {event_mae_wo.get(False, float('nan')):.4f}")
print(f"  이벤트일 MAE   : {event_mae_wo.get(True, float('nan')):.4f}")
print(f"  피처 중요도    :")
for col in feature_cols_wo:
    print(f"    {col:12s}: {feat_imp_wo[col]:.4f}")
print(f"  KBO 0/1 예측값 :")
print(kbo_pred_wo.to_string())

# =============================================
# Step 9: 다른 모델 비교 (이벤트 변수 있음/없음)
# =============================================
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

print("\n" + "="*50)
print("Step 9: 다른 모델 비교")
print("="*50)

# 원핫 인코딩 (선형/RF용)
X_train_oh = pd.get_dummies(df_split_train[feature_cols],
    columns=['역명', '요일', '상하구분'], dtype=int)
X_test_oh = pd.get_dummies(X_test,
    columns=['역명', '요일', '상하구분'], dtype=int)
X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

X_train_oh_noevt = X_train_oh.drop(columns=['KBO', '공휴일', 'COEX', '가중치', 'COEX_가중치'])
X_test_oh_noevt = X_test_oh.drop(columns=['KBO', '공휴일', 'COEX', '가중치', 'COEX_가중치'])

for model_name, m_with, m_without in [
    ('Ridge', Ridge(alpha=1.0, random_state=42), Ridge(alpha=1.0, random_state=42)),
    ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
]:
    print(f"\n[{model_name}]")
    # 이벤트 있음
    m_with.fit(X_train_oh, df_split_train['Estimated'])
    p_with = m_with.predict(X_test_oh)
    # 이벤트 없음
    m_without.fit(X_train_oh_noevt, df_split_train['Estimated'])
    p_without = m_without.predict(X_test_oh_noevt)

    mae_with = mean_absolute_error(y_test, p_with)
    mae_without = mean_absolute_error(y_test, p_without)

    df_ev = X_test.copy()
    df_ev['이벤트있음'] = (df_ev['KBO'] == 1) | (df_ev['COEX'] == 1)
    df_ev['오차_with'] = abs(y_test.values - p_with)
    df_ev['오차_without'] = abs(y_test.values - p_without)
    em_with = df_ev.groupby('이벤트있음')['오차_with'].mean()
    em_without = df_ev.groupby('이벤트있음')['오차_without'].mean()

    print(f"  전체 MAE      : with={mae_with:.4f} | without={mae_without:.4f} | 차이={mae_with - mae_without:+.4f}")
    print(f"  이벤트일 MAE  : with={em_with.get(True, 0):.4f} | without={em_without.get(True, 0):.4f}")
    print(f"  비이벤트일 MAE: with={em_with.get(False, 0):.4f} | without={em_without.get(False, 0):.4f}")

    if hasattr(m_with, 'coef_'):
        coefs = pd.Series(m_with.coef_, index=X_train_oh.columns)
        print(f"  이벤트 변수 계수:")
        for k in ['KBO', '공휴일', 'COEX', '가중치', 'COEX_가중치']:
            print(f"    {k:12s}: {coefs.get(k, 0):.4f}")
'''

# =============================================
# Step 10: 동일 조건 비교 - CatBoost vs RandomForest (원핫, 가중치 제외)
# =============================================
from sklearn.ensemble import RandomForestRegressor

print("\n" + "="*50)
print("Step 10: CatBoost vs RandomForest 동일 조건 비교")
print("="*50)

feature_cols_final = ['역명', '요일', '상하구분', '시간대', 'KBO', '공휴일', 'COEX']

# 원핫 인코딩 (역명, 요일, 상하구분만)
X_train_final = pd.get_dummies(
    df_split_train[feature_cols_final],
    columns=['역명', '요일', '상하구분'], dtype=int
)
X_test_final = pd.get_dummies(
    X_test[feature_cols_final],
    columns=['역명', '요일', '상하구분'], dtype=int
)
X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

# 이벤트 변수 제거 버전 (ablation용)
X_train_noevt = X_train_final.drop(columns=['KBO', '공휴일', 'COEX'])
X_test_noevt = X_test_final.drop(columns=['KBO', '공휴일', 'COEX'])

for model_name, m_with, m_without in [
    ('CatBoost', CatBoostRegressor(random_state=42, verbose=False),
        CatBoostRegressor(random_state=42, verbose=False)),
    ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
]:
    print(f"\n[{model_name}]")
    # 이벤트 있음
    m_with.fit(X_train_final, df_split_train['Estimated'])
    p_with = m_with.predict(X_test_final)
    # 이벤트 없음
    m_without.fit(X_train_noevt, df_split_train['Estimated'])
    p_without = m_without.predict(X_test_noevt)

    mae_with = mean_absolute_error(y_test, p_with)
    mae_without = mean_absolute_error(y_test, p_without)

    df_ev = X_test.copy()
    df_ev['역명_원본'] = encoders['역명'].inverse_transform(df_ev['역명'].astype(int))
    df_ev['이벤트있음'] = (df_ev['KBO'] == 1) | (df_ev['COEX'] == 1)
    df_ev['오차_with'] = abs(y_test.values - p_with)
    df_ev['오차_without'] = abs(y_test.values - p_without)
    df_ev['예측값_with'] = p_with
    em_with = df_ev.groupby('이벤트있음')['오차_with'].mean()
    em_without = df_ev.groupby('이벤트있음')['오차_without'].mean()

    df_kbo = df_ev[df_ev['역명_원본'].isin(['잠실', '종합운동장'])]
    kbo_pred = df_kbo.groupby(['역명_원본', 'KBO'])['예측값_with'].mean()

    print(f"  전체 MAE      : with={mae_with:.4f} | without={mae_without:.4f} | 차이={mae_with - mae_without:+.4f}")
    print(f"  이벤트일 MAE  : with={em_with.get(True, 0):.4f} | without={em_without.get(True, 0):.4f}")
    print(f"  비이벤트일 MAE: with={em_with.get(False, 0):.4f} | without={em_without.get(False, 0):.4f}")
    print(f"  KBO 0/1 예측값 (이벤트 있음):")
    print(kbo_pred.to_string())

'''
# =============================================
# Step 10: 평가 (완료 기준 3가지)
# =============================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n[Step 7] 완료 기준 평가")
print(f"  [기준 1] 전체 MAE: {mae:.4f}  (목표: <= 3.25)")

df_result = X_test.copy()
df_result['실제값'] = y_test.values
df_result['예측값'] = y_pred
df_result['오차'] = abs(df_result['실제값'] - df_result['예측값'])
df_result['역명_원본'] = encoders['역명'].inverse_transform(df_result['역명'].astype(int))

# 기준 2: 피처 중요도
feat_imp = pd.Series(model.feature_importances_, index=feature_cols)
print(f"\n  [기준 2] 피처 중요도  (목표: KBO/COEX >> 0.001)")
print(feat_imp.sort_values(ascending=False).to_string())

# 기준 3: KBO=1 vs KBO=0 예측값 비교
target_stations = ['잠실', '종합운동장']
df_kbo_check = df_result[df_result['역명_원본'].isin(target_stations)].copy()
print(f"\n  [기준 3] KBO 0/1별 실제값 vs 예측값  (목표: KBO=1 예측값 > KBO=0 예측값)")
print(df_kbo_check.groupby(['역명_원본', 'KBO'])[['실제값', '예측값']].mean().to_string())

# 참고: 이벤트일/비이벤트일 MAE
df_result['이벤트있음'] = (df_result['KBO'] == 1) | (df_result['COEX'] == 1)
print("\n  [참고] 이벤트일/비이벤트일 MAE")
print(df_result.groupby('이벤트있음')['오차'].agg(['mean', 'count']).to_string())
'''
