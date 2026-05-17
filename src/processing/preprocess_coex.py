import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
coex_path = os.path.join(BASE_DIR, '..', '..', 'data', 'raw', 'coex_event_data.xlsx')
df_coex_raw = pd.read_excel(coex_path)

def load_coex_data():
    df_coex = df_coex_raw.rename(columns={'날짜|역명': '날짜', '삼성역': 'COEX_가중치'})
    df_coex['날짜'] = pd.to_datetime(df_coex['날짜'])

    return df_coex