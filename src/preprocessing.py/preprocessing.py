import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv("../data/raw/train.csv")
test_df = pd.read_csv("../data/raw/test.csv")

train_df.info()
train_df.head()
train_df.isna().sum()

# 대부분이 결측치인 행 제거
train_df.drop(19327, inplace=True)

# 분석에서 필요없는 컬럼 제거
train_df.drop(columns=["id", "line", "name", "mold_name", "emergency_stop", "registration_time"], inplace=True)

'''
결측치 처리 (molten_temp)
동일코드 앞 생산 온도, 동일 코드 뒤 생산 온도 평균
'''

# 🔹 원본 molten_temp를 새로운 열로 복사
train_df['molten_temp_filled'] = train_df['molten_temp']

# 🔹 금형별 시간 순 정렬 후 선형 보간
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.interpolate(method='linear'))
)

# 🔹 여전히 남아있는 결측치(맨 앞/뒤)는 그룹별 중앙값으로 채우기
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.fillna(x.median()))
)
train_df[['molten_temp', 'molten_temp_filled']]
train_df['molten_temp'].isna().sum()
train_df['molten_temp_filled'].isna().sum()
train_df.drop(columns=["molten_temp"], inplace=True)

# 결측치 처리
# molten_volume
train_df.groupby("mold_code")["molten_volume"].count()
train_df.loc[train_df["mold_code"] == 8412, "molten_volume"].hist()
