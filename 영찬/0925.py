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

# 결측치 처리 (molten_volume)
'''
# 보고 싶은 컬럼만 선택
train_df = train_df[~(train_df['tryshot_signal']=="D")]
train_df['tryshot_signal'].value_counts()
df_selected = train_df[['time','date','count','molten_volume','mold_code','sleeve_temperature','passorfail']].copy()
df_selected.dropna(subset=['molten_volume'], inplace=True)
df_selected = df_selected[df_selected['molten_volume']<2000]

df_selected_8412 = df_selected[df_selected['mold_code']==8412].reset_index(drop=True)

df_selected

# 예시: 데이터 불러오기
# df = pd.read_csv("train.csv")  

# mold_code별로 그래프 그리기
mold_codes = df_selected['mold_code'].unique()

plt.figure(figsize=(15, 10))

for i, mold in enumerate(mold_codes, 1):
    plt.subplot(len(mold_codes), 1, i)
    mold_df = df_selected [df_selected['mold_code'] == mold].head(300)
    sns.scatterplot(data=mold_df, x='count', y='molten_volume', hue='passorfail', palette='Set1', alpha=0.6)
    plt.title(f'Mold Code: {mold}')
    plt.xlabel('Count')
    plt.ylabel('Molten Volume')

plt.tight_layout()
plt.show()
'''
train_df.info()

#KNN 실습
train_df.loc[train_df['mold_code'] == 8573, 'molten_volume'] = -1

from sklearn.impute import KNNImputer
import pandas as pd

# 결측치 위치 저장
missing_mask = train_df['molten_volume'].isnull()

# KNN Imputer 초기화 (k=3)
knn_imputer = KNNImputer(n_neighbors=3)

# molten_volume을 DataFrame 형태로 변환
molten_volume_df = train_df[['molten_volume']]

# KNN으로 전체 변환
imputed_values = knn_imputer.fit_transform(molten_volume_df).flatten()

# 결측치였던 부분만 대체
train_df.loc[missing_mask, 'molten_volume'] = imputed_values[missing_mask]

# 그래프 그리기
mold_codes = train_df['mold_code'].unique()

plt.figure(figsize=(15, 10))

for i, mold in enumerate(mold_codes, 1):
    plt.subplot(len(mold_codes), 1, i)
    mold_df = train_df [train_df['mold_code'] == mold].head(300)
    sns.scatterplot(data=mold_df, x='count', y='molten_volume', hue='passorfail', palette='Set1', alpha=0.6)
    plt.title(f'Mold Code: {mold}')
    plt.xlabel('Count')
    plt.ylabel('Molten Volume')

plt.tight_layout()
plt.show()
