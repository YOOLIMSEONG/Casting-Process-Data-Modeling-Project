import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/raw/train.csv")
df.info()
df["mold_name"].unique()

# 확인할 변수들
cols = [
    "facility_operation_cycleTime",
    "production_cycletime",
    "low_section_speed",
    "high_section_speed",
    "molten_volume"
]

# facility_operation_cycleTime
# 설비 작동 사이클 시간
# 결측치 0개
# 유니크 개수 195개
# iqr기반 이상치 5401개
target = df["facility_operation_cycleTime"]
target.describe()
target.isna().sum()
target.nunique()
target.value_counts().head(20)

plt.figure(figsize=(10,5))
plt.hist(target, bins=30, edgecolor='k')
plt.title(f"Distribution of {target.name}")
plt.xlabel(target.name)
plt.ylabel("Frequency")
plt.show()

q1 = target.quantile(0.25)
q3 = target.quantile(0.75)
iqr = q3 - q1
outlier_mask = (target < (q1 - 1.5 * iqr)) | (target > (q3 + 1.5 * iqr))
outliers = target.loc[outlier_mask]
outliers.count()

# production_cycletime
# 제품 생산 사이클 시간
# 결측치 0개
# 유니크 개수 201개
# iqr 기반 이상치 5439개
target = df["production_cycletime"]
target.describe()
target.isna().sum()
target.nunique()
target.value_counts().head(20)

plt.figure(figsize=(10,5))
plt.hist(target, bins=30, edgecolor='k')
plt.title(f"Distribution of {target.name}")
plt.xlabel(target.name)
plt.ylabel("Frequency")
plt.show()

q1 = target.quantile(0.25)
q3 = target.quantile(0.75)
iqr = q3 - q1
outlier_mask = (target < (q1 - 1.5 * iqr)) | (target > (q3 + 1.5 * iqr))
outliers = target.loc[outlier_mask]
outliers.count()

# low_section_speed
# 저속 구간 속도
# 결측치 1개
# 유니크 개수 117개
# iqr 기반 이상치 22686개
target = df["low_section_speed"]
target.describe()
target.isna().sum()
target.nunique()
target.value_counts().head(50)

plt.figure(figsize=(10,5))
plt.hist(target, bins=30, edgecolor='k')
plt.title(f"Distribution of {target.name}")
plt.xlabel(target.name)
plt.ylabel("Frequency")
plt.show()

q1 = target.quantile(0.25)
q3 = target.quantile(0.75)
iqr = q3 - q1
outlier_mask = (target < (q1 - 1.5 * iqr)) | (target > (q3 + 1.5 * iqr))
outliers = target.loc[outlier_mask]
outliers.count()

# high_section_speed
# 고속 구간 속도
# 결측치 1개
# 유니크 개수 218개
# iqr 기반 이상치 28774개
target = df["high_section_speed"]
target.describe()
target.isna().sum()
target.nunique()
target.value_counts().head(20)

plt.figure(figsize=(10,5))
plt.hist(target[target > 150], bins=30, edgecolor='k')
plt.title(f"Distribution of {target.name}")
plt.xlabel(target.name)
plt.ylabel("Frequency")
plt.show()

q1 = target.quantile(0.25)
q3 = target.quantile(0.75)
iqr = q3 - q1
outlier_mask = (target < (q1 - 1.5 * iqr)) | (target > (q3 + 1.5 * iqr))
outliers = target.loc[outlier_mask]
outliers.count()

# molten_volume
# 용탕량 : 용해된 금속이 주입되는 양
# 결측치 34992개
# 유니크 개수 121개
# iqr 기반 이상치 1571개
# 대부분의 이상치는 2767에 분포
target = df["molten_volume"]
target.describe()
target.isna().sum()
target.nunique()
target.value_counts().head(20)

plt.figure(figsize=(10,5))
plt.hist(target, bins=30, edgecolor='k')
plt.title(f"Distribution of {target.name}")
plt.xlabel(target.name)
plt.ylabel("Frequency")
plt.show()

df.loc[df["molten_volume"] > 2500, "passorfail"].value_counts()

q1 = target.quantile(0.25)
q3 = target.quantile(0.75)
iqr = q3 - q1
outlier_mask = (target < (q1 - 1.5 * iqr)) | (target > (q3 + 1.5 * iqr))
outliers = target.loc[outlier_mask]
outliers.count()

df.loc[df["molten_volume"].isna(), "passorfail"].value_counts()

plt.figure(figsize=(10,5))
plt.hist(df.loc[df["molten_volume"].isna(), "passorfail"], bins=30, edgecolor='k')
plt.title(f"Distribution of {target.name}")
plt.xlabel(target.name)
plt.ylabel("Frequency")
plt.show()