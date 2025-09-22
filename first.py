import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 불러오기
# train에는 없고 test에는 있는 값들이 있음
df = pd.read_csv("train.csv")

df.info()
# time / date/ count/ working/ emrgency_stop/molten_tmp / heating_furnace

################################################################################
# EDA
####################################################################################

# time 
# 년월일
# 결측치 확인
df['time'].isnull().sum()

# time 컬럼을 object > datetime 형식으로 변환
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d', errors='coerce')

# 변환 결과 확인
print(df['time'].dtypes)
print(df['time'].head())
df['time'].sort_values()

# 날짜 별 생산량 차이
daily_counts = df[['id','time']].groupby(['time']).count()
len(daily_counts)
# 시각화
plt.figure(figsize=(12,6))
plt.plot(daily_counts.index, daily_counts['id'], marker='o', linestyle='-')

plt.title("일자별 데이터 개수", fontsize=14)
plt.xlabel("날짜", fontsize=12)
plt.ylabel("개수", fontsize=12)
plt.grid(True)
plt.show()

# 공장 휴무일 확인
start = df['time'].min()
end = df['time'].max()

# 두 날짜 차이
diff = (end - start).days + 1   # 양 끝 날짜 모두 포함하려면 +1
print(diff)  # 70 # 66일 동안 돌아감 # 4일 휴무

####################################################################################
# date ( time과 다르게 시간 분 초)
####################################################################################

# object > time 데이터 타입 변경
df['date'].dtypes
df['date'] = pd.to_datetime(df['date'], format='%H:%M:%S').dt.time

# 결측치 확인
df['time'].isnull().sum()

# 시간별 생산량 확인
# 2️⃣ 시간만 뽑아서 hour 컬럼 생성
df['hour'] = pd.to_datetime(df['date'].astype(str)).dt.hour

# 3️⃣ 시간별 생산량 집계
hourly_counts = df.groupby('hour').size()

# 4️⃣ 시각화
plt.figure(figsize=(12,6))
hourly_counts.plot(kind='bar')
plt.title("시간대별 생산량")
plt.xlabel("시간 (시)")
plt.ylabel("생산량")
plt.xticks(rotation=0)
plt.show()

# 주말 또는 공휴일이랑 평일이랑 비교해보면 좋을 듯

####################################################################################
# count
####################################################################################

# 결측치 확인
df['count'].isnull().sum()

# 
df[['count','time']].groupby('time').max()

####################################################################################
# working
####################################################################################
df.info()
# 결측치 확인
df['working'].isnull().sum() # 1개 > 드랍

# 상태별 건수 집계 # 정지 38개
status_counts = df['working'].value_counts()

# 막대그래프 시각화
plt.figure(figsize=(6,4))
status_counts.plot(kind='bar', color=['green','red'])
plt.title("설비 상태별 생산량")
plt.xlabel("상태")
plt.ylabel("건수")
plt.xticks(rotation=0)
plt.show()

# 정지 총 38개 > 2개는 정상
df[df['working']=='정지']['passorfail'].value_counts()

####################################################################################
# emrgency_stop
####################################################################################

# 결측치 확인
df['emergency_stop'].isnull().sum() # 1개 드랍

# 분포 확인
df['emergency_stop'].value_counts() # ON 값 1개만 있어서 없애도 되는 열인 듯?

# 결측치 행 확인
df[df['emergency_stop'].isna()]

####################################################################################
#molten_temp
####################################################################################

# 결측치 2261개
df['molten_temp'].isnull().sum()

# 히스토그렘
plt.figure(figsize=(10,6))
df['molten_temp'].hist(bins=100)  # bin 수를 50으로 늘림
plt.xlabel("용탕 온도")
plt.ylabel("건수")
plt.title("용탕 온도 분포")
plt.show()

df['molten_temp'].describe()

# 이상치로 의심되는 데이터 개수
df[df['molten_temp'] < 600] # 434 개
df[df['molten_temp'] < 100]['passorfail'].value_counts()

# 의심되는 애들 제거하고 새로 시각화
df = df[df['molten_temp'] >= 600]

# 히스토그렘
plt.figure(figsize=(10,6))
df['molten_temp'].hist(bins=100)  # bin 수를 50으로 늘림
plt.xlabel("용탕 온도")
plt.ylabel("건수")
plt.title("용탕 온도 분포")
plt.show()

# 불량 양품 비교
plt.figure(figsize=(10,6))
sns.boxplot(x='passorfail', y='molten_temp', data=df)
plt.xticks([0,1], ['양품', '불량'])
plt.xlabel("품질")
plt.ylabel("용탕 온도")
plt.title("불량/양품에 따른 용탕 온도 분포")
plt.show()

# 구간 별 불량률 계산해야할듯

####################################################################################
# heating_furnace
####################################################################################

# 결측치 38208개 
df['heating_furnace'].isnull().sum()

# 값 확인
df['heating_furnace'].value_counts()

# 결측치에 'nan' 문자열로 채우기
df['heating_furnace'] = df['heating_furnace'].fillna('nan')

# 상태별 개수 집계
furnace_counts = df['heating_furnace'].value_counts()

# 막대그래프 시각화
plt.figure(figsize=(6,4))
furnace_counts.plot(kind='bar', color=['orange','blue','gray'])
plt.title("가열로 구분별 샘플 수 (결측치 'nan' 포함)")
plt.xlabel("가열로")
plt.ylabel("건수")
plt.xticks(rotation=0)
plt.show()
df
df['mold_name'].value_counts()