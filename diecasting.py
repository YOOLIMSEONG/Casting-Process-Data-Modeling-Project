# %%
import numpy as np
import pandas as pd
import seaborn as sns
import os
# %%
print(os.getcwd())
# %%
os.chdir("..\\open")
os.chdir(".\\diecasting_dat")
# %%
df = pd.read_csv('train.csv')
# %%
df.info() 
# %%
df.columns
# %%
df.shape
# %%
df.isnull().sum()
# %%
df_cont = df[df['tryshot_signal'] != 'D']
df_test = df[df['tryshot_signal'] == 'D']
# %%
sns.scatterplot(data=df_cont, x='molten_temp', y='sleeve_temperature', hue='passorfail')
# %%
df['diff1'] = df['upper_mold_temp1'] - df['lower_mold_temp1']
df['diff2'] = df['upper_mold_temp2'] - df['lower_mold_temp2']
# %%
sns.histplot(data=df,x='Coolant_temperature')
# %%
# %%
sns.histplot(data=df, x='physical_strength', hue='passorfail')
# %%
df['physical_strength'].min()
# %%
df.info()
# %%
df.info()
# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# %%
# 시간 컬럼 변환
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# 불량 데이터 필터링
df['is_fail'] = df['passorfail'] ==1 # 대소문자 구분 제거

# 일별 불량 건수 계산
daily_defects = df.groupby(df['datetime'].dt.date)['is_fail'].sum()

# 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(15, 5))
daily_defects.plot(kind='line', marker='o', color='crimson')
plt.title('📉 일별 불량 발생 추이 (passorfail 기준)')
plt.xlabel('날짜')
plt.ylabel('불량 건수')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
passfail_counts = df.groupby(['mold_code', 'passorfail']).size().unstack(fill_value=0)

# ✅ 시각화: stacked bar chart
passfail_counts.sort_index(inplace=True)  # mold_code 순서 정렬
passfail_counts.plot(kind='bar', stacked=True, figsize=(15, 6), colormap='Set2')

# ✅ 시각화 옵션
plt.title('Mold Code 별 Pass/Fail 분포 (Stacked Bar)')
plt.xlabel('Mold Code')
plt.ylabel('건수')
plt.xticks(rotation=90)
plt.legend(title='검사 결과')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
# %%
passfail_counts
# %%
df[df['mold_code'] == 8917]['heating_furnace'].value_counts()
# %%
sns.kdeplot(data=df[df['mold_code'] == 8573], x='molten_temp')
# %%
subset_cols = ['count', 'time', 'mold_code']
# %%
duplicates = df[df.duplicated(subset=subset_cols, keep=False)]
# %%
df.groupby(subset_cols).size().reset_index(name='duplicates').query('duplicates > 1')
# %%
df_8573 = df[df['mold_code'] == 8573]
# %%
df_8573.to_csv('df_8573.csv', index=False)
# %%
df['mold_code'].unique()
# %%
df_8722 = df[df['mold_code']==8722]
df_8412 = df[df['mold_code']==8412]
df_8917 = df[df['mold_code']==8917]
df_8600 = df[df['mold_code']==8600]
# %%
df_8722.to_csv('df_8722.csv', index=False)
df_8412.to_csv('df_8412.csv', index=False)
df_8917.to_csv('df_8917.csv', index=False)
df_8600.to_csv('df_8600.csv', index=False)

# %%
sns.kdeplot(data = df_8722, x = 'molten_temp', hue='passorfail')
# %%
df_8722['is_fail']
# %%
daily_defects_8722 = df_8722.groupby(df_8722['datetime'].dt.date)['is_fail'].sum()
# %%
df_cont = df[df['tryshot_signal'] != 'D']
# %%
daily_defects_by_mold = (
    df_cont.groupby([df_cont['datetime'].dt.floor('D'), 'mold_code'])['is_fail']
      .sum()
      .unstack(fill_value=0)   # 열로 mold_code 펼치기
)

print(daily_defects_by_mold.head())
# %%
plt.figure(figsize=(15, 6))
for col in daily_defects_by_mold.columns:
    plt.plot(daily_defects_by_mold.index,
             daily_defects_by_mold[col],
             marker='o',
             label=f'Mold {col}')

plt.title('📉 mold_code별 일별 불량 발생 추이')
plt.xlabel('날짜')
plt.ylabel('불량 건수')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend(title='Mold Code')
plt.tight_layout()
plt.show()
# %%
hourly_defects = (
    df_cont.groupby(df_cont['datetime'].dt.hour)['is_fail']
      .sum()
)

print(hourly_defects)
# %%
hourly_by_mold = (
    df_cont.groupby([df_cont['datetime'].dt.hour, 'mold_code'])['is_fail']
      .sum()
      .unstack(fill_value=0)
)

hourly_by_mold.plot(figsize=(15,6), marker='o')
plt.title('⏰ mold_code별 시간대별 불량 추이')
plt.xlabel('시간 (시)')
plt.ylabel('불량 건수')
plt.xticks(range(0,24))
plt.grid(True)
plt.legend(title='Mold Code')
plt.tight_layout()
plt.show()
# %%
hourly_temp_by_mold = (
    df.groupby([df['datetime'].dt.hour, 'mold_code'])['molten_temp']
      .mean()
      .unstack()
)
# %%
hourly_temp_by_mold.plot(figsize=(15,6), marker='o')
plt.title('⏰ mold_code별 시간대 평균 Molten Temp')
plt.xlabel('시간 (시)')
plt.ylabel('온도 (℃)')
plt.xticks(range(0,24))
plt.grid(True)
plt.legend(title='Mold Code')
plt.tight_layout()
plt.show()
# %%
df['tryshot_signal'].unique()
# %%
df_try = df[df['tryshot_signal'] == 'D']
df_cont = df[df['tryshot_signal'] != 'D']
# %%
sns.histplot(data = df_cont[df_cont['passorfail']==1], x = 'physical_strength')
# %%
df_cont
# %%

df_cont['datetime'] = pd.to_datetime(df_cont['datetime'])
# %%
df_clip = df_cont.copy()
df_clip['bt_ma50'] = df_clip['biscuit_thickness'].rolling(50, min_periods=1).mean()
# %%
plt.figure(figsize=(14,5))
plt.plot(df_clip['datetime'], df_clip['biscuit_thickness'], linestyle='-', marker='.', alpha=0.35, label='Raw')
plt.plot(df_clip['datetime'], df_clip['bt_ma50'], linewidth=2, label='MA(50)')
plt.title('시간에 따른 Biscuit Thickness 추이')
plt.xlabel('시간')
plt.ylabel('두께')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%
sns.kdeplot(data=df_cont, x='biscuit_thickness', hue='passorfail')
# %%
df.info()
# %%
daily_bt = df_cont.groupby(df_cont['datetime'].dt.date)['biscuit_thickness'].mean()

plt.figure(figsize=(15,5))
daily_bt.plot(marker='o')
plt.title('📉 일별 평균 Biscuit Thickness 추이')
plt.xlabel('날짜')
plt.ylabel('평균 두께')
plt.grid(True)
plt.show()
# %%
corr = df_cont[['biscuit_thickness','molten_temp','cast_pressure',
           'low_section_speed','high_section_speed']].corr()

print(corr['biscuit_thickness'])
# %%
import seaborn as sns

sns.boxplot(data=df_cont, x='is_fail', y='biscuit_thickness')
plt.title('불량 여부별 Biscuit Thickness 분포')
plt.show()
# %%
mold_bt = df_cont.groupby('mold_code')['biscuit_thickness'].mean().sort_values()

plt.figure(figsize=(12,6))
mold_bt.plot(kind='bar')
plt.title('금형별 평균 Biscuit Thickness')
plt.ylabel('평균 두께')
plt.show()
# %%
import seaborn as sns
import matplotlib.pyplot as plt

variables = ['biscuit_thickness','molten_temp','cast_pressure',
             'low_section_speed','high_section_speed']

plt.figure(figsize=(15,8))
for i, col in enumerate(variables, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df_cont, x='is_fail', y=col)
    plt.title(f'{col} by is_fail')
    plt.xlabel('불량 여부 (False=정상, True=불량)')
    plt.ylabel(col)

plt.tight_layout()
plt.show()
# %%
import seaborn as sns
import matplotlib.pyplot as plt

# 비교할 변수들
vars_to_check = ['biscuit_thickness','molten_temp','cast_pressure']


for var in vars_to_check:
    g = sns.FacetGrid(df_cont, col="mold_code", col_wrap=2, height=4, sharey=False)
    g.map_dataframe(sns.boxplot, x="is_fail", y=var, palette="Set2")
    g.set_axis_labels("불량 여부", var)
    g.set_titles(col_template="Mold {col_name}")
    plt.suptitle(f'Mold별 불량 vs 정상 {var} 분포', y=1.05)
    plt.tight_layout()
    plt.show()
# %%
mean_bt = df_clip['biscuit_thickness'].mean()
std_bt  = df_clip['biscuit_thickness'].std()

# ±3σ 범위 밖 → 이상치
outliers = df_clip[np.abs(df_clip['biscuit_thickness'] - mean_bt) > 3*std_bt]
# %%
fails = df_clip[df_clip['is_fail'] == True]
# %%
plt.figure(figsize=(14,5))

# 원시 데이터
plt.plot(df_clip['datetime'], df_clip['biscuit_thickness'],
         linestyle='-', marker='.', alpha=0.3, label='Raw')

# 이동평균
plt.plot(df_clip['datetime'], df_clip['bt_ma50'],
         linewidth=2, color='blue', label='MA(50)')

# 이상치 표시
plt.scatter(outliers['datetime'], outliers['biscuit_thickness'],
            color='orange', marker='x', s=60, label='Outlier (3σ)')

# 불량 샷 표시
plt.scatter(fails['datetime'], fails['biscuit_thickness'],
            color='red', marker='o', s=40, label='Fail Shot')

plt.title('시간에 따른 Biscuit Thickness 추이 (이상치·불량 표시)')
plt.xlabel('시간')
plt.ylabel('두께')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
sns.scatterplot(x ='sleeve_temperature', y='molten_temp', data=df_cont, hue='passorfail')
# %%
sns.boxplot(x='mold_code', y='molten_volume', data=df_cont)
# %%
sns.scatterplot(y='facility_operation_cycleTime', x='production_cycletime', hue='mold_code', data= df_cont)
# %%
vars_to_check = ['molten_volume','biscuit_thickness','molten_temp',
                 'cast_pressure','low_section_speed','high_section_speed']
# %%
sns.pairplot(df_cont, diag_kind='kde', plot_kws={'alpha':0.4, 's':20})
plt.suptitle("Molten Volume과 주요 변수 간 pairwise scatterplot", y=1.02)
plt.show()
# %%
import seaborn as sns
import matplotlib.pyplot as plt
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
y_vars = [
    'biscuit_thickness', 'molten_temp', 'cast_pressure',
    'low_section_speed', 'high_section_speed',
    'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3'
]
# %%
x_col = 'molten_volume'
y_vars = [
    'biscuit_thickness', 'molten_temp', 'cast_pressure',
    'low_section_speed', 'high_section_speed', 'sleeve_temperature'
    'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3'
]
use_hue = 'is_fail'         # 색상 구분 컬럼 (없으면 None)
max_points = 30000          # 데이터가 많을 때 샘플 수 (비활성화하려면 None)
ncols = 3                   # 그리드 컬럼 수
marker_size = 12            # 점 크기
alpha_pt = 0.35             # 점 투명도

# =========================
# 전처리
# =========================
# 실제 존재하는 컬럼만 사용
available_y = [c for c in y_vars if c in df.columns]
cols_needed = [x_col] + available_y + ([use_hue] if (use_hue and use_hue in df.columns) else [])
df_scatter = df[cols_needed].copy()

# 결측 처리: x 결측 제거, y가 전부 결측인 행 제거
df_scatter = df_scatter.dropna(subset=[x_col])
df_scatter = df_scatter.dropna(how='all', subset=available_y)

# 샘플링(옵션)
if (max_points is not None) and (len(df_scatter) > max_points):
    df_scatter = df_scatter.sample(max_points, random_state=42)

# hue 사용 가능 여부 판단
hue_ok = (use_hue is not None) and (use_hue in df_scatter.columns) and (df_scatter[use_hue].nunique() > 1)

# =========================
# 그리드 생성
# =========================
n = len(available_y)
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = np.atleast_1d(axes).ravel()

# =========================
# 플로팅
# =========================
for i, var in enumerate(available_y):
    ax = axes[i]
    # 산점도
    if hue_ok:
        # 첫 번째 서브플롯에만 범례 표시
        sns.scatterplot(
            data=df_scatter, x=x_col, y=var,
            hue=use_hue, alpha=alpha_pt, s=marker_size,
            ax=ax, legend=(i == 0)
        )
        if i == 0:
            ax.legend(title=use_hue, loc='best')
    else:
        sns.scatterplot(
            data=df_scatter, x=x_col, y=var,
            alpha=alpha_pt, s=marker_size, ax=ax
        )

    # 상관계수 계산(피어슨)
    sub = df_scatter[[x_col, var]].dropna()
    if (len(sub) >= 2) and (sub[x_col].nunique() > 1) and (sub[var].nunique() > 1):
        r = sub.corr(method='pearson').iloc[0, 1]
        title = f'{var}  (r = {r:.2f})'
    else:
        title = f'{var}  (r = N/A)'

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(var)
    ax.grid(True, alpha=0.3)

# 남는 축 제거
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(f'{x_col} vs 기타 변수 – 1:1 Scatterplot Grid', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
# %%
df_cont.columns
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== 사용자 환경 ======
DATETIME_COL = "datetime"     # 시간 컬럼
DEFECT_COL   = "is_fail"      # 불량 여부(0/1). 없으면 passorfail에서 매핑
MOLD_COL     = "mold_code"    # 금형 코드
USE_WORKING_ONLY = True       # working==1 인 샷만 분석할지 여부
RESAMPLE_RULE = "D"           # 'H','D','W' 등
ROLL_SHORT, ROLL_LONG = 7, 30 # 이동평균 윈도우 (리샘플 기준 단위)

# ====== 전처리 유틸 ======
def ensure_datetime_index(df, dt_col=DATETIME_COL, tz="Asia/Seoul"):
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).sort_values(dt_col)
    if df[dt_col].dt.tz is None:
        df[dt_col] = df[dt_col].dt.tz_localize(tz)
    return df.set_index(dt_col)

def to_binary_defect_series(df):
    if DEFECT_COL in df.columns:
        s = (df[DEFECT_COL].astype(float) > 0).astype(int)
    else:
        # passorfail 컬럼이 'PASS/FAIL' 또는 'OK/NG'라면 여기서 매핑
        pf = df["passorfail"].astype(str).str.upper()
        s = pf.isin({"FAIL","NG"}).astype(int)
    return s

# ====== 집계 ======
def build_ts(df, rule=RESAMPLE_RULE):
    df = ensure_datetime_index(df)
    # (옵션) 가동 샷만 사용
    if USE_WORKING_ONLY and "working" in df.columns:
        df = df[df["working"] == 1]

    y = to_binary_defect_series(df)
    shots   = df.resample(rule).size().rename("shots")
    defects = y.resample(rule).sum().rename("defects")

    ts = pd.concat([shots, defects], axis=1).fillna(0)
    ts["defect_rate_%"] = np.where(ts["shots"]>0, ts["defects"]/ts["shots"]*100, np.nan)

    # 이동평균
    ts["defects_ma_s"] = ts["defects"].rolling(ROLL_SHORT, min_periods=1).mean()
    ts["defects_ma_l"] = ts["defects"].rolling(ROLL_LONG,  min_periods=1).mean()
    ts["rate_ma_s"]    = ts["defect_rate_%"].rolling(ROLL_SHORT, min_periods=1).mean()
    ts["rate_ma_l"]    = ts["defect_rate_%"].rolling(ROLL_LONG,  min_periods=1).mean()
    return ts

# ====== 플롯 ======
def plot_defect_count(ts, title="시간 경과에 따른 불량 발생 추이"):
    plt.figure(figsize=(12,4))
    plt.plot(ts.index, ts["defects"], marker='.', linewidth=1, label="불량 수")
    plt.plot(ts.index, ts["defects_ma_s"], linewidth=2, label=f"MA({ROLL_SHORT})")
    plt.plot(ts.index, ts["defects_ma_l"], linewidth=2, label=f"MA({ROLL_LONG})")
    plt.title(title)
    plt.xlabel("시간"); plt.ylabel("건수"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_defect_rate(ts, title="시간 경과에 따른 불량률 추이"):
    plt.figure(figsize=(12,4))
    plt.plot(ts.index, ts["defect_rate_%"], marker='.', linewidth=1, label="불량률(%)")
    plt.plot(ts.index, ts["rate_ma_s"], linewidth=2, label=f"MA({ROLL_SHORT})")
    plt.plot(ts.index, ts["rate_ma_l"], linewidth=2, label=f"MA({ROLL_LONG})")
    plt.title(title)
    plt.xlabel("시간"); plt.ylabel("불량률(%)"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_by_mold(df, rule=RESAMPLE_RULE, max_molds=6):
    if MOLD_COL not in df.columns:
        print("[알림] mold_code 컬럼이 없어 스킵합니다."); return
    df = ensure_datetime_index(df)
    if USE_WORKING_ONLY and "working" in df.columns:
        df = df[df["working"] == 1]
    y = to_binary_defect_series(df)

    top_molds = df[MOLD_COL].value_counts().head(max_molds).index
    n = len(top_molds); cols = min(3, n); rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, mold in zip(axes, top_molds):
        d = df[df[MOLD_COL]==mold]
        ys = y.loc[d.index]
        shots = d.resample(rule).size()
        defects = ys.resample(rule).sum()
        ts = pd.concat([shots.rename("shots"), defects.rename("defects")], axis=1).fillna(0)
        ts["rate_%"] = np.where(ts["shots"]>0, ts["defects"]/ts["shots"]*100, np.nan)
        ts["rate_ma"] = ts["rate_%"].rolling(ROLL_SHORT, min_periods=1).mean()
        ax.plot(ts.index, ts["rate_%"], marker='.', linewidth=1, label="불량률(%)")
        ax.plot(ts.index, ts["rate_ma"], linewidth=2, label=f"MA({ROLL_SHORT})")
        ax.set_title(f"mold={mold}"); ax.set_xlabel("시간"); ax.set_ylabel("불량률(%)")
        ax.grid(True); ax.legend()

    for j in range(len(top_molds), len(axes)):
        axes[j].axis("off")
    fig.suptitle("mold_code별 불량률 추이", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.96])
    plt.show()

# %%
df_cont_B = df_cont[df_cont['heating_furnace'] == 'B']
# %%
df_cont_B
# %%
df_clip_B = df_cont_B.copy()
df_clip_B['bt_ma50'] = df_clip_B['molten_volume'].rolling(50, min_periods=1).mean()
# %%
plt.figure(figsize=(14,5))
plt.plot(df_clip_B['datetime'], df_clip_B['molten_volume'], linestyle='-', marker='.', alpha=0.35, label='Raw')
plt.plot(df_clip_B['datetime'], df_clip_B['bt_ma50'], linewidth=2, label='MA(50)')
plt.title('시간에 따른 molten_voulme 추이')
plt.xlabel('시간')
plt.ylabel('체적')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==== 0) 파라미터 ====
GAP_MIN = "30min"      # 공백 판단 임계치
ROLL_WIN = "2H"        # 시간기반 이동평균 창(예: '30min','2H','1D')
SHOW_FAIL = True       # 불량 샷 표시
SHOW_OUTLIER = True    # 3σ 이상치 표시

# ==== 1) 사본 + 전처리 ====
g = df_cont_B.copy()

# datetime 파싱/정렬 + 필수 컬럼
g["datetime"] = pd.to_datetime(g["datetime"], errors="coerce")
need_cols = ["datetime", "molten_volume"]
if SHOW_FAIL and "is_fail" not in g.columns:
    SHOW_FAIL = False
g = g.dropna(subset=need_cols).sort_values("datetime")

# ==== 2) 시간기반 이동평균 (불규칙 간격 대응) ====
#   - set_index 후 time-based rolling
g_ma = (g.set_index("datetime")
          .rolling(ROLL_WIN, min_periods=1)["molten_volume"]
          .mean()
          .rename("mv_ma"))
g = g.join(g_ma, on="datetime")

# ==== 3) 큰 시간 공백 식별 + 세그먼트 id ====
gap_mask = g["datetime"].diff().gt(pd.Timedelta(GAP_MIN))
seg_id = gap_mask.cumsum()

# ==== 4) 3σ 간이 이상치 (옵션) ====
if SHOW_OUTLIER:
    mu = g["molten_volume"].mean()
    sd = g["molten_volume"].std()
    outlier_idx = (g["molten_volume"] < mu - 3*sd) | (g["molten_volume"] > mu + 3*sd)
    g["_is_outlier"] = outlier_idx
else:
    g["_is_outlier"] = False

# ==== 5) 플롯 ====
fig, ax = plt.subplots(figsize=(14, 5))

# (a) 공백 구간 음영 처리
if gap_mask.any():
    gap_times = g.loc[gap_mask, "datetime"]
    # 음영: gap 이전 시점 ~ gap 시점 사이
    # 첫 시점
    prev_t = g["datetime"].iloc[0]
    for t in gap_times:
        if (t - prev_t) > pd.Timedelta(GAP_MIN):
            ax.axvspan(prev_t, t, color="grey", alpha=0.06, linewidth=0)
        prev_t = t

# (b) Raw: 세그먼트별로 선 연결 (빈 구간은 끊김)
for _, seg in g.groupby(seg_id):
    ax.plot(seg["datetime"], seg["molten_volume"],
            linestyle='-', marker='.', markersize=2, alpha=0.35, color='tab:blue', label=None)

# (c) 이동평균선
ax.plot(g["datetime"], g["mv_ma"], linewidth=2, color='tab:orange', label=f"MA({ROLL_WIN})")

# (d) 불량 샷 표시(옵션)
if SHOW_FAIL:
    fails = g[g["is_fail"] == True]
    if not fails.empty:
        ax.scatter(fails["datetime"], fails["molten_volume"],
                   s=28, color='red', label='Fail Shot', zorder=3)

# (e) 이상치 표시(옵션)
if SHOW_OUTLIER:
    outs = g[g["_is_outlier"]]
    if not outs.empty:
        ax.scatter(outs["datetime"], outs["molten_volume"],
                   marker='x', s=42, color='darkorange', label='Outlier (±3σ)', zorder=3)

# ==== 6) 서식 ====
ax.set_title("시간에 따른 molten_volume 추이")
ax.set_xlabel("시간"); ax.set_ylabel("체적")
ax.grid(True, alpha=0.3)

# x축 날짜 포맷
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
fig.autofmt_xdate()

# 범례: 중복 방지
handles, labels = ax.get_legend_handles_labels()
if handles:
    # 중복 제거
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='best')

fig.tight_layout()
plt.show()
# %%
sns.scatterplot(data=df_cont_B, x='datetime', y='molten_volume')
# %%
droped_B = df_cont_B.dropna()

# %%
sns.scatterplot(data=droped_B, x='datetime', y='molten_volume')
# %%
droped_B['date']
# %%
droped_B.info()
# %%
df_cont_B
# %%
sns.scatterplot(data=df_cont_B, x='time', y='molten_volume')
# %%
df_cont_B
# %%
df_cont_B['molten_volume']
# %%
df_cont_A = df_cont[df_cont['heating_furnace'] == 'A']
# %%
df_cont_A['molten_volume'].isnull().sum()
# %%
sns.scatterplot(data=df_cont, x='upper_mold_temp1', y='upper_mold_temp2', hue='passorfail')
# %%
sns.scatterplot(
    data=df_cont.query("upper_mold_temp3 <= 400 and upper_mold_temp2 <= 400"),
    x="upper_mold_temp3", y="upper_mold_temp2", hue="passorfail"
)
# %%
import plotly.express as px

dfp = df_cont.query(
    "upper_mold_temp1 <= 400 and upper_mold_temp2 <= 400 and upper_mold_temp3 <= 400"
)

fig = px.scatter_3d(
    dfp,
    x="upper_mold_temp1", y="upper_mold_temp2", z="upper_mold_temp3",
    color="passorfail",
    title="Upper Mold Temp1–3 (≤ 400°C)",
    opacity=0.85
)
fig.update_traces(marker=dict(size=3))  # ← 마커 작게(예: 3)
fig.show()
# %%
fig.update_traces(marker=dict(size=3))
fig.update_layout(width=800, height=600, margin=dict(l=0, r=0, t=60, b=0))
fig.show()
# %%
sns.histplot(data=df_cont, x='Coolant_temperature')
# %%
df_cont['Coolant_temperature'].describe()
# %%
fig.update_layout(
    updatemenus=[dict(
        buttons=[
            dict(label="All", method="restyle", args=[{"transforms": []}]),
            # 금형 값들로 버튼 생성
        ],
        x=1.15, y=1.0
    )]
)
# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %%
import umap
# %%

mold_temp_dat = dfp[
    [
        "upper_mold_temp1",
        "upper_mold_temp2",
        "upper_mold_temp3",
    ]
].values
# %%
scaled_temp_dat = StandardScaler().fit_transform(mold_temp_dat)
# %%
reducer = umap.UMAP()
# %%
embedding = reducer.fit_transform(scaled_temp_dat)
embedding.shape
# %%
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[int(x)] for x in dfp['passorfail']],
    s=2)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the diecast dataset', fontsize=24)
# %%
df_cont.info()
# %%
df_cont.shape
# %%
df_counts = (
    df_cont.groupby(cols)
    .size()
    .reset_index(name="count")
    .query("count > 1")
    .sort_values("count", ascending=False)
)

df_counts.head(20)
# %%
df_cont['sleeve_temperature'].describe()
# %%
sns.scatterplot(x='upper_mold_temp3', y ='Coolant_temperature', data=df_cont)
# %%
df_cont['upper_mold_temp3'].unique()
# %%
temp_cols = df_cont.filter(like="temp").columns.tolist()
print(temp_cols)
# %%
df_cont['sleeve_temperature'].unique()
# %%
df_cont['Coolant_temperature'].sort_values().unique()
# %%
temp_cols
# %%
mask = (df_cont[temp_cols] >= 1400).any(axis=1)   # 하나라도 1400 이상
df_filtered = df_cont.loc[~mask].copy()
# %%
sns.scatterplot(x='upper_mold_temp3', y ='Coolant_temperature', data=df_filtered)
# %%
df_filtered.info()
# %%
df_cont.info()
# %%
