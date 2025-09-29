import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

### ==============================
# 0. 데이터 로드
### ==============================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "train_v1.csv"

# date/time이 test파일에 없을 수도 있기 때문에 안전하게 로드할 수 있는 코드
try:
    train_df = pd.read_csv(DATA_FILE, parse_dates=["date", "time"])
except Exception:
    train_df = pd.read_csv(DATA_FILE)

# 가끔 생기는 Unnamed 인덱스 컬럼 자동 제거
unnamed_cols = [c for c in train_df.columns if c.lower().startswith("unnamed")]
if unnamed_cols:
    train_df = train_df.drop(columns=unnamed_cols)

# 타겟이 문자열이면 숫자로 바꾸기, 오류뜨면 0으로 채우기
if train_df["passorfail"].dtype == "object":
    train_df["passorfail"] = pd.to_numeric(train_df["passorfail"], errors="coerce").fillna(0).astype(int)

# =====================================
# 1. 데이터 준비
# =====================================
X = train_df.drop(columns=['passorfail'])
y = train_df['passorfail'].astype(int)

# 문자열 컬럼 라벨인코딩 (간단 인코딩)
string_cols = X.select_dtypes(include=['object']).columns.tolist()
if string_cols:
    for col in string_cols:
        X[col] = X[col].astype('category').cat.codes

# 날짜/시간 처리 (있으면 파생변수 만들고 원본 제거)
date_time_cols = []

if 'date' in X.columns:
    if X['date'].dtype == 'object':
        X['date'] = pd.to_datetime(X['date'], errors='coerce')
    if pd.api.types.is_datetime64_any_dtype(X['date']):
        X['year'] = X['date'].dt.year
        X['month'] = X['date'].dt.month
        X['day'] = X['date'].dt.day
        X['dayofweek'] = X['date'].dt.dayofweek
        date_time_cols.append('date')

if 'time' in X.columns:
    if X['time'].dtype == 'object':
        X['time'] = pd.to_datetime(X['time'], errors='coerce')
    if pd.api.types.is_datetime64_any_dtype(X['time']):
        X['hour'] = X['time'].dt.hour
        X['minute'] = X['time'].dt.minute
        date_time_cols.append('time')

if date_time_cols:
    X = X.drop(columns=date_time_cols)

# 혹시 남은 비수치 컬럼 삭제(안전망)
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    X = X.drop(columns=non_numeric_cols)

# 훈련/테스트 분할 (심한 불균형 → stratify 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =====================================
# 2. 랜덤포레스트 모델 학습
# =====================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # 불균형 데이터 대응
)

rf_model.fit(X_train, y_train)

# =====================================
# 3. 기본 성능 평가
# =====================================
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

performance_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [accuracy, precision, recall, f1]
})

# =====================================
# 4. 변수 중요도 분석
# =====================================
feature_importance_df = pd.DataFrame({
    '변수명': X.columns,
    '중요도': rf_model.feature_importances_
}).sort_values('중요도', ascending=False)

top_10_features = feature_importance_df.head(10)

perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
perm_importance_df = pd.DataFrame({
    '변수명': X.columns,
    '순열중요도': perm_importance.importances_mean
}).sort_values('순열중요도', ascending=False)
top_10_perm = perm_importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_10_features['변수명'], top_10_features['중요도'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(top_10_perm['변수명'], top_10_perm['순열중요도'])
plt.xlabel('Permutation Importance')
plt.title('Top 10 Permutation Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =====================================
# 5. 교차검증
# =====================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_accuracy = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
cv_scores_f1 = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
cv_scores_precision = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='precision_weighted', n_jobs=-1)
cv_scores_recall = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='recall_weighted', n_jobs=-1)

cv_results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
    'Mean': [cv_scores_accuracy.mean(), cv_scores_f1.mean(),
             cv_scores_precision.mean(), cv_scores_recall.mean()],
    'Std': [cv_scores_accuracy.std(), cv_scores_f1.std(),
            cv_scores_precision.std(), cv_scores_recall.std()]
})
print("\n[CV 성능(Train Fold)]")
print(cv_results_df)


# =====================================
# 6. 혼동 행렬
# =====================================
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['True_0','True_1'], columns=['Pred_0','Pred_1'])

#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
# =====================================
# 0. 단일 변수 하나로 정확도 확인하기 -> 0.989604...
# =====================================
from sklearn.metrics import accuracy_score

# cast_pressure 하나만 사용
X_single = X[['cast_pressure']]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single, y, test_size=0.3, stratify=y, random_state=42
)

rf_single = RandomForestClassifier(random_state=42)
rf_single.fit(X_train_s, y_train_s)
y_pred_s = rf_single.predict(X_test_s)

print("Accuracy (cast_pressure only):", accuracy_score(y_test_s, y_pred_s))
# =====================================
# 1. cast_pressure과 다른 변수들 간 상관관계 살펴보기
# =====================================
corr = train_df.corr(numeric_only=True)
corr['cast_pressure'].sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

# 한글 폰트 설정 (Windows: 맑은 고딕)
plt.rcParams['axes.unicode_minus'] = False
rc('font', family='Malgun Gothic')

# cast_pressure와 다른 변수들의 상관관계 추출
corr = train_df.corr(numeric_only=True)
corr_cast = corr[['cast_pressure']].sort_values(by='cast_pressure', ascending=False)

# 히트맵 그리기
plt.figure(figsize=(4,8))
sns.heatmap(corr_cast, annot=True, cmap="RdBu_r", center=0, fmt=".3f")
plt.title("주조 압력(cast_pressure)과 다른 변수들의 상관관계", fontsize=13)
plt.show
# =====================================
# 2. cast_pressure/passorfail 관계 산점도
# =====================================

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

# 한글 폰트 설정 (Windows 기준: 맑은 고딕)
plt.rcParams['axes.unicode_minus'] = False
rc('font', family='Malgun Gothic')

plt.figure(figsize=(10,6))
sns.stripplot(
    data=train_df,
    x="passorfail",       # 합격 여부 (0=합격, 1=불합격)
    y="cast_pressure",    # 주조 압력 값
    jitter=0.3,           # 살짝 흩뿌려서 겹침 방지
    alpha=0.5,            # 점 투명도
    palette="Set1"
)

plt.title("주조 압력과 합격 여부 산점도", fontsize=14)
plt.xlabel("합격 여부 (0=합격, 1=불합격)")
plt.ylabel("주조 압력 (cast_pressure)")
plt.show()



#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

# =====================================
# 1차원 임계값(단일 변수 규칙)으로 성능 평가
#  - cast_pressure 하나만 사용
#  - 임계값은 '훈련셋'에서 고르고, '테스트셋'에 적용
# =====================================
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1) 단일 변수용 훈련/테스트 (이미 위에서 만든 X_train_s, X_test_s, y_train_s, y_test_s 재사용)
x_tr = X_train_s['cast_pressure'].to_numpy()
y_tr = y_train_s.to_numpy().astype(int)
x_te = X_test_s['cast_pressure'].to_numpy()
y_te = y_test_s.to_numpy().astype(int)

# 2) 훈련셋에서 임계값 후보 생성 (극단 제외, 1%~99% 구간)
cands = np.quantile(x_tr, np.linspace(0.01, 0.99, 199))

def pick_best_threshold(x, y):
    best = None
    for direction in ('low_pass', 'high_pass'):  # low_pass: x<=t -> 합격(1), high_pass: x>=t -> 합격(1)
        for t in cands:
            if direction == 'low_pass':
                y_pred = (x <= t).astype(int)
            else:
                y_pred = (x >= t).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            acc = accuracy_score(y, y_pred)
            # F1을 우선, 동률이면 정확도 높은 쪽
            if (best is None) or (f1 > best['f1']) or (f1 == best['f1'] and acc > best['acc']):
                best = {'t': float(t), 'dir': direction, 'f1': float(f1), 'acc': float(acc)}
    return best

# 3) 훈련셋에서 최적 임계값/방향 선택
best_tr = pick_best_threshold(x_tr, y_tr)

# 4) 선택된 임계값을 테스트셋에 적용하여 성능 측정
if best_tr['dir'] == 'low_pass':
    y_hat = (x_te <= best_tr['t']).astype(int)
    rule_text = f"cast_pressure ≤ {best_tr['t']:.3f} → 합격(1), 초과 → 불합격(0)"
else:
    y_hat = (x_te >= best_tr['t']).astype(int)
    rule_text = f"cast_pressure ≥ {best_tr['t']:.3f} → 합격(1), 미만 → 불합격(0)"

acc  = accuracy_score(y_te, y_hat)
pre  = precision_score(y_te, y_hat, zero_division=0)
rec  = recall_score(y_te, y_hat, zero_division=0)
f1sc = f1_score(y_te, y_hat, zero_division=0)
cm   = confusion_matrix(y_te, y_hat)

print("\n[단일 변수 임계값 규칙 성능 (테스트셋)]")
print("규칙:", rule_text)
print(f"Accuracy={acc:.4f}, Precision={pre:.4f}, Recall={rec:.4f}, F1={f1sc:.4f}")
print("[Confusion Matrix]\n", cm)
print(classification_report(y_te, y_hat, digits=4))

# (선택) 시각화: 테스트셋 분포 + 임계값 표시
import seaborn as sns
import matplotlib.pyplot as plt
te_plot = pd.DataFrame({'cast_pressure': x_te, 'passorfail': y_te})
plt.figure(figsize=(9,5))
sns.histplot(data=te_plot, x='cast_pressure', hue='passorfail', bins=50, kde=True, alpha=0.5)
plt.axvline(best_tr['t'], ls='--', lw=2, color='k', label=f"임계값 = {best_tr['t']:.2f}")
plt.title("cast_pressure 단일 임계값 규칙 (테스트셋 분포 + 임계선)")
plt.xlabel("cast_pressure"); plt.ylabel("Count"); plt.legend(); plt.tight_layout(); plt.show()




