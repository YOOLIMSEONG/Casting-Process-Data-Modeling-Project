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


# =====================================
# cast_pressure 제거 후 모델 성능 확인
# =====================================
X_drop = X.drop(columns=['cast_pressure'])

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_drop, y, test_size=0.3, random_state=42, stratify=y
)

rf_model_d = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model_d.fit(X_train_d, y_train_d)
y_pred_d = rf_model_d.predict(X_test_d)

accuracy_d = accuracy_score(y_test_d, y_pred_d)
precision_d = precision_score(y_test_d, y_pred_d, average='weighted', zero_division=0)
recall_d = recall_score(y_test_d, y_pred_d, average='weighted', zero_division=0)
f1_d = f1_score(y_test_d, y_pred_d, average='weighted', zero_division=0)

print("\n=== cast_pressure 제거 후 성능 ===")
print(f"Accuracy: {accuracy_d:.4f}")
print(f"Precision: {precision_d:.4f}")
print(f"Recall: {recall_d:.4f}")
print(f"F1-Score: {f1_d:.4f}")


# =====================================
# cast_pressure 제거 후 변수 중요도 분석
# =====================================
X_drop = X.drop(columns=['cast_pressure'])

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_drop, y, test_size=0.3, random_state=42, stratify=y
)

rf_model_d = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model_d.fit(X_train_d, y_train_d)

# --- 변수 중요도
feature_importance_df_d = pd.DataFrame({
    '변수명': X_drop.columns,
    '중요도': rf_model_d.feature_importances_
}).sort_values('중요도', ascending=False)

top_10_features_d = feature_importance_df_d.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_features_d, x='중요도', y='변수명', palette="viridis")
plt.title("cast_pressure 제거 후 상위 10개 변수 중요도", fontsize=14)
plt.xlabel("중요도")
plt.ylabel("변수명")
plt.tight_layout()
plt.show()

# --- 순열 중요도 (Permutation Importance)
perm_importance_d = permutation_importance(
    rf_model_d, X_test_d, y_test_d, n_repeats=10, random_state=42, n_jobs=-1
)
perm_importance_df_d = pd.DataFrame({
    '변수명': X_drop.columns,
    '순열중요도': perm_importance_d.importances_mean
}).sort_values('순열중요도', ascending=False)

top_10_perm_d = perm_importance_df_d.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_perm_d, x='순열중요도', y='변수명', palette="magma")
plt.title("cast_pressure 제거 후 상위 10개 변수 (Permutation Importance)", fontsize=14)
plt.xlabel("순열중요도")
plt.ylabel("변수명")
plt.tight_layout()
plt.show()

