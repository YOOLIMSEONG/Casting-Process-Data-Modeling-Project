import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import optuna

# 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "train_v1.csv"

# 데이터 로드
df = pd.read_csv(DATA_FILE)

# 타깃 처리
target_col = "passorfail"

# y 선택
y = df[target_col].copy()
y = y.astype(int)

# X 선택
X = df.drop(columns=[target_col])

# 수치형, 범주형 컬럼
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 전처리 파이프라인
num_pipe = Pipeline([("scaler", StandardScaler())])
cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

transformers = []
transformers.append(("num", num_pipe, numeric_cols))
transformers.append(("cat", cat_pipe, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# Optuna objective (수정된 부분)
def objective(trial):
    # solver (고정 후보)
    solver = trial.suggest_categorical("solver", ["saga", "liblinear", "lbfgs"])

    # penalty는 항상 동일한 후보 집합으로 샘플 (동적 변경 금지)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

    # C (log-uniform)
    C = trial.suggest_loguniform("C", 1e-4, 1e2)

    # l1_ratio는 항상 샘플해두고 필요하면 사용 (elasticnet일 때만)
    l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)

    # class_weight 선택
    use_balanced = trial.suggest_categorical("class_weight_balanced", [True, False])
    class_weight = "balanced" if use_balanced else None

    max_iter = trial.suggest_categorical("max_iter", [200, 500, 1000])

    # --- 호환성 보정: solver가 지원하지 않는 penalty면 안전하게 변경 ---
    # lbfgs: only supports 'l2' (놓치면 강제 변경)
    if solver == "lbfgs" and penalty != "l2":
        penalty_effective = "l2"
    # liblinear: supports l1, l2 but NOT elasticnet
    elif solver == "liblinear" and penalty == "elasticnet":
        penalty_effective = "l1"  # elasticnet 불가 -> l1으로 대체(또는 l2)
    else:
        penalty_effective = penalty

    # --- lr kwargs 조립 ---
    lr_kwargs = {
        "C": C,
        "solver": solver,
        "penalty": penalty_effective,
        "class_weight": class_weight,
        "max_iter": max_iter,
        "random_state": 42,
    }
    # elasticnet일때만 l1_ratio 전달 (그 외엔 무시)
    if penalty_effective == "elasticnet":
        lr_kwargs["l1_ratio"] = l1_ratio

    # 모델 파이프라인
    model = Pipeline([("preproc", preprocessor), ("clf", LogisticRegression(**lr_kwargs))])

    # CV로 average_precision(=PR AUC) 계산
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"avg_prec": "average_precision"}
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    avg_prec_mean = np.mean(cv_results["test_avg_prec"])

    # (옵션) 실패 조합을 아주 낮은 점수로 반환하는 방법 대신 위 보정으로 오류를 피함.
    return avg_prec_mean

# Optuna 실행
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
N_TRIALS = 60
print(f"Start optimization with {N_TRIALS} trials...")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Best trial:")
trial = study.best_trial
print("  Value (best avg precision): ", trial.value)
print("  Params: ")
for k, v in trial.params.items():
    print(f"    {k}: {v}")

# 베스트 파라미터로 모델 재학습 및 평가 (같은 방식)
best_params = trial.params
solver = best_params.get("solver")
penalty = best_params.get("penalty")
C = best_params.get("C")
class_weight = "balanced" if best_params.get("class_weight_balanced", False) else None
max_iter = best_params.get("max_iter", 1000)
l1_ratio = best_params.get("l1_ratio", None)

# 동일한 호환성 보정 적용
if solver == "lbfgs" and penalty != "l2":
    penalty_effective = "l2"
elif solver == "liblinear" and penalty == "elasticnet":
    penalty_effective = "l1"
else:
    penalty_effective = penalty

lr_kwargs = {"C": C, "solver": solver, "penalty": penalty_effective, "class_weight": class_weight, "max_iter": max_iter, "random_state": 42}
if penalty_effective == "elasticnet" and l1_ratio is not None:
    lr_kwargs["l1_ratio"] = l1_ratio

best_model = Pipeline([("preproc", preprocessor), ("clf", LogisticRegression(**lr_kwargs))])
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
if hasattr(best_model, "predict_proba"):
    y_score = best_model.predict_proba(X_test)[:, 1]
else:
    try:
        y_score = best_model.decision_function(X_test)
    except Exception:
        y_score = y_pred

# 규칙 기반 오버라이드 (있다면)
if "emergency_stop" in X_test.columns:
    mask = X_test["emergency_stop"].isna()
    if mask.any():
        y_pred = y_pred.copy()
        y_pred[mask] = 1
        if isinstance(y_score, np.ndarray):
            y_score[mask] = 1.0

# 지표 출력
roc_auc = roc_auc_score(y_test, y_score)
pr_auc = average_precision_score(y_test, y_score)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n===== Final metrics on test set (best model) =====")
print(f"ROC AUC : {roc_auc:.4f}")
print(f"PR AUC  : {pr_auc:.4f} (average precision)")
print(f"F1      : {f1:.4f}")
print(f"Recall  : {recall:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ROC & PR curves 시각화
fpr, tpr, _ = roc_curve(y_test, y_score)
precision, recall_vals, _ = precision_recall_curve(y_test, y_score)

plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend()

plt.tight_layout()
plt.show()
