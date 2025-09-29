import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

# imbalanced-learn (SMOTE/SMOTENC + imblearn Pipeline)
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

# Optuna
import optuna

# parallel / thread control
from joblib import parallel_backend
from threadpoolctl import threadpool_limits

# ---------------------------
# 파일 경로 설정
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE_TRAIN = BASE_DIR / "data" / "processed" / "train_v1.csv"
DATA_FILE_TEST = BASE_DIR / "data" / "processed" / "test_v1.csv"


# ---------------------------
# 데이터 로드
# ---------------------------
train_df = pd.read_csv(DATA_FILE_TRAIN)
test_df = pd.read_csv(DATA_FILE_TEST)
train_df.drop(columns=["date", "time"], inplace=True)

# ---------------------------
# 타깃 처리
# ---------------------------
target_col = "passorfail"

y_train = train_df[target_col].copy().astype(int)
X_train = train_df.drop(columns=[target_col])

y_test = test_df[target_col].copy().astype(int)
X_test = test_df[train_df.columns]

# 숫자/범주형 컬럼
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()


# ---------------------------
# Preprocessing 구성
# ---------------------------

num_pipe_for_smote = SkPipeline([("scaler", StandardScaler())])
cat_pipe_for_smote = SkPipeline([("ord", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

preprocessor_for_smote = ColumnTransformer(
    transformers=[
        ("num", num_pipe_for_smote, numeric_cols),
        ("cat", cat_pipe_for_smote, categorical_cols),
    ],
    remainder="drop"
)


n_num = len(numeric_cols)
n_cat = len(categorical_cols)
cat_indices_for_smotenc = list(range(n_num, n_num + n_cat)) if n_cat > 0 else []

# 3) Post-SMOTE processing: OneHotEncoder for categorical columns, passthrough numeric
# We will use ColumnTransformer that expects the SMOTEd numpy array layout above.
# For numeric: passthrough (they are already scaled)
# For categorical: OneHotEncoder
from sklearn.compose import ColumnTransformer as CT_after_smote

if n_cat > 0:
    post_smote_transformer = CT_after_smote(
        transformers=[
            ("num_passthrough", "passthrough", list(range(0, n_num))),
            ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_indices_for_smotenc),
        ],
        remainder="drop"
    )
else:
    # 카테고리가 없으면 숫자만 passthrough
    post_smote_transformer = CT_after_smote(
        transformers=[("num_passthrough", "passthrough", list(range(0, n_num)))],
        remainder="drop"
    )

# ---------------------------
# Optuna objective (SMOTENC 포함)
# ---------------------------
def objective(trial):
    """
    Optuna objective: trial로부터 하이퍼파라미터를 얻어 imblearn 파이프라인을 만들어
    cross_validate로 average_precision(=PR AUC)을 교차검증하여 반환합니다.
    """
    # ---------- 모델 하이퍼파라미터 ----------
    solver = trial.suggest_categorical("solver", ["liblinear", "saga", "lbfgs"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C = trial.suggest_loguniform("C", 1e-4, 1e2)
    l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
    use_balanced = trial.suggest_categorical("class_weight_balanced", [True, False])
    class_weight = "balanced" if use_balanced else None
    max_iter = trial.suggest_categorical("max_iter", [200, 500, 1000])

    # solver/penalty 호환성 보정
    if solver == "lbfgs" and penalty != "l2":
        penalty_effective = "l2"
    elif solver == "liblinear" and penalty == "elasticnet":
        penalty_effective = "l1"
    else:
        penalty_effective = penalty

    lr_kwargs = {
        "C": C,
        "solver": solver,
        "penalty": penalty_effective,
        "class_weight": class_weight,
        "max_iter": max_iter,
        "random_state": 42,
    }
    if penalty_effective == "elasticnet":
        lr_kwargs["l1_ratio"] = l1_ratio

    # ---------- SMOTE/SMOTENC 하이퍼파라미터 ----------
    use_smote = trial.suggest_categorical("use_smote", [True, False])
    # use_smotenc only meaningful if there are categorical columns
    use_smotenc = False
    if n_cat > 0 and use_smote:
        use_smotenc = trial.suggest_categorical("use_smotenc", [True, False])
    # sampling strategy: float -> minority to majority ratio after sampling or 'auto'
    sampling_ratio = trial.suggest_float("sampling_ratio", 0.5, 1.0)  # e.g., 1.0 means balance to equal
    smote_k = trial.suggest_int("smote_k", 3, 7)

    # build sampler
    sampler = None
    if use_smote:
        if use_smotenc and n_cat > 0:
            # SMOTENC: requires categorical_features indices relative to input array
            sampler = SMOTENC(
                categorical_features=cat_indices_for_smotenc,
                sampling_strategy=sampling_ratio,
                k_neighbors=smote_k,
                random_state=42,)
        else:
            sampler = SMOTE(
                sampling_strategy=sampling_ratio,
                k_neighbors=smote_k,
                random_state=42,)

    # ---------------------------
    # 구성: imblearn Pipeline:
    # steps: preprocessor_for_smote -> (sampler if any) -> post_smote_transformer -> classifier
    # Note: preprocessor_for_smote and post_smote_transformer are ColumnTransformers;
    # imblearn Pipeline applies fit/transform sequentially and ensures sampling occurs only in fit().
    # ---------------------------

    steps = [("preproc_for_smote", preprocessor_for_smote)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("post_smote", post_smote_transformer))
    steps.append(("clf", LogisticRegression(**lr_kwargs)))

    imb_pipeline = ImbPipeline(steps)

    # ---------------------------
    # cross_validate (CV)
    # ---------------------------
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold로 빠르게 수행
    scoring = {"avg_prec": "average_precision"}

    # 병렬 안전 실행: BLAS/OpenMP 내부 스레드를 1로 제한 및 loky 백엔드 사용
    with threadpool_limits(limits=1):
        with parallel_backend("loky", n_jobs=-1):
            try:
                cv_results = cross_validate(
                    imb_pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    error_score="raise",
                    return_train_score=False
                )
            except Exception as e:
                # 예: SMOTE의 k_neighbors가 너무 큰 경우 ValueError 발생 가능
                # 이런 trial은 실패로 간주하고 매우 낮은 점수(=0.0) 리턴하여 Optuna가 피하게 함
                # 디버깅용으로 예외 메시지도 출력
                print("Trial failed during cross_validate:", e)
                return 0.0

    avg_prec_mean = float(np.mean(cv_results["test_avg_prec"]))
    # Optuna는 maximize 하도록 study 설정할 것
    return avg_prec_mean

# ---------------------------
# Optuna 스터디 실행 (빠르게 설정)
# ---------------------------
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
N_TRIALS = 20  # 빠른 실행을 위해 기본 20 (필요하면 늘리세요)
print(f"Start optimization with {N_TRIALS} trials...")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ---------------------------
# 결과 출력
# ---------------------------
print("Best trial:")
trial = study.best_trial
print("  Value (best avg precision): ", trial.value)
print("  Params: ")
for k, v in trial.params.items():
    print(f"    {k}: {v}")

# ---------------------------
# 베스트 파라미터로 최종 학습 (전체 train 사용)
# ---------------------------
best_params = trial.params

# 모델 하이퍼파라미터 재구성(동일 보정 적용)
solver = best_params.get("solver")
penalty = best_params.get("penalty")
C = best_params.get("C")
class_weight = "balanced" if best_params.get("class_weight_balanced", False) else None
max_iter = best_params.get("max_iter", 1000)
l1_ratio = best_params.get("l1_ratio", None)

if solver == "lbfgs" and penalty != "l2":
    penalty_effective = "l2"
elif solver == "liblinear" and penalty == "elasticnet":
    penalty_effective = "l1"
else:
    penalty_effective = penalty

lr_kwargs_final = {
    "C": C,
    "solver": solver,
    "penalty": penalty_effective,
    "class_weight": class_weight,
    "max_iter": max_iter,
    "random_state": 42,
}
if penalty_effective == "elasticnet" and l1_ratio is not None:
    lr_kwargs_final["l1_ratio"] = l1_ratio

# SMOTE/SMOTENC 선택 재구성
use_smote_final = best_params.get("use_smote", False)
use_smotenc_final = best_params.get("use_smotenc", False) if n_cat > 0 else False
sampling_ratio_final = best_params.get("sampling_ratio", 1.0)
smote_k_final = best_params.get("smote_k", 5)

sampler_final = None
if use_smote_final:
    if use_smotenc_final and n_cat > 0:
        sampler_final = SMOTENC(
            categorical_features=cat_indices_for_smotenc,
            sampling_strategy=sampling_ratio_final,
            k_neighbors=smote_k_final,
            random_state=42,
        )
    else:
        sampler_final = SMOTE(
            sampling_strategy=sampling_ratio_final,
            k_neighbors=smote_k_final,
            random_state=42,
        )

# Pipeline 재구성 (train 전체 데이터에 fit)
steps_final = [("preproc_for_smote", preprocessor_for_smote)]
if sampler_final is not None:
    steps_final.append(("sampler", sampler_final))
steps_final.append(("post_smote", post_smote_transformer))
steps_final.append(("clf", LogisticRegression(**lr_kwargs_final)))

final_pipeline = ImbPipeline(steps_final)

# 안전한 학습: BLAS 스레드 제한 적용
with threadpool_limits(limits=1):
    final_pipeline.fit(X_train, y_train)

# ---------------------------
# 예측 및 성능 평가
# ---------------------------
y_pred = final_pipeline.predict(X_test)

# predict_proba는 pipeline의 마지막 추정기가 제공하면 사용
try:
    y_score = final_pipeline.predict_proba(X_test)[:, 1]
except Exception:
    # probability 미지원 시 decision_function 사용 시도
    try:
        y_score = final_pipeline.decision_function(X_test)
    except Exception:
        y_score = y_pred.astype(float)

# emergency_stop 규칙(있다면)
if "emergency_stop" in X_test.columns:
    mask = X_test["emergency_stop"].isna()
    if mask.any():
        y_pred = y_pred.copy()
        y_pred[mask] = 1
        if isinstance(y_score, np.ndarray):
            y_score[mask] = 1.0

roc_auc = roc_auc_score(y_test, y_score)
pr_auc = average_precision_score(y_test, y_score)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n===== Final metrics on test set (best model with SMOTENC/SMOTE) =====")
print(f"ROC AUC : {roc_auc:.4f}")
print(f"PR AUC  : {pr_auc:.4f} (average precision)")
print(f"F1      : {f1:.4f}")
print(f"Recall  : {recall:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# ROC & PR 곡선 시각화
# ---------------------------
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
