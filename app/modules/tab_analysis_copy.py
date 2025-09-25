from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"
SCALER_FILE = BASE_DIR / "data" / "interim" / "std_scaler_v1.joblib"
MODEL_FILE = BASE_DIR / "data" / "interim" / "rf_model_v1.joblib"

# --- 데이터 로드 ---
df = pd.read_csv(DATA_FILE, encoding="utf-8", low_memory=False)
df = df[
    [
        "cast_pressure",
        "count",
        "upper_mold_temp1",
        "low_section_speed",
        "lower_mold_temp2",
        "high_section_speed",
        "upper_mold_temp2",
        "lower_mold_temp1",
        "biscuit_thickness",
        "sleeve_temperature",
    ]
]

# --- 스케일러/모델 로드 ---
std_scaler = joblib.load(SCALER_FILE)
rf_model = joblib.load(MODEL_FILE)

# --- 슬라이더 메타(최솟값~최댓값) ---
slider_meta = {}
for col in df.columns:
    vmin = float(df[col].min())
    vmax = float(df[col].max())
    vdef = float(df[col].median())
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    step = (vmax - vmin) / 200.0 if vmax > vmin else 0.1
    series = df[col].dropna()
    looks_integer = (series.round() == series).all()
    if looks_integer:
        step = max(1, round(step))
    slider_meta[col] = {"min": vmin, "max": vmax, "value": vdef, "step": step}

def panel():
    main_inputs = []
    for col in df.columns:
        meta = slider_meta[col]
        main_inputs.append(
            ui.input_slider(
                col, f"{col}:", min=meta["min"], max=meta["max"],
                value=meta["value"], step=meta["step"]
            )
        )

    return ui.nav_panel(
        "개별 예측",
        ui.page_sidebar(
            ui.sidebar(
                ui.input_select("model", "모델 선택:", choices=["랜덤 포레스트"]),
                ui.input_action_button("predict", "예측 실행"),
                ui.card(ui.output_text("prediction_result")),
            ),
            ui.layout_column_wrap(*main_inputs),
        ),
    )

def server(input, output, session):
    # 초기 메시지
    result_text = reactive.Value("예측 실행 버튼을 눌러 결과를 확인하세요")

    # 버튼 클릭시에만 예측 수행
    @reactive.effect
    @reactive.event(input.predict)   # <-- predict 버튼이 '이벤트 트리거'
    def _run_prediction():
        # 현재 슬라이더 값 읽기 (이 안에선 자동으로 isolate 되어 버튼에만 반응)
        input_row = {col: float(getattr(input, col)()) for col in df.columns}
        input_df = pd.DataFrame([input_row], columns=df.columns)

        # 스케일링 -> 예측
        X_scaled = std_scaler.transform(input_df)
        pred = rf_model.predict(X_scaled)[0]
        result = "불량" if int(pred) == 1 else "양품"

        # 출력 텍스트 업데이트
        result_text.set(f"예측 결과: {result}")

    @output
    @render.text
    def prediction_result():
        # 항상 현재의 result_text만 표시 (버튼 누르기 전엔 안내 문구)
        return result_text.get()

app = App(panel, server)
