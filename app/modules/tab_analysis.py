from shiny import App, ui, render
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # 예시
import joblib
from sklearn.preprocessing import StandardScaler


# 예시 데이터프레임
df = pd.read_csv('../Casting-Process-Data-Modeling-Project/data/raw/train.csv', encoding='utf-8')
df = df[["cast_pressure", "count", "upper_mold_temp1", "low_section_speed", "lower_mold_temp2", 
         "high_section_speed", "upper_mold_temp2", "lower_mold_temp1", "biscuit_thickness", "sleeve_temperature"]]

#스케일링
std_scaler = joblib.load("../Casting-Process-Data-Modeling-Project/data/interim/std_scaler_v1.joblib")
# 학습된 랜덤 포레스트 모델 로드 (예시)
# rf_model = joblib.load("rf_model.pkl")
# rf_model = RandomForestClassifier()
# rf_model.fit(df, df['passorfail'])  # 예시 학습, 실제로는 이미 학습된 모델 사용
rf_model = joblib.load("../Casting-Process-Data-Modeling-Project/data/interim/rf_model_v1.joblib")

def panel():
    # 메인 영역에 데이터프레임 열만큼 input 생성
    main_inputs = []
    for col in df.columns:
        main_inputs.append(ui.input_numeric(col, f"{col}:", value=float(df[col].iloc[0])))

    return ui.nav_panel(
        "개별 예측",
        ui.page_sidebar(
            # 사이드바 영역: 모델만 선택
            ui.sidebar(
                ui.input_select(
                    "model", "모델 선택:",
                    choices=["랜덤 포레스트"]  # 실제 모델 이름
                ),
                ui.input_action_button("predict", "예측 실행"),
                ui.card(ui.output_text("prediction_result"))
            ),
            # 메인 영역: 데이터 입력
            ui.layout_column_wrap(
                *main_inputs
            )
        )
    )

def server(input, output, session):

    @output
    @render.text
    def prediction_result():
        if input.predict() > 0:
            # 입력값을 df 형태로 변환
            input_data = pd.DataFrame({
                col: [float(getattr(input, col)())]
                for col in df.columns
            })
            #
            std_scaler.transform(input_data)
            # 랜덤 포레스트 모델 예측
            pred = rf_model.predict(input_data)[0]
            result = "불량" if pred == 1 else "양품"
            return f"예측 결과: {result}"
        return "예측 결과 대기 중..."

app = App(panel, server)
