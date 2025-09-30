from shiny import App, ui, render, reactive
import pandas as pd
import joblib
from pathlib import Path

# 경로 및 데이터 로드
BASE_DIR = Path(__file__).resolve().parents[2]
df = pd.read_csv(BASE_DIR / "data" / "processed" / "test_v1.csv", encoding="utf-8", low_memory=False)
std_scaler = joblib.load(BASE_DIR / "data" / "interim" / "std_scaler_v1.joblib")
rf_model = joblib.load(BASE_DIR / "data" / "interim" / "rf_model_v1.joblib")


# 변수 정의
ALL_COLUMNS = ['count', 'working', 'molten_temp', 'facility_operation_cycleTime', 'production_cycletime',
               'low_section_speed', 'high_section_speed', 'cast_pressure', 'biscuit_thickness',
               'upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1', 'lower_mold_temp2',
               'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 'EMS_operation_time', 'mold_code']

CATEGORICAL_COLS = ['working', 'EMS_operation_time', 'mold_code']
EXCLUDE_COLS = ['time', 'date', 'registration_time', 'tryshot_signal', 'emergency_stop', 'passorfail']

COLUMN_NAMES_KR = {
    'count': '일자별 생산 번호', 'working': '가동 여부', 'molten_temp': '용탕 온도',
    'facility_operation_cycleTime': '설비 작동 사이클 시간', 'production_cycletime': '제품 생산 사이클 시간',
    'low_section_speed': '저속 구간 속도', 'high_section_speed': '고속 구간 속도',
    'cast_pressure': '주조 압력', 'biscuit_thickness': '비스켓 두께',
    'upper_mold_temp1': '상금형 온도1', 'upper_mold_temp2': '상금형 온도2',
    'lower_mold_temp1': '하금형 온도1', 'lower_mold_temp2': '하금형 온도2',
    'sleeve_temperature': '슬리브 온도', 'physical_strength': '형체력',
    'Coolant_temperature': '냉각수 온도', 'EMS_operation_time': '전자교반 가동 시간', 'mold_code': '금형 코드'
}

all_cols = [col for col in ALL_COLUMNS if col in df.columns and col not in EXCLUDE_COLS]
categorical_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
numeric_cols = [col for col in all_cols if col not in categorical_cols]

# CSS 및 JS
custom_css = """
<style>
body { font-family: -apple-system, sans-serif; background-color: #f5f7fa; }
.accordion-section { 
    background: white; 
    border-radius: 16px; 
    margin-bottom: 20px; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
    overflow: hidden; 
}
.accordion-header { 
    background: #142D4A; 
    color: white; 
    padding: 20px 28px; 
    cursor: pointer; 
    display: flex; 
    justify-content: space-between; 
    border: none; 
    width: 100%; 
    text-align: left; 
    font-size: 16px; 
    font-weight: 600; 
    border-radius: 16px 16px 0 0;
}
.accordion-header:hover { background-color: #0d1f33; }
.accordion-content { 
    padding: 24px 28px; 
    background: #ffffff;
    border-radius: 0 0 16px 16px;
}
.input-item { margin-bottom: 20px; }
.irs--shiny .irs-bar { background: #142D4A; }
.irs--shiny .irs-handle { border: 2px solid #142D4A; background: white; }
.irs--shiny .irs-from, .irs--shiny .irs-to, .irs--shiny .irs-single { background: #142D4A; }
#predict:hover { background: #b91f1f !important; transform: translateY(-1px); }
#load_defect_sample:hover { background: #a8a6a6 !important; transform: translateY(-1px); box-shadow: 0 4px 8px rgba(194, 192, 192, 0.4) !important; }
.hidden { display: none !important; }
#draggable-prediction { 
    background: white; 
    border-radius: 16px; 
    box-shadow: 0 4px 16px rgba(0,0,0,0.15); 
}
#draggable-prediction .card-header {
    border-radius: 16px 16px 0 0 !important;
}
</style>
<script>
function toggleAccordion(id) { var c = document.getElementById(id); c.style.display = c.style.display === "none" ? "block" : "none"; }
function togglePredictionCard() { document.getElementById('draggable-prediction').classList.toggle('hidden'); }
let isDragging = false, currentX, currentY, initialX, initialY, xOffset = 0, yOffset = 0;
document.addEventListener('DOMContentLoaded', function() {
    const drag = document.getElementById('draggable-prediction');
    if (drag) {
        const header = drag.querySelector('.card-header');
        if (header) {
            header.addEventListener('mousedown', e => { initialX = e.clientX - xOffset; initialY = e.clientY - yOffset; isDragging = true; });
            document.addEventListener('mousemove', e => { if (isDragging) { e.preventDefault(); currentX = e.clientX - initialX; currentY = e.clientY - initialY; xOffset = currentX; yOffset = currentY; drag.style.transform = `translate(${currentX}px, ${currentY}px)`; }});
            document.addEventListener('mouseup', () => { initialX = currentX; initialY = currentY; isDragging = false; });
        }
    }
    const btn = document.getElementById('settings-button');
    if (btn) {
        btn.addEventListener('click', togglePredictionCard);
        btn.addEventListener('mouseenter', function() { this.style.transform = 'rotate(90deg)'; });
        btn.addEventListener('mouseleave', function() { this.style.transform = 'rotate(0deg)'; });
    }
});
</script>
"""

# 메타데이터 생성
def create_input_metadata():
    metadata = {}
    for col in categorical_cols:
        vals = sorted([str(v) for v in df[col].dropna().unique()])
        metadata[col] = {'type': 'categorical', 'choices': vals, 'default': vals[0] if vals else ""}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0: continue
        vmin, vmax, vdef = float(s.min()), float(s.max()), float(s.median())
        if vmin == vmax: vmin -= 1.0; vmax += 1.0
        step = max(1, round((vmax - vmin) / 200.0)) if (s.round() == s).all() else (vmax - vmin) / 200.0
        metadata[col] = {'type': 'numeric', 'min': vmin, 'max': vmax, 'value': vdef, 'step': step}
    return metadata

input_metadata = create_input_metadata()

def create_widgets(cols, is_categorical=False):
    widgets = []
    for col in cols:
        if col not in input_metadata: continue
        meta = input_metadata[col]
        label = COLUMN_NAMES_KR.get(col, col)
        if is_categorical:
            widgets.append(ui.div(ui.input_select(col, label, choices=meta['choices'], selected=meta['default']), class_="input-item"))
        else:
            widgets.append(ui.div(ui.input_slider(col, label, min=meta['min'], max=meta['max'], value=meta['value'], step=meta['step']), class_="input-item"))
    return widgets

def panel():
    cat_widgets = create_widgets(categorical_cols, True)
    num_widgets = create_widgets(numeric_cols, False)
    
    cat_rows = [ui.layout_columns(*cat_widgets[i:i+4], col_widths=[3,3,3,3]) for i in range(0, len(cat_widgets), 4)]
    num_rows = [ui.layout_columns(*num_widgets[i:i+4], col_widths=[3,3,3,3]) for i in range(0, len(num_widgets), 4)]
    
    return ui.nav_panel("개별 예측", ui.page_fluid(
        ui.HTML(custom_css),
        ui.div(ui.div(
            ui.div("예측 결과", class_="card-header", style="background: #142D4A; color: white; padding: 16px 20px; border-radius: 16px 16px 0 0; font-weight: 600; cursor: move;"),
            ui.div(
                ui.div(ui.output_ui("prediction_result"), style="margin-bottom: 15px; text-align: center;"),
                ui.input_action_button("predict", "▶ 예측 실행", style="width: 100%; background: #dc3545; color: white; border: none; border-radius: 10px; padding: 12px 24px; font-weight: 600; margin-bottom: 10px;"),
                ui.input_action_button("load_defect_sample", "▶ 불량 샘플 랜덤 추출", style="width: 100%; background: #C2C0C0; color: #2c3e50; border: none; border-radius: 10px; padding: 12px 24px; font-weight: 600;"),
                style="background: white; padding: 20px; border-radius: 0 0 16px 16px;"
            )
        ), id="draggable-prediction", style="position: fixed; bottom: 20px; right: 100px; width: 320px; z-index: 1000;"),
        ui.HTML('<div id="settings-button" style="position: fixed; bottom: 20px; right: 20px; width: 60px; height: 60px; background: #142D4A; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.2); z-index: 1000; transition: transform 0.2s;"><svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M12 1v6m0 6v6M17 12h6m-6 0H1"></path></svg></div>'),
        ui.div(
            ui.div(ui.tags.button(ui.div(ui.span("변수 설정", style="font-size: 16px;"), ui.span("▼", style="font-size: 12px;"), style="display: flex; justify-content: space-between; width: 100%;"), onclick="toggleAccordion('variables_content')", class_="accordion-header"),
                   ui.div(*cat_rows, *num_rows, id="variables_content", class_="accordion-content", style="display: block;"), class_="accordion-section", style="margin-bottom: 16px;"),
            ui.div(ui.tags.button(ui.div(ui.span("불량 샘플", style="font-size: 16px;"), ui.span("▼", style="font-size: 12px;"), style="display: flex; justify-content: space-between; width: 100%;"), onclick="toggleAccordion('defect_sample_content')", class_="accordion-header"),
                   ui.div(ui.output_ui("defect_sample_table"), id="defect_sample_content", class_="accordion-content", style="display: block;"), class_="accordion-section"),
            style="padding: 24px; max-width: 1400px; margin: 0 auto;"
        )
    ))

def server(input, output, session):
    result_text = reactive.Value("예측 실행 버튼을 눌러 결과를 확인하세요")
    is_predicted = reactive.Value(False)
    show_defect_samples = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.load_defect_sample)
    def _load_samples():
        show_defect_samples.set(True)

    @output
    @render.ui
    def defect_sample_table():
        if not show_defect_samples.get():
            return ui.div("불량 샘플 랜덤 추출 버튼을 눌러주세요", style="text-align: center; color: #6c757d; padding: 20px;")
        
        if 'passorfail' not in df.columns: return ui.div("오류: passorfail 컬럼을 찾을 수 없습니다", style="text-align: center; color: #dc3545; padding: 20px;")
        defect = df[df['passorfail'] == 1.0].copy()
        if len(defect) == 0: return ui.div("불량 샘플이 없습니다", style="text-align: center; color: #6c757d; padding: 20px;")
        
        # 랜덤으로 1개 행만 선택
        random_sample = defect.sample(n=1).iloc[0]
        
        html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 12px;"><thead style="background: #142D4A; color: white;"><tr>'
        for col in defect.columns: html += f'<th style="padding: 10px; text-align: left; border: 1px solid #dee2e6; white-space: nowrap;">{col}</th>'
        html += '</tr></thead><tbody><tr style="background: white;">'
        for col in defect.columns: html += f'<td style="padding: 8px; border: 1px solid #dee2e6; white-space: nowrap;">{"-" if pd.isna(random_sample[col]) else str(random_sample[col])}</td>'
        html += '</tr>'
        html += f'</tbody></table></div><div style="margin-top: 10px; color: #6c757d; font-size: 13px;">랜덤 추출된 불량 샘플 (전체 {len(defect)}개 중)</div>'
        return ui.HTML(html)

    @reactive.effect
    @reactive.event(input.predict)
    def _run_prediction():
        try:
            input_row = {}
            for col in all_cols:
                if col not in input_metadata: continue
                meta = input_metadata[col]
                value = getattr(input, col)()
                if meta['type'] == 'categorical':
                    dtype = df[col].dtype
                    input_row[col] = int(value) if 'int' in str(dtype) else float(value) if 'float' in str(dtype) else value
                else:
                    input_row[col] = float(value)
            
            X = pd.DataFrame([input_row], columns=all_cols)
            X_scaled = std_scaler.transform(X[std_scaler.feature_names_in_])
            pred = rf_model.predict(X_scaled)[0]
            result_text.set("불량" if int(pred) == 1 else "양품")
            is_predicted.set(True)
        except Exception as e:
            result_text.set(f"오류 발생: {str(e)}")
            is_predicted.set(True)

    @output
    @render.ui
    def prediction_result():
        if is_predicted.get():
            text = result_text.get()
            color = "#28a745" if "양품" in text else "#dc3545" if "불량" in text else "#2c3e50"
            return ui.div(text, style=f"font-size: 24px; font-weight: 700; color: {color};")
        return ui.div(result_text.get(), style="font-size: 14px; font-weight: 400; color: #6c757d;")



app = App(panel, server)