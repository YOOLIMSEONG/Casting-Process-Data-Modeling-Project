# app.py
from shiny import App, ui
from modules import tab_analysis

# 앱 UI 정의
app_ui = ui.page_navbar(
    tab_analysis.panel(),  # 분석 탭
    title="공정 불량 분류 대시보드"
)

# 서버 정의
def server(input, output, session):
    tab_analysis.server(input, output, session)

# 앱 실행 객체
app = App(app_ui, server)