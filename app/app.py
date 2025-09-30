# app.py
from shiny import App, ui
from modules import tab_analysis_copy, tab_model_performance, tab_process_explanation, tab_preprocessing

# 전역 CSS 스타일
app_css = """
<style>
/* 전체 페이지 배경 */
body {
    background: #383636;
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* 외부 컨테이너 (큰 둥근 사각형) - 검정색 */
.outer-container {
    background: #000000;
    border-radius: 32px;
    padding: 24px;
    margin: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    min-height: calc(100vh - 40px);
}

/* 내부 컨테이너 (작은 둥근 사각형) */
.inner-container {
    background: #ffffff;
    border-radius: 24px;
    padding: 0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    min-height: calc(100vh - 128px);
    overflow: hidden;
    position: relative;
}

/* 네비게이션 스타일 개선 */
.nav-pills {
    background: #2A2D30;
    padding: 20px 16px;
    border-radius: 24px 0 0 24px;
    margin: 0;
    min-height: calc(100vh - 128px);
    transition: all 0.3s ease;
}

.nav-pills.collapsed {
    width: 0 !important;
    padding: 0 !important;
    overflow: hidden;
}

.nav-pills .nav-link {
    color: #ecf0f1;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 8px;
    transition: all 0.3s ease;
    font-weight: 500;
    font-size: 15px;
}

.nav-pills .nav-link:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateX(4px);
}

.nav-pills .nav-link.active {
    background: #4A90E2;
    color: white;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
}

/* 탭 콘텐츠 영역 - 패딩만 조정 */
.tab-content {
    background: white;
    padding: 32px;
    min-height: calc(100vh - 128px);
    transition: all 0.3s ease;
}

/* navset-pill-list 레이아웃 조정 */
.bslib-sidebar-layout {
    border-radius: 24px;
    overflow: hidden;
}

/* 토글 버튼 */
.nav-toggle-btn {
    position: absolute;
    top: 20px;
    left: 20px;
    z-index: 1001;
    width: 40px;
    height: 40px;
    background: #2A2D30;
    border: none;
    border-radius: 8px;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.nav-toggle-btn:hover {
    background: #4A90E2;
    transform: scale(1.05);
}

.nav-toggle-btn svg {
    width: 24px;
    height: 24px;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    let navCollapsed = false;
    
    // 토글 버튼 생성
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'nav-toggle-btn';
    toggleBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>';
    
    const innerContainer = document.querySelector('.inner-container');
    if (innerContainer) {
        innerContainer.appendChild(toggleBtn);
    }
    
    toggleBtn.addEventListener('click', function() {
        const sidebar = document.querySelector('.bslib-sidebar-layout > aside');
        
        if (sidebar) {
            navCollapsed = !navCollapsed;
            
            if (navCollapsed) {
                sidebar.style.display = 'none';
                toggleBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
            } else {
                sidebar.style.display = 'block';
                toggleBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>';
            }
        }
    });
});
</script>
"""

# 앱 UI 정의
app_ui = ui.page_fluid(
    ui.HTML(app_css),
    
    # 외부 컨테이너 (첫 번째 둥근 사각형 - 검정색)
    ui.div(
        # 내부 컨테이너 (두 번째 둥근 사각형 - 흰색)
        ui.div(
            # 헤더 제거하고 바로 네비게이션과 콘텐츠
            ui.navset_pill_list(
                tab_analysis_copy.panel(),
                tab_model_performance.panel(),
                tab_process_explanation.panel(),
                tab_preprocessing.panel(),
                id="main_nav"
            ),
            class_="inner-container"
        ),
        class_="outer-container"
    ),
)

# 서버 정의
def server(input, output, session):
    tab_analysis_copy.server(input, output, session)
    tab_model_performance.server(input, output, session)
    tab_process_explanation.server(input, output, session)
    tab_preprocessing.server(input, output, session)

# 앱 실행 객체
app = App(app_ui, server)