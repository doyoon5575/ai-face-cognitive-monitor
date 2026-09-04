"""
Streamlit 기반 실시간 안면 정서 및 바이오마커 대시보드
- 실시간 타임라인 차트 (Smile, Frown, Flatness)
- 일별/주별 요약 통계 (Average Flatness, Emotional Variability, Smile Peak Rate)
- 3일 연속 정서 둔마(Flatness > 0.85) 스크리닝 알림 배너
- 모의 데이터 시뮬레이션 및 CSV 내보내기 기능
"""

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database.db_handler import AffectDBHandler

# 페이지 기본 설정
st.set_page_config(
    page_title="Facial Affect & Emotional Biomarker Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 다크 모던 테마 및 커스텀 CSS
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .dashboard-header {
        background: linear-gradient(135deg, #1f242d 0%, #161b22 100%);
        padding: 24px 30px;
        border-radius: 16px;
        border: 1px solid #30363d;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .dashboard-title {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .dashboard-subtitle {
        font-size: 14px;
        color: #8b949e;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 4px;
    }
    .metric-desc {
        font-size: 12px;
        color: #7ee787;
        margin-top: 2px;
    }

    /* Alert Banner */
    .alert-banner-danger {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.15) 0%, rgba(218, 54, 51, 0.25) 100%);
        border: 1px solid #f85149;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
        color: #ff7b72;
    }
    .alert-banner-safe {
        background: linear-gradient(135deg, rgba(46, 160, 67, 0.15) 0%, rgba(35, 134, 54, 0.25) 100%);
        border: 1px solid #3fb950;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 24px;
        color: #7ee787;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_handler():
    return AffectDBHandler()


db = get_db_handler()

# ----------------- 사이드바 설정 -----------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/docs/images/mediapipe_logo.png", width=180)
    st.markdown("### ⚙️ 시스템 제어 & 설정")
    
    auto_refresh = st.checkbox("실시간 자동 새로고침 (5초)", value=False)
    if auto_refresh:
        st.caption("⏱️ 5초 주기로 최신 데이터를 갱신합니다.")
        st.markdown("<script>setTimeout(function(){ window.location.reload(); }, 5000);</script>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 데이터 시뮬레이션")
    st.caption("웹캠 미사용 시 대시보드 기능을 테스트하기 위해 모의 데이터를 생성할 수 있습니다.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("정상 데이터 생성", use_container_width=True):
            db.seed_mock_data(days=7, trigger_alert=False)
            st.success("7일치 정상 데이터 적재 완료!")
            st.rerun()
    with col_s2:
        if st.button("⚠️ 3일 둔마 데이터", use_container_width=True):
            db.seed_mock_data(days=7, trigger_alert=True)
            st.warning("3일 연속 정서둔마 데이터 적재 완료!")
            st.rerun()

    if st.button("🗑️ DB 로그 초기화", use_container_width=True):
        with db._get_connection() as conn:
            conn.cursor().execute("DELETE FROM affect_logs")
            conn.commit()
        st.info("데이터베이스가 초기화되었습니다.")
        st.rerun()

    st.markdown("---")
    st.markdown("### 📋 정보")
    st.markdown("""
    - **Engine**: MediaPipe Tasks Vision (0.10.x+)
    - **Features**: 52 Facial Blendshapes
    - **Metrics**: Smile, Frown, Flatness, Blink, 3D Pose
    """)

# ----------------- 대시보드 헤더 -----------------
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">
        <span>🧠 실시간 안면 정서 및 바이오마커 모니터링 시스템</span>
    </div>
    <div class="dashboard-subtitle">
        Google MediaPipe Tasks 52 Facial Blendshapes 기반 실시간 미세 표정 및 정서 둔마(Flatness) 분석 대시보드
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- 정서 둔마 위험 스크리닝 알림 -----------------
screening_result = db.check_screening_alert(threshold_flatness=0.85, consecutive_days=3)

if screening_result["alert_triggered"]:
    st.markdown(f"""
    <div class="alert-banner-danger">
        <h4 style="margin:0 0 6px 0; color:#ff7b72;">🚨 [위험 알림] 3일 연속 정서 둔마(Emotional Flatness) 고위험 감지</h4>
        <div>{screening_result["message"]}</div>
        <div style="font-size:12px; margin-top:8px; opacity:0.9;">
            최근 3일 감지 일자 및 둔마도: {", ".join([f"<b>{d}</b> ({v:.2f})" for d, v in screening_result["dates_triggered"]])}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-banner-safe">
        <span style="font-weight:600;">✅ 정서 지표 정상 상태:</span> {screening_result["message"]}
    </div>
    """, unsafe_allow_html=True)

# ----------------- 핵심 KPI 메트릭 카드 -----------------
daily_df = db.get_daily_summary(days=7)
recent_df = db.get_recent_logs(limit_seconds=120)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

avg_flatness_7d = daily_df["avg_flatness"].mean() if not daily_df.empty else 0.0
emotional_var_7d = daily_df["emotional_variability"].mean() if not daily_df.empty else 0.0
smile_peak_pct = daily_df["smile_peak_pct"].mean() if not daily_df.empty else 0.0
total_blinks = daily_df["total_blinks"].sum() if not daily_df.empty else 0

with kpi1:
    flatness_color = "#f85149" if avg_flatness_7d > 0.80 else "#7ee787"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">평균 정서 둔마도 (Flatness)</div>
        <div class="metric-value" style="color: {flatness_color};">{avg_flatness_7d:.2f}</div>
        <div class="metric-desc">기준: 0.85 이상 주의 (최근 7일)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">정서 가변성 (Variability)</div>
        <div class="metric-value">{emotional_var_7d:.3f}</div>
        <div class="metric-desc">감정 표현의 표준편차 (Std Dev)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">스마일 피크 비율 (Smile Rate)</div>
        <div class="metric-value" style="color: #58a6ff;">{smile_peak_pct:.1f}%</div>
        <div class="metric-desc">미소 표현(Smile >= 0.40) 빈도</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">총 눈 깜빡임 (Total Blinks)</div>
        <div class="metric-value" style="color: #d2a8ff;">{int(total_blinks):,}회</div>
        <div class="metric-desc">누적 감지 횟수</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- 탭 인터페이스 -----------------
tab1, tab2, tab3 = st.tabs(["📈 실시간 스트림 분석", "📊 주간/일별 정서 추이", "🗄️ 원시 데이터 & 내보내기"])

# TAB 1: 실시간 스트림 분석
with tab1:
    st.subheader("🎥 실시간 웹캠 분석 & 제어")

    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        if not st.session_state.cam_running:
            if st.button("▶️ 카메라 켜기 (Start)", type="primary", use_container_width=True):
                st.session_state.cam_running = True
                st.rerun()
        else:
            if st.button("⏹️ 카메라 끄기 (Stop)", type="secondary", use_container_width=True):
                st.session_state.cam_running = False
                st.rerun()

    with col_ctrl2:
        selected_cam_id = st.selectbox("카메라 장치 선택", options=[0, 1, 2], index=0, format_func=lambda x: f"카메라 ID: {x}")

    with col_ctrl3:
        if st.session_state.cam_running:
            st.success("🟢 [LIVE] 카메라가 활성화되어 실시간 표정을 분석 중입니다.")
        else:
            st.info("⚪ [OFF] 카메라가 꺼져 있습니다. '▶️ 카메라 켜기' 버튼을 눌러주세요.")

    st.markdown("---")

    col_cam1, col_cam2 = st.columns([1.2, 1])

    with col_cam1:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        if st.session_state.cam_running:
            import cv2
            import mediapipe as mp
            import time
            from core.tracker import WebcamFaceTracker

            tracker = WebcamFaceTracker(camera_id=selected_cam_id, target_fps=15, db_handler=db, visualize=False)
            
            # 카메라 장치 열기
            cap = cv2.VideoCapture(selected_cam_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(selected_cam_id)

            if not cap.isOpened():
                st.session_state.cam_running = False
                st.error(f"카메라 ID {selected_cam_id}를 열 수 없습니다. 카메라 권한 또는 연결 상태를 확인해주세요.")
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                start_t = time.time()
                prev_t = time.time()
                fps = 15.0

                try:
                    # 실시간 프레임 루프
                    while st.session_state.cam_running and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.05)
                            continue

                        frame = cv2.flip(frame, 1)
                        h, w, _ = frame.shape

                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        timestamp_ms = int((time.time() - start_t) * 1000)

                        detection_result = tracker._landmarker.detect_for_video(mp_image, timestamp_ms)
                        metrics = None
                        if detection_result and detection_result.face_blendshapes:
                            blendshapes = detection_result.face_blendshapes[0]
                            landmarks = detection_result.face_landmarks[0] if detection_result.face_landmarks else None
                            metrics = tracker.analyzer.analyze(blendshapes, landmarks, w, h)

                            db.add_record(
                                smile_score=metrics.smile_score,
                                frown_score=metrics.frown_score,
                                flatness_score=metrics.flatness_score,
                                blink_detected=metrics.blink_detected,
                                head_pose_valid=metrics.head_pose.is_valid,
                                yaw=metrics.head_pose.yaw,
                                pitch=metrics.head_pose.pitch,
                                roll=metrics.head_pose.roll
                            )

                        cur_t = time.time()
                        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, cur_t - prev_t))
                        prev_t = cur_t

                        hud_frame = tracker.draw_hud(frame, metrics, fps)
                        hud_rgb = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(hud_rgb, channels="RGB", use_container_width=True)
                        time.sleep(0.03)
                finally:
                    cap.release()
                    db.flush()
        else:
            frame_placeholder.markdown("""
            <div style="background-color:#161b22; border:2px dashed #30363d; border-radius:12px; height:360px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#8b949e;">
                <div style="font-size:48px; margin-bottom:12px;">📷</div>
                <div style="font-size:16px; font-weight:600; color:#e6edf3;">카메라가 비활성화되어 있습니다</div>
                <div style="font-size:13px; margin-top:4px;">상단의 <b>[▶️ 카메라 켜기]</b> 버튼을 누르면 실시간 안면 인식이 시작됩니다.</div>
            </div>
            """, unsafe_allow_html=True)

    with col_cam2:
        st.markdown("#### ⏱️ 최근 2분간 감정 변화 타임라인")
        if recent_df.empty:
            st.info("💡 실시간 데이터 수집 대기 중입니다.")
        else:
            fig_recent = go.Figure()
            fig_recent.add_trace(go.Scatter(
                x=recent_df["timestamp"], y=recent_df["smile_score"],
                mode="lines", name="Smile", line=dict(color="#3fb950", width=2.5)
            ))
            fig_recent.add_trace(go.Scatter(
                x=recent_df["timestamp"], y=recent_df["frown_score"],
                mode="lines", name="Frown", line=dict(color="#d29922", width=2)
            ))
            fig_recent.add_trace(go.Scatter(
                x=recent_df["timestamp"], y=recent_df["flatness_score"],
                mode="lines", name="Flatness", line=dict(color="#f85149", width=2, dash="dot")
            ))
            fig_recent.add_hline(y=0.85, line_dash="dash", line_color="rgba(248, 81, 73, 0.6)")
            fig_recent.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"), height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(range=[-0.05, 1.05], gridcolor="#21262d"),
                xaxis=dict(gridcolor="#21262d")
            )
            st.plotly_chart(fig_recent, use_container_width=True)

    # 헤드 포즈 각도 변화 차트
    if not recent_df.empty:
        st.markdown("#### 👤 헤드 포즈 회전 각도 모니터링 (Yaw, Pitch, Roll)")
        fig_pose = go.Figure()
        fig_pose.add_trace(go.Scatter(x=recent_df["timestamp"], y=recent_df["yaw"], mode="lines", name="Yaw (좌우)", line=dict(color="#58a6ff")))
        fig_pose.add_trace(go.Scatter(x=recent_df["timestamp"], y=recent_df["pitch"], mode="lines", name="Pitch (상하)", line=dict(color="#bc8cff")))
        fig_pose.add_trace(go.Scatter(x=recent_df["timestamp"], y=recent_df["roll"], mode="lines", name="Roll (기울임)", line=dict(color="#39c5bb")))
        fig_pose.add_hline(y=35, line_dash="dot", line_color="red")
        fig_pose.add_hline(y=-35, line_dash="dot", line_color="red")
        
        fig_pose.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title="각도 (도)", range=[-45, 45]),
            height=240,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_pose, use_container_width=True)

# TAB 2: 주간/일별 정서 추이
with tab2:
    st.subheader("📅 최근 7일간 일별 정서 지표 및 가변성 통계")
    
    if daily_df.empty:
        st.info("💡 집계된 일별 데이터가 없습니다.")
    else:
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            # 일평균 Flatness vs Smile
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Bar(
                x=daily_df["date"], y=daily_df["avg_flatness"],
                name="일평균 Flatness", marker_color="#f85149"
            ))
            fig_daily.add_trace(go.Bar(
                x=daily_df["date"], y=daily_df["avg_smile"],
                name="일평균 Smile", marker_color="#3fb950"
            ))
            fig_daily.update_layout(
                title="일별 정서 둔마(Flatness) vs 미소(Smile) 평균",
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                barmode="group",
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d", range=[0, 1.0]),
                height=340
            )
            st.plotly_chart(fig_daily, use_container_width=True)

        with col_c2:
            # 감정 가변성 (Emotional Variability)
            fig_var = go.Figure()
            fig_var.add_trace(go.Scatter(
                x=daily_df["date"], y=daily_df["emotional_variability"],
                mode="lines+markers", name="감정 가변성 (StdDev)",
                line=dict(color="#d2a8ff", width=3),
                marker=dict(size=8)
            ))
            fig_var.update_layout(
                title="일별 감정 가변성(Emotional Variability) 추이",
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d", range=[0, 0.4]),
                height=340
            )
            st.plotly_chart(fig_var, use_container_width=True)

        st.markdown("#### 📑 일별 상세 통계 테이블")
        st.dataframe(
            daily_df[[
                "date", "total_frames", "avg_flatness", "avg_smile", 
                "avg_frown", "emotional_variability", "smile_peak_pct", "total_blinks"
            ]].rename(columns={
                "date": "날짜",
                "total_frames": "프레임 수",
                "avg_flatness": "평균 둔마도",
                "avg_smile": "평균 웃음",
                "avg_frown": "평균 찡그림",
                "emotional_variability": "감정 가변성",
                "smile_peak_pct": "스마일 피크 비율(%)",
                "total_blinks": "눈 깜빡임 총수"
            }),
            use_container_width=True
        )

# TAB 3: 원시 데이터 & 내보내기
with tab3:
    st.subheader("🗄️ SQLite 데이터베이스 로그 조회 및 CSV 내보내기")
    
    with db._get_connection() as conn:
        all_df = pd.read_sql_query("SELECT * FROM affect_logs ORDER BY timestamp DESC LIMIT 500", conn)
    
    st.write(f"최근 데이터 {len(all_df)}건 표시:")
    st.dataframe(all_df, use_container_width=True)
    
    if not all_df.empty:
        csv_data = all_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 CSV 파일로 다운로드",
            data=csv_data,
            file_name=f"affect_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
