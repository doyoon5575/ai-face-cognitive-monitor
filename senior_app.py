"""
AI 다중 생체신호(안면·음성) 컨디션 & 인지 건강 모니터링 시스템 (senior_app.py)
- '시니어' 명칭 삭제 및 아동, 성인, 시니어, 장애인, 재활 환자 등 누구나 사용 가능한 범용 포용적 디자인
- 좌측: 실시간 웹캠 안면 랜드마크 선(Face Mesh), 입술/눈가 미세 바이오마커, 마이크 음성 활력 게이지 (LIVE)
- 우측: 실시간 두뇌 인지 & 컨디션 자가진단 퀴즈 (LIVE 동시 진행 일체형 듀얼 스크린)
- 융합 리포트: 퀴즈 해결력 + 얼굴 미세 피로도 + 목소리 톤을 결합한 다차원 컨디션 성적표
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

# UTF-8 콘솔 인코딩 보장
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 프로젝트 루트 경로 등록
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from database.db_handler import SeniorAffectDBHandler
from core.multimodal_analyzer import MultimodalAffectAnalyzer
from core.assessment import CognitiveConditionAssessment

# ───── Streamlit 페이지 설정 ─────
st.set_page_config(
    page_title="AI 다중 생체신호 컨디션 & 인지 모니터",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───── 세련된 범용 모던 CSS ─────
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Pretendard', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    }
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 22px 28px; border-radius: 16px; color: white;
        margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    .app-header h1 { font-size: 2.1rem !important; font-weight: 800; margin: 0; color: #fff; }
    .app-header p { font-size: 1.1rem; margin-top: 6px; color: #cbd5e1; line-height: 1.5; }
    .status-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.04);
    }
    .quiz-container {
        background: #f8fafc; border: 2px solid #e2e8f0;
        border-radius: 16px; padding: 22px; height: 100%;
    }
    .quiz-q-title {
        font-size: 1.18rem; font-weight: 700; color: #0f172a; margin-bottom: 10px;
    }
    .quiz-badge {
        display: inline-block; background: #e0f2fe; color: #0369a1;
        padding: 3px 9px; border-radius: 6px; font-size: 0.85rem; font-weight: 700;
        margin-bottom: 6px;
    }
    .live-badge {
        background: #ef4444; color: white; padding: 4px 10px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 800; display: inline-block; animation: pulse 2s infinite;
    }
    .metric-val-large {
        font-size: 1.8rem; font-weight: 800; color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


# ───── 컴포넌트 로딩 (캐싱) ─────
@st.cache_resource
def load_db():
    return SeniorAffectDBHandler()

@st.cache_resource
def load_multimodal_analyzer():
    return MultimodalAffectAnalyzer()


def main():
    db = load_db()

    # ───── 상단 대형 헤더 ─────
    st.markdown("""
    <div class="app-header">
        <h1>🧠 AI 다중 생체신호(안면·음성) 컨디션 & 인지 모니터</h1>
        <p>웹캠과 마이크를 통해 <b>얼굴 랜드마크 선, 입술·눈가 미세 표정, 목소리 활력도</b>를 실시간 분석하며, <b>두뇌 인지 퀴즈</b>를 동시에 진행하는 통합 헬스케어 시스템입니다.</p>
    </div>
    """, unsafe_allow_html=True)

    # ───── 사이드바: 장치 및 환경 설정 ─────
    with st.sidebar:
        st.markdown("### ⚙️ 장치 설정")
        camera_id = st.selectbox(
            "카메라 선택",
            options=[1, 0],
            format_func=lambda x: f"카메라 {x}번 {'(노트북 실제 웹캠 - 추천 ✅)' if x == 1 else '(가상/보조 장치 ❌)'}"
        )
        skip_rate = st.slider("안면 분석 주기 (프레임 스킵)", 2, 8, 4)

        st.markdown("---")
        st.markdown("### 🎙️ 마이크 음성 상태")
        st.caption("• 노트북 내장 마이크를 통해 발화 성량(RMS)과 목소리 톤을 자동 감지합니다.")
        st.caption("• 문제를 풀면서 편안하게 소리 내어 답변해 보세요.")

        st.markdown("---")
        st.markdown("### 🧪 시연 데이터")
        if st.button("📊 모의 데이터 생성 (5일치)"):
            with st.spinner("모의 데이터를 생성 중입니다..."):
                db.populate_mock_data(days=5)
                st.success("데이터 생성이 완료되었습니다!")
                st.rerun()

    # ───── 핵심: 좌우 동시 진행 일체형 듀얼 스크린 ─────
    col_left, col_right = st.columns([1.2, 1.0], gap="medium")

    # [좌측 영역]: 실시간 카메라 & 음성 피드 (LIVE)
    with col_left:
        st.markdown("### 🎥 다중 생체신호 거울 (LIVE)")

        tab_browser, tab_local = st.tabs([
            "📱 모바일/웹 카메라 (외부·클라우드 추천)",
            "💻 로컬 PC 웹캠 (내 컴퓨터 전용)"
        ])

        # ── 1. 스마트폰/웹 브라우저 카메라 모드 (클라우드 상시 접속 최적화) ──
        with tab_browser:
            st.markdown("""
            <div style="background: #f1f5f9; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; font-size: 0.92rem; color: #334155;">
                ✨ <b>스마트폰, 태블릿, 외부 브라우저</b>로 접속하셨을 때 사용합니다.<br>
                카메라로 얼굴을 촬영하시면 <b>안면 랜드마크 선, 입술·눈가 바이오마커, 표정</b>이 즉시 정밀 분석되어 우측 퀴즈와 동기화됩니다.
            </div>
            """, unsafe_allow_html=True)

            camera_img = st.camera_input("📷 내 얼굴 촬영하여 즉시 진단", key="cloud_browser_camera")

            with st.expander("📁 또는 기존 얼굴 사진 파일 업로드"):
                uploaded_file = st.file_uploader("얼굴 사진 선택 (JPG, PNG)", type=["jpg", "jpeg", "png"], key="upload_face")

            target_img_bytes = None
            if camera_img is not None:
                target_img_bytes = camera_img.getvalue()
            elif uploaded_file is not None:
                target_img_bytes = uploaded_file.getvalue()

            if target_img_bytes:
                with st.spinner("AI가 안면 랜드마크 및 다중 생체신호를 분석 중입니다..."):
                    analyzer = load_multimodal_analyzer()
                    np_arr = np.frombuffer(target_img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # 좌우 반전 거울 효과
                        frame = cv2.flip(frame, 1)
                        analysis = analyzer.analyze_frame(frame)
                        if analysis:
                            st.session_state['latest_analysis'] = analysis
                            db.add_record(
                                dominant_emotion=analysis['dominant'],
                                emotion_ko=analysis['dominant_ko'],
                                confidence=analysis['confidence'],
                                scores=analysis['scores'],
                                care_message=analysis['care_message']
                            )
                            db.flush()

                        rendered = analyzer.render_hud(frame, analysis)
                        rgb_frame = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                        st.image(rgb_frame, use_container_width=True, caption="AI 안면 랜드마크 & 생체 바이오마커 분석 결과")

                        if analysis:
                            st.success(f"✅ 분석 완료: **{analysis['dominant_ko']}** (일치도 {analysis['confidence']:.0f}%) | 눈가 피로도 {analysis.get('eye_fatigue', 20):.0f}% | 안면 대칭도 {analysis.get('symmetry', 95):.0f}%")
                            st.info(f"💡 {analysis.get('care_message', '')}")
            else:
                st.markdown("""
                <div style="
                    background-color: #0f172a; border: 2px dashed #334155;
                    border-radius: 14px; min-height: 240px;
                    display: flex; flex-direction: column; justify-content: center;
                    align-items: center; color: #94a3b8; text-align: center; padding: 24px;">
                    <div style="font-size: 3.2rem; margin-bottom: 10px;">🤳</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">스마트폰/웹 카메라 준비 완료</div>
                    <div style="font-size: 0.95rem; color: #94a3b8; line-height: 1.5;">
                        위의 <b>[사진 촬영]</b> 버튼을 누르시면<br>
                        접속 중인 스마트폰/PC의 카메라로 즉시 생체신호가 측정됩니다.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── 2. 로컬 PC 웹캠 실시간 연속 스트리밍 모드 ──
        with tab_local:
            st.caption("※ 내 노트북에서 직접 실행할 때 30fps 연속 실시간 스트리밍을 제공합니다.")
            camera_on = st.toggle("로컬 웹캠 실시간 연속 스트리밍 시작", value=False, key="local_cam_toggle",
                                  help="스위치를 켜면 웹캠의 안면 랜드마크 선과 입술·눈가 미세 표정, 음성 활력도가 실시간 측정됩니다.")

            if camera_on:
                status_msg = st.empty()
                status_msg.info("🔄 카메라 및 오디오 스트림에 연결하는 중...")

                cap = None
                for backend_name, backend in [("DirectShow", cv2.CAP_DSHOW), ("기본", cv2.CAP_ANY)]:
                    cap = cv2.VideoCapture(camera_id, backend)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            status_msg.success(f"✅ 카메라 {camera_id}번 & 마이크 연결 성공! 다중 분석을 시작합니다.")
                            break
                        else:
                            cap.release()
                            cap = None

                if cap is None or not cap.isOpened():
                    status_msg.error(f"❌ 카메라 {camera_id}번에 연결할 수 없습니다. 사이드바에서 다른 번호를 선택해 보세요.")
                    st.info("💡 클라우드 서버에는 물리 웹캠이 연결되어 있지 않습니다. 상단의 [📱 모바일/웹 카메라] 탭을 이용해 주세요!")
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    analyzer = load_multimodal_analyzer()
                    frame_placeholder = st.empty()
                    frame_idx = 0
                    last_analysis = None

                    try:
                        while camera_on:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                time.sleep(0.04)
                                continue

                            # 거울 모드 좌우 반전
                            frame = cv2.flip(frame, 1)
                            frame_idx += 1

                            if frame_idx % skip_rate == 0 or last_analysis is None:
                                analysis = analyzer.analyze_frame(frame)
                                if analysis:
                                    last_analysis = analysis
                                    # 세션 상태에 최신 미세 지표 저장 (우측 퀴즈와 실시간 동기화)
                                    st.session_state['latest_analysis'] = analysis
                                    db.add_record(
                                        dominant_emotion=analysis['dominant'],
                                        emotion_ko=analysis['dominant_ko'],
                                        confidence=analysis['confidence'],
                                        scores=analysis['scores'],
                                        care_message=analysis['care_message']
                                    )

                            rendered = analyzer.render_hud(frame, last_analysis)
                            rgb_frame = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb_frame)

                            time.sleep(0.04)

                    except Exception as e:
                        st.error(f"스트리밍 오류 발생: {e}")
                    finally:
                        cap.release()
                        db.flush()
                        status_msg.info("📷 장치가 안전하게 종료되었습니다.")
            else:
                st.markdown("""
                <div style="
                    background-color: #0f172a; border: 3px dashed #334155;
                    border-radius: 16px; min-height: 280px;
                    display: flex; flex-direction: column; justify-content: center;
                    align-items: center; color: #94a3b8; text-align: center; padding: 24px;">
                    <div style="font-size: 3.5rem; margin-bottom: 12px;">🪞</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin-bottom: 8px;">로컬 웹캠 대기 중</div>
                    <div style="font-size: 0.95rem; color: #94a3b8; line-height: 1.6;">
                        위의 <b>[로컬 웹캠 실시간 연속 스트리밍 시작]</b> 스위치를 켜시면<br>
                        노트북 웹캠을 통한 실시간 연속 분석이 시작됩니다.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # [우측 영역]: 실시간 두뇌 인지 & 컨디션 퀴즈 (동시 진행)
    with col_right:
        st.markdown("### 🧠 두뇌 인지 & 컨디션 퀴즈 (동시 진행)")

        # 현재 카메라 관찰 상태 실시간 배너
        latest_ana = st.session_state.get('latest_analysis', None)
        if latest_ana:
            dom_text = latest_ana.get('dominant_ko', '평온')
            mouth_d = latest_ana.get('mouth_dynamic', 30.0)
            eye_f = latest_ana.get('eye_fatigue', 20.0)
            audio_m = latest_ana.get('audio', {})
            mic_state = audio_m.get('tone_state', '대기')

            st.markdown(f"""
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.95rem; color: #1e40af;">
                🔴 <b>실시간 생체 관찰 중</b>: [표정: <b>{dom_text}</b>] | [입술 반응도: <b>{mouth_d:.0f}%</b>] | [눈가 피로도: <b>{eye_f:.0f}%</b>] | [목소리: <b>{mic_state}</b>]
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.95rem; color: #64748b;">
                💡 좌측의 <b>[실시간 모니터링 시작]</b>을 켜시면 문제를 푸는 동안 표정과 목소리가 함께 분석됩니다.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-bottom: 12px; font-size: 1.0rem; color: #334155;">
            문제를 편안하게 읽으시며 목소리로도 소리 내어 말씀해 보세요. 
        </div>
        """, unsafe_allow_html=True)

        user_answers = {}

        # 5문항 인터랙티브 렌더링
        for q in CognitiveConditionAssessment.QUESTIONS:
            qid = q["id"]
            st.markdown(f"""
            <div class="quiz-container" style="margin-bottom: 14px; padding: 16px;">
                <div class="quiz-badge">{q['category']}</div>
                <div class="quiz-q-title">{qid}. {q['question']}</div>
            </div>
            """, unsafe_allow_html=True)

            choice = st.radio(
                f"정답 선택 ({qid}번)",
                options=q["options"],
                index=None,
                key=f"sync_quiz_{qid}",
                label_visibility="collapsed"
            )
            user_answers[qid] = choice

        # 평가 제출 버튼
        if st.button("📊 [두뇌 인지 & 컨디션 종합 진단 결과 보기]", type="primary", use_container_width=True):
            unanswered = [qid for qid, ans in user_answers.items() if not ans]
            if unanswered:
                st.warning(f"아직 선택하지 않은 문항이 있습니다 (문항 번호: {unanswered}). 모든 문항을 선택해 주세요.")
            else:
                recent_dom = latest_ana.get('dominant', 'neutral') if latest_ana else 'neutral'
                recent_eye_f = latest_ana.get('eye_fatigue', 20.0) if latest_ana else 20.0
                recent_voice_v = latest_ana.get('audio', {}).get('vitality', 60.0) if latest_ana else 60.0

                result = CognitiveConditionAssessment.evaluate(
                    user_answers=user_answers,
                    dominant_emotion=recent_dom,
                    eye_fatigue=recent_eye_f,
                    voice_vitality=recent_voice_v
                )

                st.markdown(f"""
                <div style="background-color: {result['color']}; border-radius: 16px; padding: 22px; color: white; margin-top: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                    <div style="font-size: 1.15rem; font-weight: 600; opacity: 0.95;">{result['icon']} 종합 판정: {result['level']}</div>
                    <div style="font-size: 2.1rem; font-weight: 800; margin: 6px 0;">인지 퀴즈 점수: {result['score']}점 / 100점</div>
                    <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 6px;">{result['title']}</div>
                    <div style="font-size: 1.05rem; line-height: 1.55; opacity: 0.95;">{result['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📋 맞춤형 권장 생활 가이드")
                st.info(result["action_guide"])

                with st.expander("🔍 문항별 정답 및 상세 해설 보기 (클릭)"):
                    for d in result["details"]:
                        mark = "✅ 정답" if d["is_correct"] else "❌ 오답"
                        st.write(f"**문항 {d['id']} [{d['category']}]**: {d['question']}")
                        st.write(f"- 내 선택: **{d['selected']}** ({mark})  |  정답: **{d['answer']}** (+{d['points']}점)")
                        st.divider()

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # ───── 하단 통계 섹션: 오늘의 컨디션 요약 및 변화 추이 ─────
    st.subheader("📈 실시간 컨디션 추이 & 데이터 분석")
    t1, t2 = st.tabs(["실시간 표정 흐름", "주간 통계 요약"])

    df_recent = db.get_recent_logs(limit=60)

    with t1:
        if not df_recent.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_recent['timestamp'], y=df_recent['happy_score'], mode='lines+markers', name='기쁨/활력', line=dict(color='#2ecc71', width=3)))
            fig.add_trace(go.Scatter(x=df_recent['timestamp'], y=df_recent['neutral_score'], mode='lines', name='평온/안정', line=dict(color='#3498db', width=2)))
            fig.add_trace(go.Scatter(x=df_recent['timestamp'], y=df_recent['sad_score'], mode='lines', name='울적/피로', line=dict(color='#9b59b6', width=2)))
            fig.add_trace(go.Scatter(x=df_recent['timestamp'], y=df_recent['angry_score'], mode='lines', name='답답/긴장', line=dict(color='#e74c3c', width=2)))
            fig.update_layout(height=320, yaxis=dict(range=[0, 100]), template="plotly_white",
                              margin=dict(l=20, r=20, t=20, b=20),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("수집된 관찰 데이터가 없습니다. 카메라를 켜거나 사이드바에서 [모의 데이터 생성]을 눌러보세요.")

    with t2:
        df_weekly = db.get_weekly_stats(days=7)
        if not df_weekly.empty:
            fig_w = px.bar(df_weekly, x='log_date', y='smile_moments',
                           title="일별 활짝 웃으신 순간 횟수", labels={'log_date': '날짜', 'smile_moments': '웃음 횟수'},
                           color='smile_moments', color_continuous_scale='Greens')
            fig_w.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("주간 데이터가 없습니다. 사이드바에서 [모의 데이터 생성]을 눌러보세요.")


if __name__ == "__main__":
    main()
