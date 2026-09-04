import sys
import os

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
from core.senior_analyzer import SeniorEmotionAnalyzer
from database.db_handler import SeniorAffectDBHandler

def test_system():
    print("[1/3] 시니어 감정 분석기 초기화 중...")
    analyzer = SeniorEmotionAnalyzer()
    print(f" - 한글 폰트 경로: {analyzer.font_path}")

    print("[2/3] 시니어 HUD 오버레이 렌더링 테스트 중...")
    dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    dummy_analysis = {
        'dominant': 'happy',
        'dominant_ko': '활짝 웃음 (기쁨)',
        'emoji': '😊',
        'confidence': 91.2,
        'scores': {'happy': 91.2, 'neutral': 5.2, 'sad': 1.1, 'angry': 0.5, 'surprise': 1.0, 'fear': 0.5, 'disgust': 0.5},
        'care_message': '어르신의 활짝 웃으시는 모습이 참 보기 좋습니다! ☀️',
        'region': {'x': 300, 'y': 150, 'w': 360, 'h': 300},
        'weather': '맑음 ☀️',
        'color_rgb': (46, 204, 113),
        'color_bgr': (113, 204, 46)
    }

    out_frame = analyzer.render_senior_hud(dummy_frame, dummy_analysis, fps=30.0)
    assert out_frame.shape == (540, 960, 3), "렌더링된 프레임 크기 불일치"
    print(" - HUD 렌더링 프레임 크기:", out_frame.shape)

    print("[3/3] SQLite DB 핸들러 및 날씨 통계 테스트 중...")
    db = SeniorAffectDBHandler()
    db.add_record('happy', '활짝 웃음 (기쁨)', 91.2, dummy_analysis['scores'], dummy_analysis['care_message'], force_flush=True)
    summary = db.get_daily_weather_summary()
    print(f" - 오늘 마음 날씨: {summary['weather']}")
    print(f" - 마음 상태: {summary['dominant_state']}")
    print(f" - 총 기록 건수: {summary['total_records']}")
    print(f" - 웃음 횟수: {summary['smile_count']}")

    print("\n[SUCCESS] 시니어 감정 분석 시스템 전체 검증 성공!")

if __name__ == "__main__":
    test_system()
