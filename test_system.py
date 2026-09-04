"""
전체 시스템 구성 요소 단위 테스트 및 검증 스크립트
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from core.analyzer import FacialAffectAnalyzer
from database.db_handler import AffectDBHandler
from download_model import MODEL_PATH


def test_analyzer():
    print("--> 1. FacialAffectAnalyzer 수식 검증 시작")
    analyzer = FacialAffectAnalyzer(max_rotation_degrees=35.0, blink_threshold=0.45)

    # 모의 Blendshape 카테고리 데이터 생성
    mock_categories = [
        SimpleNamespace(category_name="mouthSmileLeft", score=0.6),
        SimpleNamespace(category_name="mouthSmileRight", score=0.8),
        SimpleNamespace(category_name="mouthFrownLeft", score=0.1),
        SimpleNamespace(category_name="mouthFrownRight", score=0.1),
        SimpleNamespace(category_name="browDownLeft", score=0.0),
        SimpleNamespace(category_name="browDownRight", score=0.0),
        SimpleNamespace(category_name="eyeBlinkLeft", score=0.8),
        SimpleNamespace(category_name="eyeBlinkRight", score=0.8),
    ]

    metrics = analyzer.analyze(mock_categories)
    print(f"    Smile Score: {metrics.smile_score} (예상: 0.70)")
    assert abs(metrics.smile_score - 0.70) < 1e-4, "Smile score 불일치"

    print(f"    Frown Score: {metrics.frown_score} (예상: 0.05)")
    assert abs(metrics.frown_score - 0.05) < 1e-4, "Frown score 불일치"

    expected_flatness = 1.0 - min(1.0, 0.7 * 1.5 + 0.05 * 1.5)  # 1.0 - 1.0 = 0.0
    print(f"    Flatness Score: {metrics.flatness_score} (예상: {expected_flatness})")
    assert abs(metrics.flatness_score - expected_flatness) < 1e-4, "Flatness score 불일치"

    print(f"    Blink Score: {metrics.blink_score} (예상: 0.80), Detected: {metrics.blink_detected}")
    assert metrics.blink_detected is True, "Blink 감지 실패"

    print("    [PASS] Analyzer 수식 계산 검증 완료!")


def test_db_handler():
    print("\n--> 2. AffectDBHandler 기능 검증 시작")
    test_db_path = Path(__file__).parent / "test_affect.db"
    if test_db_path.exists():
        test_db_path.unlink()

    db = AffectDBHandler(db_path=str(test_db_path), batch_interval_sec=1.0, batch_max_size=5)

    # 10개 레코드 추가 (배치 플러시 테스트)
    for i in range(10):
        db.add_record(
            smile_score=0.1 * i,
            frown_score=0.05 * i,
            flatness_score=0.9 - (0.05 * i),
            blink_detected=(i % 3 == 0),
            head_pose_valid=True,
            yaw=1.2, pitch=-0.5, roll=0.1
        )
    db.flush()

    recent_df = db.get_recent_logs(limit_seconds=60)
    print(f"    삽입된 레코드 수: {len(recent_df)}")
    assert len(recent_df) == 10, f"레코드 수 불일치: {len(recent_df)}"

    # 3일 연속 둔마 시뮬레이션 및 스크리닝 알림 테스트
    db.seed_mock_data(days=5, trigger_alert=True)
    alert = db.check_screening_alert(threshold_flatness=0.85, consecutive_days=3)
    print(f"    스크리닝 알림 결과: Triggered={alert['alert_triggered']}, 연속일수={alert['consecutive_count']}")
    assert alert['alert_triggered'] is True, "3일 연속 둔마 알림 감지 실패"

    # Windows 파일 잠금 해제를 위해 가비지 컬렉션
    import gc
    del db
    gc.collect()
    try:
        if test_db_path.exists():
            test_db_path.unlink()
    except Exception:
        pass
    print("    [PASS] DB 로깅 및 집계/알림 검증 완료!")


def test_mediapipe_model():
    print("\n--> 3. MediaPipe Tasks Face Landmarker 로딩 검증 시작")
    assert MODEL_PATH.exists(), "face_landmarker.task 파일이 존재하지 않습니다."
    
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    with open(MODEL_PATH, "rb") as f:
        model_buffer = f.read()

    base_options = python.BaseOptions(model_asset_buffer=model_buffer)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    assert landmarker is not None, "FaceLandmarker 인스턴스 생성 실패"
    print("    [PASS] MediaPipe Tasks Face Landmarker 로딩 검증 완료!")


if __name__ == "__main__":
    test_analyzer()
    test_db_handler()
    test_mediapipe_model()
    print("\n==========================================")
    print(">> [SUCCESS] ALL TESTS PASSED SUCCESSFULLY! <<")
    print("==========================================")
