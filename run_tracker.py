"""
실시간 안면 표정 및 정서 바이오마커 웹캠 트래커 실행 스크립트
사용법:
    python run_tracker.py
    python run_tracker.py --camera-id 0 --fps 15
    python run_tracker.py --no-gui  (헤드리스 백그라운드 모드)
"""

import argparse
import sys
from pathlib import Path

from download_model import download_face_landmarker_model
from core.tracker import WebcamFaceTracker
from database.db_handler import AffectDBHandler


def main():
    parser = argparse.ArgumentParser(description="Real-Time Facial Affect & Emotional Biomarker Tracker")
    parser.add_argument("--camera-id", type=int, default=0, help="웹캠 카메라 디바이스 ID (기본값: 0)")
    parser.add_argument("--fps", type=int, default=15, help="타겟 FPS (기본값: 15)")
    parser.add_argument("--no-gui", action="store_true", help="화면 표시 없이 백그라운드에서 추론 및 DB 로깅만 수행")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite 데이터베이스 파일 경로")
    args = parser.parse_args()

    # 1. 모델 확인 및 자동 다운로드
    model_path = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        print("[INFO] 모델 파일이 없습니다. 자동으로 다운로드를 진행합니다...")
        download_face_landmarker_model(model_path)

    # 2. DB 핸들러 초기화
    db_handler = AffectDBHandler(db_path=args.db_path)

    # 3. 트래커 초기화 및 실행
    tracker = WebcamFaceTracker(
        model_path=str(model_path),
        camera_id=args.camera_id,
        target_fps=args.fps,
        db_handler=db_handler,
        visualize=not args.no_gui
    )

    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt 감지. 종료합니다.")
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
