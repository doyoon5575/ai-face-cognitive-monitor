"""
시니어(어르신) 실시간 안면 표정 및 감정 모니터링 실행기 (senior_tracker.py)
- 웹캠을 통해 실시간 얼굴 인식 및 7대 감정 분석
- 큼직하고 시인성 높은 한글 HUD 및 따뜻한 안부 케어 메시지 표시
- SQLite DB(affect_logs.db -> senior_affect_logs) 시계열 배치 저장
"""

import argparse
import sys
import time
from pathlib import Path
import cv2

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# 프로젝트 루트 경로 등록
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from core.senior_analyzer import SeniorEmotionAnalyzer
from database.db_handler import SeniorAffectDBHandler


def parse_args():
    parser = argparse.ArgumentParser(description="시니어 실시간 안면 감정 모니터링 시스템")
    parser.add_argument("--camera-id", type=int, default=0, help="카메라 장치 인덱스 (기본값: 0)")
    parser.add_argument("--width", type=int, default=960, help="화면 가로 해상도 (기본값: 960)")
    parser.add_argument("--height", type=int, default=540, help="화면 세로 해상도 (기본값: 540)")
    parser.add_argument("--skip-frames", type=int, default=4, help="추론 주기 (N프레임마다 1회 추론으로 속도 최적화, 기본값: 4)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(" [시니어 안면 표정 케어 시스템] 실시간 모니터링을 시작합니다.")
    print(f" - 카메라 인덱스: {args.camera_id}")
    print(f" - 해상도: {args.width}x{args.height}")
    print(" - 종료 방법: 화면을 클릭한 후 키보드 [Q] 또는 [ESC] 키를 누르세요.")
    print("=" * 60)

    # 1. 분석기 및 DB 핸들러 초기화
    analyzer = SeniorEmotionAnalyzer()
    db_handler = SeniorAffectDBHandler()

    # 2. 웹캠 연결
    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[경고] 카메라 {args.camera_id}번을 기본 백엔드로 재시도합니다...")
        cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"[오류] 카메라(장치 {args.camera_id})에 연결할 수 없습니다. 웹캠 연결 상태를 확인해 주세요.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "시니어 마음 건강 거울 (종료: Q 또는 ESC)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    frame_count = 0
    last_analysis = None
    fps_timer = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[알림] 카메라 프레임을 읽어올 수 없습니다.")
                break

            # 거울 모드 (좌우 반전 - 어르신들이 보기에 자연스러움)
            frame = cv2.flip(frame, 1)
            frame_count += 1

            # N프레임마다 1회 감정 추론 실행 (저사양 노트북에서도 부드러운 30FPS 유지)
            if frame_count % args.skip_frames == 0 or last_analysis is None:
                analysis = analyzer.analyze_frame(frame)
                if analysis:
                    last_analysis = analysis
                    # DB 버퍼에 기록
                    db_handler.add_record(
                        dominant_emotion=analysis['dominant'],
                        emotion_ko=analysis['dominant_ko'],
                        confidence=analysis['confidence'],
                        scores=analysis['scores'],
                        care_message=analysis['care_message']
                    )

            # FPS 계산
            if frame_count % 15 == 0:
                now = time.time()
                fps_display = 15.0 / max(0.001, (now - fps_timer))
                fps_timer = now

            # 시니어 전용 대형 한글 HUD 렌더링
            rendered_frame = analyzer.render_senior_hud(frame, last_analysis, fps=fps_display)

            cv2.imshow(window_name, rendered_frame)

            # 키 입력 처리 (Q 또는 ESC)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n[안내] 사용자가 종료 키를 눌렀습니다.")
                break

    except KeyboardInterrupt:
        print("\n[안내] 키보드 인터럽트로 종료합니다.")
    finally:
        print("[안내] 데이터베이스에 남은 버퍼를 저장하고 자원을 해제합니다...")
        db_handler.flush()
        cap.release()
        cv2.destroyAllWindows()
        print("[완료] 프로그램이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    main()
