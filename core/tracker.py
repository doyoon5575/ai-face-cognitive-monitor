"""
OpenCV 웹캠 캡처 및 MediaPipe Tasks Face Landmarker 추론 엔진
- 15~30 FPS 실시간 스트림 처리
- 52종 Blendshape 및 얼굴 랜드마크 추출
- 실시간 시각화 오버레이 (HUD, 감정 게이지 바, 헤드포즈 상태)
- SQLite 자동 시계열 저장 연동
"""

import os
import time
from pathlib import Path
from typing import Callable, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from database.db_handler import AffectDBHandler
from .analyzer import AffectMetrics, FacialAffectAnalyzer

# MediaPipe Tasks Vision API 임포트
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class WebcamFaceTracker:
    def __init__(
        self,
        model_path: Optional[str] = None,
        camera_id: int = 0,
        target_fps: int = 15,
        db_handler: Optional[AffectDBHandler] = None,
        visualize: bool = True
    ):
        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.model_path = str(base_dir / "models" / "face_landmarker.task")
        else:
            self.model_path = str(model_path)

        self.camera_id = camera_id
        self.target_fps = target_fps
        self.db_handler = db_handler or AffectDBHandler()
        self.visualize = visualize
        self.analyzer = FacialAffectAnalyzer(max_rotation_degrees=35.0)

        self._landmarker: Optional[vision.FaceLandmarker] = None
        self._is_running = False
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        """MediaPipe Face Landmarker 초기화 (VIDEO 모드, Unicode 경로 호환을 위해 바이너리 버퍼 로드)"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Face Landmarker 모델 파일이 없습니다: {self.model_path}\n"
                f"먼저 `python download_model.py`를 실행하여 모델을 다운로드해주세요."
            )

        with open(self.model_path, "rb") as f:
            model_buffer = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_buffer)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def draw_hud(
        self,
        frame: np.ndarray,
        metrics: Optional[AffectMetrics],
        fps: float
    ) -> np.ndarray:
        """실시간 정서 지표 및 HUD를 프레임에 렌더링"""
        h, w, _ = frame.shape
        overlay = frame.copy()

        # 상단 HUD 반투명 배경 (다크 테마)
        cv2.rectangle(overlay, (15, 15), (380, 260), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (15, 15), (380, 260), (70, 85, 110), 1, cv2.LINE_AA)

        # 시스템 타이틀 & FPS
        cv2.putText(frame, "AFFECT BIOMARKER MONITOR", (28, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (300, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

        if metrics is None:
            cv2.putText(frame, "NO FACE DETECTED", (28, 140),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            return frame

        # 게이지 바 렌더링 함수
        def draw_bar(y_pos: int, label: str, value: float, color: Tuple[int, int, int]):
            cv2.putText(frame, f"{label}: {value:.2f}", (28, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            # 바 배경
            cv2.rectangle(frame, (170, y_pos - 12), (360, y_pos + 2), (45, 50, 60), -1)
            # 채워진 바
            bar_w = int(190 * max(0.0, min(1.0, value)))
            cv2.rectangle(frame, (170, y_pos - 12), (170 + bar_w, y_pos + 2), color, -1)
            cv2.rectangle(frame, (170, y_pos - 12), (360, y_pos + 2), (90, 100, 120), 1)

        # 감정 지표 게이지 바
        draw_bar(75, "Smile (웃음)", metrics.smile_score, (50, 205, 50))       # 녹색
        draw_bar(105, "Frown (찡그림)", metrics.frown_score, (0, 140, 255))     # 주황색
        draw_bar(135, "Flatness (둔마)", metrics.flatness_score, (255, 105, 180)) # 핑크/자주색
        draw_bar(165, "Blink (눈감김)", metrics.blink_score, (255, 215, 0))     # 시안/골드

        # 헤드 포즈 상태
        hp = metrics.head_pose
        pose_color = (0, 255, 128) if hp.is_valid else (0, 0, 255)
        pose_text = f"Pose: Y:{hp.yaw:+4.1f} P:{hp.pitch:+4.1f} R:{hp.roll:+4.1f}"
        if not hp.is_valid:
            pose_text += " [WARN >35deg]"

        cv2.putText(frame, pose_text, (28, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, pose_color, 1, cv2.LINE_AA)

        # 상태 요약 태그
        status_text = "STATUS: "
        if metrics.flatness_score > 0.85:
            status_text += "HIGH FLATNESS (정서 둔마)"
            tag_color = (0, 0, 255)
        elif metrics.smile_score > 0.40:
            status_text += "SMILING (긍정 정서)"
            tag_color = (0, 255, 0)
        elif metrics.frown_score > 0.35:
            status_text += "FROWNING (부정 정서)"
            tag_color = (0, 165, 255)
        else:
            status_text += "NEUTRAL (보통)"
            tag_color = (200, 200, 200)

        # 조작 가이드 안내
        cv2.putText(frame, "[Space] Pause/Resume | [Q/ESC] Stop Camera", (28, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    def run(
        self,
        frame_callback: Optional[Callable[[np.ndarray, Optional[AffectMetrics]], None]] = None,
        max_duration_sec: Optional[float] = None
    ) -> None:
        """
        웹캠 스트림을 실행하고 실시간 추론 및 DB 저장을 수행합니다.
        'q' 또는 ESC 키를 누르면 종료됩니다.
        'Space' 키를 누르면 일시정지/재개됩니다.
        """
        # 카메라 열기 (자동 대체 탐색)
        cap = None
        for cam_id in [self.camera_id, 0, 1, 2]:
            print(f"[INFO] 카메라 ID {cam_id} 연결 시도 중...")
            cap_try = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if not cap_try.isOpened():
                cap_try = cv2.VideoCapture(cam_id)
            
            if cap_try.isOpened():
                ret, test_frame = cap_try.read()
                if ret and test_frame is not None:
                    cap = cap_try
                    self.camera_id = cam_id
                    print(f"[SUCCESS] 카메라 ID {cam_id}에 성공적으로 연결되었습니다!")
                    break
                cap_try.release()

        if cap is None or not cap.isOpened():
            raise RuntimeError(
                f"사용 가능한 웹캠을 열 수 없습니다.\n"
                f"1. 다른 프로그램(Zoom, Teams, 브라우저 등)이 카메라를 사용 중인지 확인해주세요.\n"
                f"2. Windows [설정] -> [개인 정보 및 보안] -> [카메라]에서 앱 접근 권한이 켜져 있는지 확인해주세요."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        window_name = "Facial Affect Monitor - MediaPipe Live Stream"
        if self.visualize:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)

        self._is_running = True
        frame_delay = 1.0 / self.target_fps
        start_time = time.time()
        prev_frame_time = time.time()
        fps = float(self.target_fps)

        print("[INFO] =================================================")
        print("[INFO] 웹캠 추적기 화면이 실행되었습니다.")
        print("[INFO] 카메라를 정면으로 바라보고 표정을 지어보세요.")
        print("[INFO] 창을 종료하려면 카메라 창에서 'q' 또는 ESC를 누르세요.")
        print("[INFO] =================================================")

        is_paused = False

        try:
            while self._is_running and cap.isOpened():
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                # 좌우 반전 (거울 모드)
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                if is_paused:
                    cv2.putText(frame, "== PAUSED (PRESS SPACE TO RESUME) ==", (50, int(h / 2)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Facial Affect Monitor - MediaPipe Live Stream", frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key in [ord('q'), 27]:
                        break
                    elif key == 32:  # Space
                        is_paused = False
                    continue

                # MediaPipe 입력 이미지 생성 (RGB 변환)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int((time.time() - start_time) * 1000)

                # Face Landmarker 추론 실행
                detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

                metrics: Optional[AffectMetrics] = None
                if detection_result and detection_result.face_blendshapes:
                    blendshapes = detection_result.face_blendshapes[0]
                    landmarks = detection_result.face_landmarks[0] if detection_result.face_landmarks else None
                    metrics = self.analyzer.analyze(blendshapes, landmarks, w, h)

                    # DB에 비동기/배치 적재
                    self.db_handler.add_record(
                        smile_score=metrics.smile_score,
                        frown_score=metrics.frown_score,
                        flatness_score=metrics.flatness_score,
                        blink_detected=metrics.blink_detected,
                        head_pose_valid=metrics.head_pose.is_valid,
                        yaw=metrics.head_pose.yaw,
                        pitch=metrics.head_pose.pitch,
                        roll=metrics.head_pose.roll
                    )

                # FPS 계산
                current_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, current_time - prev_frame_time))
                prev_frame_time = current_time

                # 시각화 및 HUD 오버레이
                if self.visualize:
                    frame = self.draw_hud(frame, metrics, fps)
                    cv2.imshow("Facial Affect Monitor - MediaPipe Live Stream", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), 27]:  # 'q' or ESC
                        print("[INFO] 사용자에 의해 카메라 끄기(종료) 요청되었습니다.")
                        break
                    elif key == 32:  # Space
                        is_paused = True
                        print("[INFO] 카메라 일시정지됨. (Space 키를 다시 누르면 재개)")

                if frame_callback:
                    frame_callback(frame, metrics)

                # 최대 실행 시간 체크
                if max_duration_sec and (time.time() - start_time >= max_duration_sec):
                    break

                # FPS 제어 (프레임 딜레이)
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self._is_running = False
            cap.release()
            cv2.destroyAllWindows()
            self.db_handler.flush()
            print("[INFO] 웹캠 추적기가 안전하게 종료되었습니다. 잔여 데이터가 DB에 커밋되었습니다.")

    def stop(self) -> None:
        """실행 중지 플래그 설정"""
        self._is_running = False
