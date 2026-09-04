"""
시니어(어르신) 친화적 실시간 안면 감정 분석 및 첨단 페이스 메시(Face Mesh) HUD 모듈
- MediaPipe Tasks 기반 478개 안면 랜드마크 윤곽선 (눈, 코, 입, 눈썹, 턱선, 안면 와이어프레임) 실시간 렌더링
- DeepFace 기반 7대 기본 감정(기쁨, 평온, 슬픔, 화남, 놀람, 불안, 불쾌) 추론
- 텍스트 네모(□) 글리프 완전 박멸 (유니코드 정규식 정제)
- 표정 세부 분석 카드 리디자인 및 명칭 중복("마음이" 2개) 해결
"""

import os
import sys
import time
import random
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 콘솔 UTF-8 보장
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def clean_korean_text(text: str) -> str:
    """윈도우 폰트(맑은 고딕)에서 네모(□)로 깨질 수 있는 이모지 및 특수 유니코드 제거"""
    if not text:
        return ""
    # 한글, 영문, 숫자, 기본 기호 및 문장부호만 보존
    cleaned = re.sub(r'[^\w\s\(\)\/\:\,\.\%\-\!\?\~\[\]\<\>\'\"]', '', text)
    # 다중 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# 7대 감정 설정 (이모지 완전 배제 - 순수 한글 라벨)
EMOTION_CONFIG = {
    'happy': {
        'ko': '활짝 웃음 (기쁨)',
        'short': '기쁨/웃음',
        'color_rgb': (46, 204, 113),    # 상쾌한 에메랄드 그린
        'color_bgr': (113, 204, 46),
        'weather': '맑음',
        'messages': [
            "어르신의 활짝 웃으시는 모습이 참 보기 좋습니다!",
            "오늘 기분이 아주 상쾌해 보이세요! 늘 행복하세요.",
            "밝은 미소 덕분에 온 세상이 환해지는 것 같습니다!"
        ]
    },
    'neutral': {
        'ko': '편안하고 평온함',
        'short': '평온/편안',
        'color_rgb': (52, 152, 219),    # 차분한 하늘색
        'color_bgr': (219, 152, 52),
        'weather': '화창',
        'messages': [
            "마음이 편안하고 차분해 보이세요.",
            "오늘도 평온하고 기분 좋은 하루 보내세요!",
            "여유롭고 건강한 하루를 보내고 계시네요. 참 좋습니다."
        ]
    },
    'surprise': {
        'ko': '깜짝 놀람',
        'short': '놀람/호기심',
        'color_rgb': (241, 196, 15),   # 밝은 노랑
        'color_bgr': (15, 196, 241),
        'weather': '무지개',
        'messages': [
            "어떤 신기하고 재미있는 소식을 접하셨나요?",
            "호기심 가득한 눈빛이 정말 젊고 활기차 보이세요!",
            "오늘도 흥미롭고 즐거운 일들이 가득하길 바랍니다."
        ]
    },
    'sad': {
        'ko': '마음이 울적함 (슬픔)',
        'short': '슬픔/울적',
        'color_rgb': (155, 89, 182),   # 부드러운 라벤더 퍼플
        'color_bgr': (182, 89, 155),
        'weather': '비',
        'messages': [
            "오늘 마음이 조금 무겁거나 적적하신가요?",
            "따뜻한 차 한 잔 드시며 좋아하시는 음악을 들어보세요.",
            "언제나 어르신을 응원하고 사랑하는 가족이 곁에 있습니다."
        ]
    },
    'angry': {
        'ko': '마음이 답답함 (화남)',
        'short': '화남/답답',
        'color_rgb': (231, 76, 60),    # 온화한 레드
        'color_bgr': (60, 76, 231),
        'weather': '천둥',
        'messages': [
            "속상하거나 답답한 일이 있으셨나요?",
            "천천히 숨을 깊게 세 번 들이쉬고 내쉬어보세요.",
            "잠시 눈을 감고 편안하게 쉬어가세요. 다 잘 풀릴 거예요."
        ]
    },
    'fear': {
        'ko': '긴장되고 불안함',
        'short': '불안/긴장',
        'color_rgb': (230, 126, 34),   # 오렌지
        'color_bgr': (34, 126, 230),
        'weather': '안개',
        'messages': [
            "마음이 불안하거나 걱정되는 일이 있으신가요?",
            "편안하게 어깨 힘을 빼고 따뜻한 온기를 느껴보세요.",
            "어르신의 하루가 평안하도록 함께합니다. 안심하세요."
        ]
    },
    'disgust': {
        'ko': '불편함/언짢음',
        'short': '불편',
        'color_rgb': (149, 165, 166),  # 부드러운 그레이
        'color_bgr': (166, 165, 149),
        'weather': '흐림',
        'messages': [
            "어디 불편하신 곳이나 언짢은 일이 있으신가요?",
            "창문을 열어 맑은 공기를 천천히 들이마셔 보세요.",
            "몸과 마음이 편안해지도록 가벼운 기지개를 켜보세요."
        ]
    }
}

# MediaPipe 얼굴 랜드마크 연결선 인덱스 정의
FACE_CONTOURS = {
    # 얼굴 외곽 턱선
    "jawline": [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109, 10
    ],
    # 좌측 눈썹
    "left_eyebrow": [70, 63, 105, 66, 107],
    # 우측 눈썹
    "right_eyebrow": [336, 296, 334, 293, 300],
    # 좌측 눈
    "left_eye": [33, 160, 158, 133, 153, 144, 33],
    # 우측 눈
    "right_eye": [263, 387, 385, 362, 380, 373, 263],
    # 콧날 및 코끝
    "nose_bridge": [168, 6, 197, 195, 5, 4, 1, 2],
    "nose_bottom": [98, 97, 2, 326, 327],
    # 입술 바깥쪽 윤곽선
    "lips_outer": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61
    ]
}


class SeniorEmotionAnalyzer:
    """시니어 안면 랜드마크 선 및 표정 분석 클래스"""

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path or self._find_korean_font()
        self._fonts = {}

        # 폰트 크기별 캐싱
        for size in [15, 17, 19, 21, 23, 26, 32, 36]:
            try:
                self._fonts[size] = ImageFont.truetype(self.font_path, size)
            except Exception:
                self._fonts[size] = ImageFont.load_default()

        # 최근 추론 결과 캐시
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_care_message: str = "카메라를 정면으로 편안하게 바라봐 주세요."
        self.last_message_change_time: float = 0.0
        self.current_dominant: str = "neutral"

        # MediaPipe Face Landmarker 초기화 (얼굴 랜드마크 선 그리기용)
        self._mp_landmarker = None
        self._init_mediapipe()

        # 모델 웜업
        self._warmup_model()

    def _find_korean_font(self) -> str:
        candidates = [
            "C:/Windows/Fonts/malgun.ttf",       # 맑은 고딕
            "C:/Windows/Fonts/malgunbd.ttf",     # 맑은 고딕 볼드
            "C:/Windows/Fonts/gulim.ttc",        # 굴림
            "C:/Windows/Fonts/batang.ttc",       # 바탕
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return "arial.ttf"

    def _init_mediapipe(self) -> None:
        """MediaPipe Face Landmarker 초기화 (models/face_landmarker.task 사용)"""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_dir = Path(__file__).resolve().parent.parent
            model_path = str(base_dir / "models" / "face_landmarker.task")

            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model_buffer = f.read()

                base_options = python.BaseOptions(model_asset_buffer=model_buffer)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.4,
                    min_face_presence_confidence=0.4,
                    min_tracking_confidence=0.4,
                    output_face_blendshapes=True
                )
                self._mp_landmarker = vision.FaceLandmarker.create_from_options(options)
                print("[SeniorAnalyzer] MediaPipe Face Landmarker 초기화 완료!")
            else:
                print(f"[SeniorAnalyzer] Face Landmarker 모델 파일 없음: {model_path}")
        except Exception as e:
            print(f"[SeniorAnalyzer] MediaPipe 초기화 경고: {e}")
            self._mp_landmarker = None

    def _warmup_model(self) -> None:
        """DeepFace 사전 로딩"""
        try:
            from deepface import DeepFace
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(
                img_path=dummy,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
        except Exception as e:
            pass

    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        return self._fonts.get(size, self._fonts.get(21, ImageFont.load_default()))

    def extract_landmarks(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """MediaPipe를 통해 478개 안면 랜드마크 픽셀 좌표 추출"""
        if self._mp_landmarker is None:
            return None
        try:
            import mediapipe as mp
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._mp_landmarker.detect(mp_image)

            if result and result.face_landmarks:
                landmarks = result.face_landmarks[0]
                coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                return coords
        except Exception:
            pass
        return None

    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """DeepFace 감정 분석 실행"""
        try:
            from deepface import DeepFace
            objs = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            if not objs:
                return self.last_result

            obj = objs[0] if isinstance(objs, list) else objs
            dominant = obj.get('dominant_emotion', 'neutral').lower()
            emotions = obj.get('emotion', {})
            region = obj.get('region', {})

            total = sum(emotions.values()) if emotions else 100.0
            norm_emotions = {k: (v / total) * 100.0 for k, v in emotions.items()}
            confidence = norm_emotions.get(dominant, 0.0)

            cfg = EMOTION_CONFIG.get(dominant, EMOTION_CONFIG['neutral'])

            now = time.time()
            if dominant != self.current_dominant or (now - self.last_message_change_time > 5.0):
                self.current_dominant = dominant
                raw_msg = random.choice(cfg['messages'])
                self.last_care_message = clean_korean_text(raw_msg)
                self.last_message_change_time = now

            result = {
                'dominant': dominant,
                'dominant_ko': clean_korean_text(cfg['ko']),
                'short_ko': cfg['short'],
                'confidence': confidence,
                'scores': norm_emotions,
                'care_message': self.last_care_message,
                'region': region,
                'weather': clean_korean_text(cfg['weather']),
                'color_rgb': cfg['color_rgb'],
                'color_bgr': cfg['color_bgr']
            }
            self.last_result = result
            return result

        except Exception as e:
            return self.last_result

    def render_senior_hud(
        self,
        frame: np.ndarray,
        analysis: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        시니어 전용 고대비 대형 한글 HUD 및 얼굴 랜드마크 선(Face Mesh) 렌더링
        """
        h, w, _ = frame.shape
        out_frame = frame.copy()

        # 1. 안면 랜드마크 선(Face Mesh Wireframe) 그리기
        landmarks = self.extract_landmarks(frame)
        accent_color_bgr = analysis['color_bgr'] if analysis else (200, 150, 50)
        accent_color_rgb = analysis['color_rgb'] if analysis else (50, 150, 200)

        if landmarks and len(landmarks) >= 468:
            # 주요 윤곽선들 그리기 (눈, 눈썹, 코, 입, 턱선)
            mesh_overlay = out_frame.copy()

            for part_name, indices in FACE_CONTOURS.items():
                pts = np.array([landmarks[i] for i in indices if i < len(landmarks)], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                # 미세하고 세련된 네온 라인
                cv2.polylines(mesh_overlay, [pts], isClosed=(part_name in ["left_eye", "right_eye", "lips_outer"]),
                              color=accent_color_bgr, thickness=1, lineType=cv2.LINE_AA)

            # 얼굴 주요 특징 포인트 (눈동자 중심, 코끝, 입꼬리, 이마 중심) 점찍기
            key_points = [1, 4, 33, 133, 263, 362, 61, 291, 10, 152]
            for kp in key_points:
                if kp < len(landmarks):
                    cv2.circle(mesh_overlay, landmarks[kp], 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(mesh_overlay, landmarks[kp], 4, accent_color_bgr, 1, cv2.LINE_AA)

            # 반투명 블렌딩으로 얼굴을 가리지 않고 미래지향적 선들 표시
            cv2.addWeighted(mesh_overlay, 0.75, out_frame, 0.25, 0, out_frame)

        # 2. PIL을 이용한 텍스트 및 UI 카드 오버레이
        rgb_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # 상단 헤더 바 (높이 86px 반투명 다크)
        draw.rectangle([(0, 0), (w, 86)], fill=(18, 22, 32, 225))
        draw.rectangle([(0, 84), (w, 87)], fill=accent_color_rgb)

        font_large = self.get_font(32)
        font_sub = self.get_font(19)
        font_msg = self.get_font(21)
        font_small = self.get_font(15)

        if analysis:
            dom_ko = clean_korean_text(analysis['dominant_ko'])
            conf = analysis['confidence']
            weather_ko = clean_korean_text(analysis['weather'])

            # 상단 제목 (네모 글리프 완전 배제)
            header_text = f"[현재 표정]  {dom_ko}"
            draw.text((24, 12), header_text, font=font_large, fill=(255, 255, 255))

            sub_text = f"상태 일치도: {conf:.0f}%  |  오늘의 마음 날씨: {weather_ko}"
            draw.text((26, 54), sub_text, font=font_sub, fill=(215, 230, 250))

            # 3. 우측 표정 세부 분석 카드 (너비 270px로 확대하여 완벽한 수평 정렬)
            scores = analysis.get('scores', {})
            # 상위 3개 감정 추출
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]

            card_w = 270
            card_h = 135
            card_x = w - card_w - 20
            card_y = 96

            # 카드 배경
            draw.rectangle([(card_x, card_y), (card_x + card_w, card_y + card_h)],
                           fill=(18, 24, 36, 215), outline=(75, 85, 105), width=1)
            draw.text((card_x + 14, card_y + 10), "표정 세부 분석", font=self.get_font(17), fill=(230, 235, 245))

            row_y = card_y + 38
            for emo_key, score_val in sorted_scores:
                cfg_item = EMOTION_CONFIG.get(emo_key, {})
                # 명칭 중복 없는 고유 짧은 이름 사용 (예: 기쁨/웃음, 평온/편안, 슬픔/울적 등)
                short_name = cfg_item.get('short', emo_key)
                emo_color = cfg_item.get('color_rgb', (200, 200, 200))

                # [1열: 감정명 72px]
                draw.text((card_x + 14, row_y), short_name, font=font_small, fill=(240, 240, 240))

                # [2열: 프로그레스 바 115px]
                bar_x1 = card_x + 92
                bar_x2 = bar_x1 + 115
                draw.rectangle([(bar_x1, row_y + 4), (bar_x2, row_y + 15)], fill=(45, 55, 70))
                fill_w = int(115 * (max(0.0, min(100.0, score_val)) / 100.0))
                if fill_w > 0:
                    draw.rectangle([(bar_x1, row_y + 4), (bar_x1 + fill_w, row_y + 15)], fill=emo_color)

                # [3열: 백분율 수치 45px]
                draw.text((bar_x2 + 10, row_y), f"{score_val:.0f}%", font=font_small, fill=(210, 225, 240))

                row_y += 28

            care_msg = clean_korean_text(analysis.get('care_message', ""))
        else:
            draw.text((24, 24), "얼굴 위치를 탐색 중입니다...", font=font_large, fill=(240, 240, 240))
            care_msg = "카메라를 정면으로 바라보시면 얼굴 윤곽과 표정을 실시간 분석합니다."

        # 4. 하단 케어 메시지 배너 (높이 76px)
        banner_h = 76
        draw.rectangle([(0, h - banner_h), (w, h)], fill=(14, 18, 28, 235))
        draw.rectangle([(0, h - banner_h), (w, h - banner_h + 3)], fill=accent_color_rgb)
        draw.text((24, h - banner_h + 24), care_msg, font=font_msg, fill=(255, 255, 240))

        # PIL -> OpenCV BGR 복원
        out_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return out_frame
