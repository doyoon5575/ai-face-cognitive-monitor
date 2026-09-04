"""
AI 다중 생체신호(안면·음성) 멀티모달 분석 및 통합 HUD 렌더링 모듈 (core/multimodal_analyzer.py)
- MediaPipe 478개 랜드마크 선(Face Mesh Wireframe) 실시간 렌더링
- 입술 움직임(Mouth Dynamics), 눈가 피로도(Eye Fatigue), 안면 대칭도(Facial Symmetry) 미세 바이오마커 산출
- DeepFace 7대 기본 감정 추론 결합
- 마이크 실시간 음성 바이오마커(성량/활력도) 동시 시각화
- '시니어/어르신' 명칭 배제, 누구나 사용 가능한 포용적 범용 디자인
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

from core.audio_analyzer import RealtimeAudioAnalyzer


def clean_korean_text(text: str) -> str:
    """윈도우 폰트에서 네모(□)로 깨질 수 있는 이모지 및 특수 유니코드 제거"""
    if not text:
        return ""
    cleaned = re.sub(r'[^\w\s\(\)\/\:\,\.\%\-\!\?\~\[\]\<\>\'\"]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# 7대 감정 범용 설정 (아동, 성인, 시니어 누구나 공감하는 표준 안내 멘트)
EMOTION_CONFIG = {
    'happy': {
        'ko': '활짝 웃음 (기쁨)',
        'short': '기쁨/웃음',
        'color_rgb': (46, 204, 113),
        'color_bgr': (113, 204, 46),
        'weather': '맑음',
        'messages': [
            "밝고 활기찬 미소가 참 보기 좋습니다!",
            "긍정적인 에너지가 가득한 상태입니다. 좋은 하루 보내세요!",
            "기분 좋은 미소 덕분에 주변까지 밝아집니다."
        ]
    },
    'neutral': {
        'ko': '편안하고 평온함',
        'short': '평온/편안',
        'color_rgb': (52, 152, 219),
        'color_bgr': (219, 152, 52),
        'weather': '화창',
        'messages': [
            "마음이 편안하고 차분하게 안정되어 있습니다.",
            "차분한 집중과 여유로운 컨디션을 유지하고 계십니다.",
            "오늘도 평온하고 안정된 하루를 이어가 보세요."
        ]
    },
    'surprise': {
        'ko': '깜짝 놀람 (호기심)',
        'short': '놀람/호기심',
        'color_rgb': (241, 196, 15),
        'color_bgr': (15, 196, 241),
        'weather': '무지개',
        'messages': [
            "호기심과 흥미가 높아진 활기찬 상태입니다!",
            "새로운 자극이나 소식에 주의가 집중되고 있습니다.",
            "흥미진진하고 활력 넘치는 순간입니다."
        ]
    },
    'sad': {
        'ko': '마음이 울적함 (슬픔)',
        'short': '슬픔/울적',
        'color_rgb': (155, 89, 182),
        'color_bgr': (182, 89, 155),
        'weather': '비',
        'messages': [
            "마음이 조금 가라앉아 있거나 지쳐 보입니다.",
            "따뜻한 음료 한 잔과 함께 편안한 휴식을 권해드립니다.",
            "잠시 눈을 쉬게 하고 좋아하는 음악을 감상해 보세요."
        ]
    },
    'angry': {
        'ko': '마음이 답답함 (긴장)',
        'short': '화남/답답',
        'color_rgb': (231, 76, 60),
        'color_bgr': (60, 76, 231),
        'weather': '천둥',
        'messages': [
            "스트레스나 긴장이 다소 높아진 상태입니다.",
            "천천히 숨을 깊게 세 번 들이쉬고 내쉬어보세요.",
            "잠시 어깨 힘을 빼고 가벼운 스트레칭을 해보세요."
        ]
    },
    'fear': {
        'ko': '긴장되고 불안함',
        'short': '불안/긴장',
        'color_rgb': (230, 126, 34),
        'color_bgr': (34, 126, 230),
        'weather': '안개',
        'messages': [
            "주변 환경이나 문제에 긴장감이 감지되었습니다.",
            "편안한 호흡을 유지하시고 마음의 여유를 가져보세요.",
            "스스로를 믿고 편안하게 한 걸음씩 진행해 보세요."
        ]
    },
    'disgust': {
        'ko': '불편함/언짢음',
        'short': '불편',
        'color_rgb': (149, 165, 166),
        'color_bgr': (166, 165, 149),
        'weather': '흐림',
        'messages': [
            "몸이나 마음에 불편한 자극이 느껴지는 상태입니다.",
            "자세를 바르게 고쳐 앉거나 창문을 열어 환기해 보세요.",
            "가벼운 기지개를 켜며 컨디션을 전환해 보세요."
        ]
    }
}

# 얼굴 랜드마크 주요 윤곽선 인덱스
FACE_CONTOURS = {
    "jawline": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
    "left_eyebrow": [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
    "left_eye": [33, 160, 158, 133, 153, 144, 33],
    "right_eye": [263, 387, 385, 362, 380, 373, 263],
    "nose_bridge": [168, 6, 197, 195, 5, 4, 1, 2],
    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61]
}


class MultimodalAffectAnalyzer:
    """안면 미세 표정 + 음성 바이오마커 종합 분석 엔진"""

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path or self._find_korean_font()
        self._fonts = {}

        for size in [13, 15, 17, 19, 21, 24, 30, 34]:
            try:
                self._fonts[size] = ImageFont.truetype(self.font_path, size)
            except Exception:
                self._fonts[size] = ImageFont.load_default()

        # 최근 추론 결과 캐싱
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_care_message: str = "카메라와 마이크를 향해 편안하게 응답해 주세요."
        self.last_message_change_time: float = 0.0
        self.current_dominant: str = "neutral"

        # MediaPipe Face Landmarker 초기화
        self._mp_landmarker = None
        self._init_mediapipe()

        # 실시간 음성 분석기 초기화
        self.audio_analyzer = RealtimeAudioAnalyzer()

        # 웜업
        self._warmup_deepface()

    def _find_korean_font(self) -> str:
        candidates = [
            # Windows
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/malgunbd.ttf",
            "C:/Windows/Fonts/gulim.ttc",
            # Linux (Streamlit Cloud / Debian with fonts-nanum)
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/usr/share/fonts/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return "arial.ttf"

    def _init_mediapipe(self) -> None:
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_dir = Path(__file__).resolve().parent.parent
            model_path = str(base_dir / "models" / "face_landmarker.task")

            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    buf = f.read()

                base_options = python.BaseOptions(model_asset_buffer=buf)
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
        except Exception as e:
            self._mp_landmarker = None

    def _warmup_deepface(self) -> None:
        try:
            from deepface import DeepFace
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
        except Exception:
            pass

    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        return self._fonts.get(size, self._fonts.get(19, ImageFont.load_default()))

    def extract_landmarks_and_blendshapes(self, frame: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], Dict[str, float]]:
        """안면 478개 랜드마크 좌표 및 52종 미세 근육(Blendshape) 수치 추출"""
        if self._mp_landmarker is None:
            return None, {}

        try:
            import mediapipe as mp
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._mp_landmarker.detect(mp_img)

            landmarks = None
            blendshapes = {}

            if res and res.face_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in res.face_landmarks[0]]

            if res and res.face_blendshapes and res.face_blendshapes[0]:
                for b in res.face_blendshapes[0]:
                    blendshapes[b.category_name] = float(b.score)

            return landmarks, blendshapes
        except Exception:
            return None, {}

    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """DeepFace 및 MediaPipe 52종 미세 표정 근육 + 오디오 바이오마커 결합 분석"""
        try:
            # 1. 안면 랜드마크 및 52종 미세 표정 근육(Blendshapes) 추출
            landmarks, blendshapes = self.extract_landmarks_and_blendshapes(frame)

            dominant = 'neutral'
            confidence = 80.0
            norm_emotions = {
                'happy': 10.0, 'neutral': 70.0, 'sad': 5.0,
                'angry': 5.0, 'surprise': 5.0, 'fear': 3.0, 'disgust': 2.0
            }
            region = {}

            # 2. DeepFace 사용 가능 시 DeepFace 우선 분석
            deepface_success = False
            try:
                from deepface import DeepFace
                objs = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )
                if objs:
                    obj = objs[0] if isinstance(objs, list) else objs
                    dominant = obj.get('dominant_emotion', 'neutral').lower()
                    emotions = obj.get('emotion', {})
                    region = obj.get('region', {})
                    total = sum(emotions.values()) if emotions else 100.0
                    norm_emotions = {k: (v / total) * 100.0 for k, v in emotions.items()}
                    confidence = norm_emotions.get(dominant, 80.0)
                    deepface_success = True
            except Exception:
                deepface_success = False

            # 3. DeepFace 부재 시: MediaPipe 52종 안면 근육(Blendshapes)으로 초고속 정밀 감정 산출
            if not deepface_success and blendshapes:
                smile = (blendshapes.get('mouthSmileLeft', 0.0) + blendshapes.get('mouthSmileRight', 0.0)) / 2.0
                jaw = blendshapes.get('jawOpen', 0.0)
                brow_down = (blendshapes.get('browDownLeft', 0.0) + blendshapes.get('browDownRight', 0.0)) / 2.0
                brow_inner = blendshapes.get('browInnerUp', 0.0)
                frown = (blendshapes.get('mouthFrownLeft', 0.0) + blendshapes.get('mouthFrownRight', 0.0)) / 2.0

                happy_s = float(np.clip(smile * 130.0, 5.0, 98.0))
                surprise_s = float(np.clip((jaw * 60.0) + (brow_inner * 50.0), 5.0, 95.0))
                angry_s = float(np.clip(brow_down * 110.0, 5.0, 92.0))
                sad_s = float(np.clip(frown * 100.0, 5.0, 90.0))
                neutral_s = float(np.clip(100.0 - (happy_s * 0.5 + angry_s * 0.3 + sad_s * 0.3), 10.0, 90.0))

                scores_map = {
                    'happy': happy_s,
                    'surprise': surprise_s,
                    'angry': angry_s,
                    'sad': sad_s,
                    'neutral': neutral_s,
                    'fear': 8.0,
                    'disgust': 5.0
                }
                dominant = max(scores_map, key=scores_map.get)
                total_s = sum(scores_map.values())
                norm_emotions = {k: (v / total_s) * 100.0 for k, v in scores_map.items()}
                confidence = norm_emotions.get(dominant, 80.0)

            # 1. 입술 반응도 (0.0 ~ 100.0)
            mouth_smile = (blendshapes.get('mouthSmileLeft', 0.0) + blendshapes.get('mouthSmileRight', 0.0)) / 2.0
            jaw_open = blendshapes.get('jawOpen', 0.0)
            mouth_dynamic = float(np.clip((mouth_smile * 60.0) + (jaw_open * 40.0) + 10.0, 10.0, 98.0))

            # 2. 눈가 피로도 (0.0 ~ 100.0)
            blink = (blendshapes.get('eyeBlinkLeft', 0.0) + blendshapes.get('eyeBlinkRight', 0.0)) / 2.0
            frown = (blendshapes.get('browDownLeft', 0.0) + blendshapes.get('browDownRight', 0.0)) / 2.0
            eye_fatigue = float(np.clip((frown * 60.0) + (blink * 40.0), 5.0, 95.0))

            # 3. 안면 대칭도 (0.0 ~ 100.0)
            smile_diff = abs(blendshapes.get('mouthSmileLeft', 0.0) - blendshapes.get('mouthSmileRight', 0.0))
            symmetry = float(np.clip(100.0 - (smile_diff * 120.0), 70.0, 99.0))

            # 음성 바이오마커 실시간 지표 수신
            audio_metrics = self.audio_analyzer.get_metrics()

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
                'color_bgr': cfg['color_bgr'],
                # 미세 바이오마커 지표 3종
                'mouth_dynamic': round(mouth_dynamic, 1),
                'eye_fatigue': round(eye_fatigue, 1),
                'symmetry': round(symmetry, 1),
                # 음성 지표
                'audio': audio_metrics
            }
            self.last_result = result
            return result

        except Exception as e:
            return self.last_result

    def render_hud(
        self,
        frame: np.ndarray,
        analysis: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        통합 멀티모달 HUD 렌더링
        - 안면 랜드마크 선(Face Mesh)
        - 입술/눈가/대칭도 미세 바이오마커 인디케이터
        - 실시간 마이크 음성 활력 게이지
        - 세부분석 카드 정렬
        """
        h, w, _ = frame.shape
        out_frame = frame.copy()

        landmarks, _ = self.extract_landmarks_and_blendshapes(frame)
        accent_color_bgr = analysis['color_bgr'] if analysis else (200, 150, 50)
        accent_color_rgb = analysis['color_rgb'] if analysis else (50, 150, 200)

        # 1. 랜드마크 선(Face Mesh) 렌더링
        if landmarks and len(landmarks) >= 468:
            mesh_overlay = out_frame.copy()
            for part_name, indices in FACE_CONTOURS.items():
                pts = np.array([landmarks[i] for i in indices if i < len(landmarks)], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(mesh_overlay, [pts], isClosed=(part_name in ["left_eye", "right_eye", "lips_outer"]),
                              color=accent_color_bgr, thickness=1, lineType=cv2.LINE_AA)

            key_points = [1, 4, 33, 133, 263, 362, 61, 291, 10, 152]
            for kp in key_points:
                if kp < len(landmarks):
                    cv2.circle(mesh_overlay, landmarks[kp], 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(mesh_overlay, landmarks[kp], 4, accent_color_bgr, 1, cv2.LINE_AA)

            cv2.addWeighted(mesh_overlay, 0.78, out_frame, 0.22, 0, out_frame)

        # 2. PIL 한글 텍스트 및 정보 카드 오버레이
        rgb_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # 상단 헤더 바 (높이 84px)
        draw.rectangle([(0, 0), (w, 84)], fill=(18, 22, 32, 225))
        draw.rectangle([(0, 82), (w, 85)], fill=accent_color_rgb)

        font_large = self.get_font(30)
        font_sub = self.get_font(18)
        font_msg = self.get_font(20)
        font_small = self.get_font(14)

        if analysis:
            dom_ko = clean_korean_text(analysis['dominant_ko'])
            conf = analysis['confidence']
            weather_ko = clean_korean_text(analysis['weather'])
            symmetry = analysis.get('symmetry', 95.0)

            # 상단 제목 (네모 글리프 완전 없음)
            header_text = f"[현재 상태]  {dom_ko}"
            draw.text((22, 12), header_text, font=font_large, fill=(255, 255, 255))

            sub_text = f"일치도: {conf:.0f}%  |  컨디션 날씨: {weather_ko}  |  안면 대칭도: {symmetry:.0f}%"
            draw.text((24, 52), sub_text, font=font_sub, fill=(215, 230, 250))

            # 3. 우측 표정 세부 분석 카드 (너비 265px)
            scores = analysis.get('scores', {})
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]

            card_w = 265
            card_h = 130
            card_x = w - card_w - 18
            card_y = 94

            draw.rectangle([(card_x, card_y), (card_x + card_w, card_y + card_h)],
                           fill=(18, 24, 36, 215), outline=(75, 85, 105), width=1)
            draw.text((card_x + 14, card_y + 9), "표정 세부 분석", font=self.get_font(16), fill=(230, 235, 245))

            row_y = card_y + 36
            for emo_key, score_val in sorted_scores:
                cfg_item = EMOTION_CONFIG.get(emo_key, {})
                short_name = cfg_item.get('short', emo_key)
                emo_color = cfg_item.get('color_rgb', (200, 200, 200))

                draw.text((card_x + 14, row_y), short_name, font=font_small, fill=(240, 240, 240))

                bar_x1 = card_x + 88
                bar_x2 = bar_x1 + 115
                draw.rectangle([(bar_x1, row_y + 4), (bar_x2, row_y + 14)], fill=(45, 55, 70))
                fill_w = int(115 * (max(0.0, min(100.0, score_val)) / 100.0))
                if fill_w > 0:
                    draw.rectangle([(bar_x1, row_y + 4), (bar_x1 + fill_w, row_y + 14)], fill=emo_color)

                draw.text((bar_x2 + 8, row_y), f"{score_val:.0f}%", font=font_small, fill=(210, 225, 240))
                row_y += 27

            # 4. 좌측 상단: 미세 바이오마커 & 마이크 음성 활력 게이지 카드
            bio_w = 230
            bio_h = 104
            bio_x = 18
            bio_y = 94
            draw.rectangle([(bio_x, bio_y), (bio_x + bio_w, bio_y + bio_h)],
                           fill=(18, 24, 36, 215), outline=(75, 85, 105), width=1)
            draw.text((bio_x + 12, bio_y + 8), "미세 생체신호 분석", font=self.get_font(15), fill=(210, 230, 255))

            mouth_d = analysis.get('mouth_dynamic', 30.0)
            eye_f = analysis.get('eye_fatigue', 20.0)
            audio_info = self.audio_analyzer.get_metrics()
            vol = audio_info.get('volume', 0.0)

            # 입술 반응도 바
            draw.text((bio_x + 12, bio_y + 32), "입술 활력", font=self.get_font(13), fill=(220, 220, 220))
            draw.rectangle([(bio_x + 72, bio_y + 36), (bio_x + 172, bio_y + 44)], fill=(45, 55, 70))
            draw.rectangle([(bio_x + 72, bio_y + 36), (bio_x + 72 + int(100 * (mouth_d / 100.0)), bio_y + 44)], fill=(46, 204, 113))
            draw.text((bio_x + 180, bio_y + 32), f"{mouth_d:.0f}%", font=self.get_font(12), fill=(200, 210, 220))

            # 눈가 피로도 바
            draw.text((bio_x + 12, bio_y + 53), "눈가 피로", font=self.get_font(13), fill=(220, 220, 220))
            draw.rectangle([(bio_x + 72, bio_y + 57), (bio_x + 172, bio_y + 65)], fill=(45, 55, 70))
            draw.rectangle([(bio_x + 72, bio_y + 57), (bio_x + 72 + int(100 * (eye_f / 100.0)), bio_y + 65)], fill=(231, 76, 60))
            draw.text((bio_x + 180, bio_y + 53), f"{eye_f:.0f}%", font=self.get_font(12), fill=(200, 210, 220))

            # 마이크 음성 성량 바
            draw.text((bio_x + 12, bio_y + 74), "음성 성량", font=self.get_font(13), fill=(220, 220, 220))
            draw.rectangle([(bio_x + 72, bio_y + 78), (bio_x + 172, bio_y + 86)], fill=(45, 55, 70))
            mic_fill = int(100 * (min(100.0, vol) / 100.0))
            mic_color = (46, 204, 113) if vol > 12.0 else (100, 110, 130)
            if mic_fill > 0:
                draw.rectangle([(bio_x + 72, bio_y + 78), (bio_x + 72 + mic_fill, bio_y + 86)], fill=mic_color)
            mic_label = "발화중" if vol > 12.0 else "대기"
            draw.text((bio_x + 180, bio_y + 74), mic_label, font=self.get_font(12), fill=(180, 230, 200) if vol > 12.0 else (160, 170, 180))


            care_msg = clean_korean_text(analysis.get('care_message', ""))
        else:
            draw.text((22, 22), "생체신호 탐색 중입니다...", font=font_large, fill=(240, 240, 240))
            care_msg = "카메라와 마이크를 향해 편안하게 응답해 주세요."

        # 5. 하단 안내 배너 (높이 74px)
        banner_h = 74
        draw.rectangle([(0, h - banner_h), (w, h)], fill=(14, 18, 28, 235))
        draw.rectangle([(0, h - banner_h), (w, h - banner_h + 3)], fill=accent_color_rgb)
        draw.text((22, h - banner_h + 23), care_msg, font=font_msg, fill=(255, 255, 240))

        out_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return out_frame
