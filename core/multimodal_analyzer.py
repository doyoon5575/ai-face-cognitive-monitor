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


# 7대 감정 범용 설정 (아동, 성인, 시니어 누구나 공감하는 표준 웰니스 안내 멘트)
EMOTION_CONFIG = {
    'happy': {
        'ko': '미소 / 활력 양호',
        'short': '미소/활력',
        'color_rgb': (46, 204, 113),
        'color_bgr': (113, 204, 46),
        'weather': '맑음',
        'messages': [
            "밝고 활기찬 미소가 참 보기 좋습니다!",
            "긍정적인 에너지가 가득한 상태입니다.",
            "기분 좋은 미소 덕분에 주변까지 밝아집니다."
        ]
    },
    'neutral': {
        'ko': '편안하고 차분함',
        'short': '평온/안정',
        'color_rgb': (52, 152, 219),
        'color_bgr': (219, 152, 52),
        'weather': '화창',
        'messages': [
            "차분한 집중과 편안한 컨디션을 유지하고 계십니다.",
            "마음이 안정되고 평온한 상태입니다.",
            "오늘도 평온하고 안정된 하루를 이어가 보세요."
        ]
    },
    'surprise': {
        'ko': '호기심 / 각성 상태',
        'short': '각성/호기심',
        'color_rgb': (241, 196, 15),
        'color_bgr': (15, 196, 241),
        'weather': '무지개',
        'messages': [
            "새로운 자극이나 문제에 집중하고 계십니다.",
            "주의 집중도가 높아진 활기찬 순간입니다."
        ]
    },
    'sad': {
        'ko': '차분함 / 피로 휴식 권장',
        'short': '차분/피로',
        'color_rgb': (155, 89, 182),
        'color_bgr': (182, 89, 155),
        'weather': '비',
        'messages': [
            "컨디션이 조금 가라앉아 보입니다. 편안하게 호흡해 보세요.",
            "따뜻한 음료 한 잔과 함께 가벼운 휴식을 권해드립니다."
        ]
    },
    'angry': {
        'ko': '주의 집중 / 긴장 상태',
        'short': '집중/긴장',
        'color_rgb': (231, 76, 60),
        'color_bgr': (60, 76, 231),
        'weather': '천둥',
        'messages': [
            "문제에 깊이 집중하거나 다소 긴장된 상태입니다.",
            "천천히 숨을 깊게 들이쉬고 어깨 힘을 살짝 빼보세요.",
            "편안한 마음으로 한 걸음씩 여유를 가져보세요."
        ]
    },
    'fear': {
        'ko': '신중함 / 주의 깊음',
        'short': '신중/긴장',
        'color_rgb': (230, 126, 34),
        'color_bgr': (34, 126, 230),
        'weather': '안개',
        'messages': [
            "매우 신중하고 조심스럽게 상황을 살피고 계십니다.",
            "자신감을 가지고 편안하게 진행해 보세요."
        ]
    },
    'disgust': {
        'ko': '자세 피로 / 스트레칭 권장',
        'short': '피로/스트레칭',
        'color_rgb': (149, 165, 166),
        'color_bgr': (166, 165, 149),
        'weather': '흐림',
        'messages': [
            "목이나 어깨에 피로가 감지될 수 있습니다.",
            "가벼운 기지개를 켜며 컨디션을 전환해 보세요."
        ]
    }
}

# 얼굴 랜드마크 주요 윤곽선 인덱스 (MediaPipe Face Mesh 468/478 포인트)
FACE_CONTOURS = {
    # 턱선 및 얼굴 외곽선 (Neon Cyan)
    "jawline": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
    # 왼쪽 눈썹 (Neon Lime)
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    # 오른쪽 눈썹 (Neon Lime)
    "right_eyebrow": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    # 왼쪽 눈 (Bright Yellow)
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
    # 오른쪽 눈 (Bright Yellow)
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362],
    # 코 능선 및 코끝 (Neon Orange)
    "nose_bridge": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
    # 콧볼 (Neon Orange)
    "nose_wings": [98, 97, 2, 326, 327],
    # 입술 바깥선 (Vivid Coral)
    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61],
    # 입술 안쪽선 (Vivid Pink)
    "lips_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
    # 팔자 주름 및 볼 윤곽선 (Subtle wireframe)
    "nasolabial_left": [116, 117, 118, 101, 36],
    "nasolabial_right": [345, 346, 347, 330, 266],
    # 이마-미간 연결선 (Wireframe)
    "forehead_mesh": [10, 151, 9, 8, 168]
}

# 부위별 고대비 네온 컬러 매핑 (BGR format, thickness=2로 또렷하게 표현)
CONTOUR_STYLES = {
    "jawline": {"color": (255, 230, 0), "thickness": 2, "closed": False},         # 네온 시안
    "left_eyebrow": {"color": (0, 255, 120), "thickness": 2, "closed": False},     # 네온 라임
    "right_eyebrow": {"color": (0, 255, 120), "thickness": 2, "closed": False},    # 네온 라임
    "left_eye": {"color": (0, 240, 255), "thickness": 2, "closed": True},          # 비비드 옐로우
    "right_eye": {"color": (0, 240, 255), "thickness": 2, "closed": True},         # 비비드 옐로우
    "nose_bridge": {"color": (0, 180, 255), "thickness": 2, "closed": False},      # 네온 오렌지
    "nose_wings": {"color": (0, 180, 255), "thickness": 2, "closed": False},       # 네온 오렌지
    "lips_outer": {"color": (120, 70, 255), "thickness": 2, "closed": True},       # 비비드 코랄 핑크
    "lips_inner": {"color": (200, 120, 255), "thickness": 2, "closed": True},      # 밝은 마젠타
    "nasolabial_left": {"color": (240, 210, 80), "thickness": 1, "closed": False}, # 세련된 보조 와이어
    "nasolabial_right": {"color": (240, 210, 80), "thickness": 1, "closed": False},
    "forehead_mesh": {"color": (240, 210, 80), "thickness": 1, "closed": False}
}

# 고휘도 앵커 포인트
KEY_ANCHOR_POINTS = [
    33, 133, 159, 145, 263, 362, 386, 374,  # 눈 주요 포인트
    1, 4, 6, 168,                           # 코끝 & 콧대
    61, 291, 0, 17, 13, 14,                  # 입술 좌/우/상/하
    152, 10                                  # 턱끝 & 이마 상단
]


class MultimodalAffectAnalyzer:
    """안면 미세 표정 + 음성 바이오마커 종합 분석 엔진"""

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path or self._find_korean_font()
        self._fonts = {}

        for size in [12, 13, 14, 15, 16, 17, 19, 21, 24, 30, 34]:
            try:
                self._fonts[size] = ImageFont.truetype(self.font_path, size)
            except Exception:
                self._fonts[size] = ImageFont.load_default()

        # 최근 추론 결과 캐싱
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_care_message: str = "카메라와 마이크를 향해 편안하게 응답해 주세요."
        self.last_message_change_time: float = 0.0
        self.current_dominant: str = "neutral"

        # MediaPipe Face Landmarker 초기화 (초고속 XNNPACK 60FPS)
        self._mp_landmarker = None
        self._init_mediapipe()

        # 실시간 음성 분석기 초기화
        self.audio_analyzer = RealtimeAudioAnalyzer()

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
                    min_face_detection_confidence=0.35,
                    min_face_presence_confidence=0.35,
                    min_tracking_confidence=0.35,
                    output_face_blendshapes=True
                )
                self._mp_landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            self._mp_landmarker = None

    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        return self._fonts.get(size, self._fonts.get(16, ImageFont.load_default()))

    def extract_landmarks_and_blendshapes(self, frame: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], Dict[str, float]]:
        """안면 478개 랜드마크 좌표 및 52종 미세 근육(Blendshape) 수치 초고속 추출 (20ms)"""
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

            if res and res.face_landmarks and len(res.face_landmarks) > 0:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in res.face_landmarks[0]]

            if res and res.face_blendshapes and len(res.face_blendshapes) > 0 and res.face_blendshapes[0]:
                for b in res.face_blendshapes[0]:
                    blendshapes[b.category_name] = float(b.score)

            return landmarks, blendshapes
        except Exception:
            return None, {}

    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """MediaPipe 52종 FACS 미세 표정 근육 + 오디오 바이오마커 실시간 고감도 결합 분석 (60FPS)"""
        try:
            # 1. 안면 랜드마크 및 52종 미세 표정 근육(Blendshapes) 추출
            landmarks, blendshapes = self.extract_landmarks_and_blendshapes(frame)

            # 얼굴 미감지 시 기본값 유지 또는 마지막 결과 반환
            if not blendshapes:
                if self.last_result:
                    res = self.last_result.copy()
                    res['landmarks'] = landmarks
                    return res
                blendshapes = {}

            # 2. 미세 근육(FACS Action Units) 추출
            smile_l = blendshapes.get('mouthSmileLeft', 0.0)
            smile_r = blendshapes.get('mouthSmileRight', 0.0)
            avg_smile = (smile_l + smile_r) / 2.0

            jaw_open = blendshapes.get('jawOpen', 0.0)
            pucker = blendshapes.get('mouthPucker', 0.0)
            funnel = blendshapes.get('mouthFunnel', 0.0)
            shrug = blendshapes.get('mouthShrugLower', 0.0)
            upper_up = (blendshapes.get('mouthUpperUpLeft', 0.0) + blendshapes.get('mouthUpperUpRight', 0.0)) / 2.0

            blink_l = blendshapes.get('eyeBlinkLeft', 0.0)
            blink_r = blendshapes.get('eyeBlinkRight', 0.0)
            avg_blink = (blink_l + blink_r) / 2.0

            squint_l = blendshapes.get('eyeSquintLeft', 0.0)
            squint_r = blendshapes.get('eyeSquintRight', 0.0)
            avg_squint = (squint_l + squint_r) / 2.0

            brow_down_l = blendshapes.get('browDownLeft', 0.0)
            brow_down_r = blendshapes.get('browDownRight', 0.0)
            avg_brow_down = (brow_down_l + brow_down_r) / 2.0

            brow_inner = blendshapes.get('browInnerUp', 0.0)
            frown_l = blendshapes.get('mouthFrownLeft', 0.0)
            frown_r = blendshapes.get('mouthFrownRight', 0.0)
            avg_frown = (frown_l + frown_r) / 2.0

            eye_wide_l = blendshapes.get('eyeWideLeft', 0.0)
            eye_wide_r = blendshapes.get('eyeWideRight', 0.0)
            avg_eye_wide = (eye_wide_l + eye_wide_r) / 2.0

            nose_sneer = (blendshapes.get('noseSneerLeft', 0.0) + blendshapes.get('noseSneerRight', 0.0)) / 2.0

            # 3. 실시간 바이오마커 3종 산출 (0.0 ~ 100.0)
            # 입술 활력도: 발화, 미소, 입모양 움직임에 따라 즉각 민감 반응
            raw_mouth = (avg_smile * 45.0) + (jaw_open * 42.0) + (pucker * 25.0) + (funnel * 25.0) + (shrug * 20.0) + (upper_up * 20.0)
            mouth_dynamic = float(np.clip(raw_mouth * 1.45 + 12.0, 10.0, 99.0))

            # 눈가 피로/긴장도: 미간 찌푸림, 눈 찡그림, 깜빡임에 따라 즉각 반응
            raw_fatigue = (avg_squint * 42.0) + (avg_brow_down * 42.0) + (avg_blink * 30.0)
            eye_fatigue = float(np.clip(raw_fatigue * 1.5 + 10.0, 8.0, 96.0))

            # 안면 대칭도: 좌우 미소, 찡그림, 눈썹 대칭성 실시간 비교
            diff_smile = abs(smile_l - smile_r)
            diff_squint = abs(squint_l - squint_r)
            diff_brow = abs(brow_down_l - brow_down_r)
            total_diff = (diff_smile * 0.5) + (diff_squint * 0.3) + (diff_brow * 0.2)
            symmetry = float(np.clip(99.0 - (total_diff * 140.0), 65.0, 99.0))

            # 4. 7대 감정 분포 산출 (MediaPipe FACS Action Units)
            score_happy = float(np.clip((avg_smile * 135.0) + (upper_up * 20.0), 2.0, 98.0))
            score_surprise = float(np.clip((jaw_open * 70.0) + (brow_inner * 50.0) + (avg_eye_wide * 50.0), 2.0, 95.0))
            score_angry = float(np.clip((avg_brow_down * 125.0) + (avg_squint * 30.0), 2.0, 92.0))
            score_sad = float(np.clip((avg_frown * 105.0) + (brow_inner * 40.0) + (shrug * 30.0), 2.0, 90.0))
            score_fear = float(np.clip((brow_inner * 60.0) + (avg_eye_wide * 40.0), 2.0, 85.0))
            score_disgust = float(np.clip((nose_sneer * 110.0) + (upper_up * 30.0), 2.0, 85.0))

            # 평온 상태는 다른 흥분/감정 지표가 낮을 때 높게 산출
            arousal = (score_happy + score_surprise + score_angry + score_sad + score_fear + score_disgust) / 6.0
            score_neutral = float(np.clip(100.0 - (arousal * 1.8), 5.0, 95.0))

            raw_scores = {
                'happy': score_happy,
                'neutral': score_neutral,
                'surprise': score_surprise,
                'angry': score_angry,
                'sad': score_sad,
                'fear': score_fear,
                'disgust': score_disgust
            }
            total_sum = sum(raw_scores.values())
            norm_emotions = {k: round((v / total_sum) * 100.0, 1) for k, v in raw_scores.items()}
            dominant = max(norm_emotions, key=norm_emotions.get)
            confidence = norm_emotions[dominant]

            # 5. 오디오 바이오마커 실시간 지표 수신
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
                'region': {},
                'landmarks': landmarks,
                'weather': clean_korean_text(cfg['weather']),
                'color_rgb': cfg['color_rgb'],
                'color_bgr': cfg['color_bgr'],
                # 미세 바이오마커 지표 3종 (실시간 반응)
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
        - 선명한 네온 안면 랜드마크 선(Face Mesh Wireframe)
        - 실시간 입술/눈가/대칭도 미세 바이오마커 인디케이터
        - 실시간 마이크 음성 활력 게이지
        - 세부분석 카드 정렬
        """
        h, w, _ = frame.shape
        out_frame = frame.copy()

        landmarks = analysis.get('landmarks') if analysis else None
        if landmarks is None:
            landmarks, _ = self.extract_landmarks_and_blendshapes(frame)

        accent_color_bgr = analysis['color_bgr'] if analysis else (255, 200, 0)
        accent_color_rgb = analysis['color_rgb'] if analysis else (0, 200, 255)

        # 1. 고대비 네온 안면 랜드마크 선(Face Mesh) 렌더링
        if landmarks and len(landmarks) >= 468:
            # 부위별 또렷한 네온 컬러 와이어프레임 렌더링 (두께 2px, LINE_AA)
            for part_name, indices in FACE_CONTOURS.items():
                pts = np.array([landmarks[i] for i in indices if i < len(landmarks)], dtype=np.int32)
                if len(pts) > 1:
                    pts = pts.reshape((-1, 1, 2))
                    style = CONTOUR_STYLES.get(part_name, {"color": accent_color_bgr, "thickness": 2, "closed": False})
                    cv2.polylines(
                        out_frame,
                        [pts],
                        isClosed=style["closed"],
                        color=style["color"],
                        thickness=style["thickness"],
                        lineType=cv2.LINE_AA
                    )

            # 고휘도 앵커 포인트 강조 (외곽 옐로우 링 + 내부 화이트 코어)
            for kp in KEY_ANCHOR_POINTS:
                if kp < len(landmarks):
                    pt = landmarks[kp]
                    cv2.circle(out_frame, pt, 4, (0, 240, 255), 1, cv2.LINE_AA)
                    cv2.circle(out_frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

            # 동공(홍채) 중심 랜드마크 (468, 473)
            if len(landmarks) >= 478:
                for iris_idx in [468, 473]:
                    if iris_idx < len(landmarks):
                        ipt = landmarks[iris_idx]
                        cv2.circle(out_frame, ipt, 4, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(out_frame, ipt, 6, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. PIL 한글 텍스트 및 정보 카드 오버레이
        rgb_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # 상단 헤더 바 (높이 44px, 슬림 다크 바)
        header_h = 44
        draw.rectangle([(0, 0), (w, header_h)], fill=(15, 20, 30, 215))
        draw.rectangle([(0, header_h - 2), (w, header_h)], fill=accent_color_rgb)

        font_header = self.get_font(16)
        font_sub = self.get_font(12)
        font_msg = self.get_font(13)
        font_small = self.get_font(12)

        if analysis:
            dom_ko = clean_korean_text(analysis['dominant_ko'])
            conf = analysis['confidence']
            weather_ko = clean_korean_text(analysis['weather'])
            symmetry = analysis.get('symmetry', 95.0)

            # 상단 제목 (깔끔한 슬림 1줄 텍스트)
            header_text = f"상태: {dom_ko}   |   일치도: {conf:.0f}%   |   안면 대칭: {symmetry:.0f}%   |   컨디션: {weather_ko}"
            draw.text((16, 12), header_text, font=font_header, fill=(255, 255, 255))

            # 3. 우측 표정 세부 분석 카드 (컴팩트: 너비 190px, 높이 88px)
            scores = analysis.get('scores', {})
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]

            card_w = 190
            card_h = 88
            card_x = w - card_w - 12
            card_y = header_h + 8

            draw.rectangle([(card_x, card_y), (card_x + card_w, card_y + card_h)],
                           fill=(18, 24, 36, 195), outline=(75, 85, 105), width=1)
            draw.text((card_x + 10, card_y + 6), "표정 세부 분석", font=font_sub, fill=(215, 225, 240))

            row_y = card_y + 26
            for emo_key, score_val in sorted_scores:
                cfg_item = EMOTION_CONFIG.get(emo_key, {})
                short_name = cfg_item.get('short', emo_key)
                emo_color = cfg_item.get('color_rgb', (200, 200, 200))

                draw.text((card_x + 10, row_y), short_name, font=font_small, fill=(230, 230, 230))

                bar_x1 = card_x + 75
                bar_x2 = bar_x1 + 65
                draw.rectangle([(bar_x1, row_y + 3), (bar_x2, row_y + 11)], fill=(45, 55, 70))
                fill_w = int(65 * (max(0.0, min(100.0, score_val)) / 100.0))
                if fill_w > 0:
                    draw.rectangle([(bar_x1, row_y + 3), (bar_x1 + fill_w, row_y + 11)], fill=emo_color)

                draw.text((bar_x2 + 6, row_y), f"{score_val:.0f}%", font=font_small, fill=(200, 215, 230))
                row_y += 19

            # 4. 좌측 상단: 미세 바이오마커 & 마이크 음성 활력 게이지 카드 (너비 180px, 높이 88px)
            bio_w = 180
            bio_h = 88
            bio_x = 12
            bio_y = header_h + 8
            draw.rectangle([(bio_x, bio_y), (bio_x + bio_w, bio_y + bio_h)],
                           fill=(18, 24, 36, 195), outline=(75, 85, 105), width=1)
            draw.text((bio_x + 10, bio_y + 6), "미세 생체신호 분석", font=font_sub, fill=(200, 220, 245))

            mouth_d = analysis.get('mouth_dynamic', 30.0)
            eye_f = analysis.get('eye_fatigue', 20.0)
            audio_info = self.audio_analyzer.get_metrics()
            vol = audio_info.get('volume', 0.0)

            # 입술 반응도 바
            draw.text((bio_x + 10, bio_y + 26), "입술 활력", font=font_small, fill=(210, 210, 210))
            draw.rectangle([(bio_x + 65, bio_y + 29), (bio_x + 135, bio_y + 37)], fill=(45, 55, 70))
            draw.rectangle([(bio_x + 65, bio_y + 29), (bio_x + 65 + int(70 * (mouth_d / 100.0)), bio_y + 37)], fill=(46, 204, 113))
            draw.text((bio_x + 140, bio_y + 26), f"{mouth_d:.0f}%", font=font_small, fill=(190, 205, 220))

            # 눈가 피로도 바
            draw.text((bio_x + 10, bio_y + 45), "눈가 피로", font=font_small, fill=(210, 210, 210))
            draw.rectangle([(bio_x + 65, bio_y + 48), (bio_x + 135, bio_y + 56)], fill=(45, 55, 70))
            draw.rectangle([(bio_x + 65, bio_y + 48), (bio_x + 65 + int(70 * (eye_f / 100.0)), bio_y + 56)], fill=(231, 76, 60))
            draw.text((bio_x + 140, bio_y + 45), f"{eye_f:.0f}%", font=font_small, fill=(190, 205, 220))

            # 마이크 음성 성량 바
            draw.text((bio_x + 10, bio_y + 64), "음성 성량", font=font_small, fill=(210, 210, 210))
            draw.rectangle([(bio_x + 65, bio_y + 67), (bio_x + 135, bio_y + 75)], fill=(45, 55, 70))
            mic_fill = int(70 * (min(100.0, vol) / 100.0))
            mic_color = (46, 204, 113) if vol > 12.0 else (100, 110, 130)
            if mic_fill > 0:
                draw.rectangle([(bio_x + 65, bio_y + 67), (bio_x + 65 + mic_fill, bio_y + 75)], fill=mic_color)
            mic_label = "발화" if vol > 12.0 else "대기"
            draw.text((bio_x + 140, bio_y + 64), mic_label, font=font_small, fill=(170, 220, 190) if vol > 12.0 else (150, 160, 170))

            care_msg = clean_korean_text(analysis.get('care_message', ""))
        else:
            draw.text((16, 12), "생체신호 탐색 중입니다...", font=font_header, fill=(230, 230, 230))
            care_msg = "카메라를 정면으로 바라보며 편안하게 응답해 주세요."

        # 5. 하단 안내 배너 (높이 38px, 슬림 바)
        banner_h = 38
        draw.rectangle([(0, h - banner_h), (w, h)], fill=(14, 18, 28, 220))
        draw.rectangle([(0, h - banner_h), (w, h - banner_h + 2)], fill=accent_color_rgb)
        draw.text((16, h - banner_h + 10), care_msg, font=font_msg, fill=(255, 255, 240))

        out_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return out_frame
