"""
MediaPipe 52종 Blendshape 기반 감정 및 정서 바이오마커 분석 모듈
- Smile Score (웃음 지수)
- Frown/Sad Score (부정/찡그림 지수)
- Flatness Score (무표정/정서 둔마 지표)
- Blink Score & Event Detection (눈 깜빡임)
- 3D Head Pose Estimation (SolvePnP 기반 Yaw, Pitch, Roll 각도 필터링)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    is_valid: bool  # 35도 이내 여부


@dataclass
class AffectMetrics:
    smile_score: float
    frown_score: float
    flatness_score: float
    blink_score: float
    blink_detected: bool
    head_pose: HeadPose
    blendshapes_raw: Dict[str, float]


class FacialAffectAnalyzer:
    """
    MediaPipe Face Blendshape 카테고리 결과 및 Landmark를 입력받아
    감정 지표 및 유효성을 계산하는 클래스
    """

    # 3D 표준 얼굴 모델 점 좌표 (단위: mm 임의 기준)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),          # 코끝 (Nose tip) - 1
        (0.0, -330.0, -65.0),     # 턱 (Chin) - 152
        (-225.0, 170.0, -135.0),  # 좌측 눈 외곽 (Left eye outer corner) - 33
        (225.0, 170.0, -135.0),   # 우측 눈 외곽 (Right eye outer corner) - 263
        (-150.0, -150.0, -125.0), # 입 좌측 끝 (Left Mouth corner) - 61
        (150.0, -150.0, -125.0)   # 입 우측 끝 (Right mouth corner) - 291
    ], dtype=np.float64)

    # 랜드마크 인덱스 매핑 (MediaPipe Face Mesh 468/478 기준)
    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(
        self,
        max_rotation_degrees: float = 35.0,
        blink_threshold: float = 0.45
    ):
        self.max_rotation_degrees = max_rotation_degrees
        self.blink_threshold = blink_threshold

        # 눈 깜빡임 이산 이벤트 감지를 위한 상태 변수
        self._is_eye_closed = False

    def parse_blendshapes(self, blendshapes_category_list: Any) -> Dict[str, float]:
        """MediaPipe Blendshapes 카테고리 리스트를 dict 형태로 변환"""
        blendshape_dict: Dict[str, float] = {}
        if not blendshapes_category_list:
            return blendshape_dict

        # blendshapes_category_list는 Categories 객체들의 리스트
        for category in blendshapes_category_list:
            name = category.category_name
            score = float(category.score)
            blendshape_dict[name] = score

        return blendshape_dict

    def estimate_head_pose(
        self,
        landmarks: List[Any],
        image_width: int,
        image_height: int
    ) -> HeadPose:
        """
        OpenCV solvePnP를 사용하여 얼굴의 3D 방향(Yaw, Pitch, Roll) 각도를 추정합니다.
        """
        if not landmarks or len(landmarks) < max(self.LANDMARK_INDICES):
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, is_valid=True)

        image_points = []
        for idx in self.LANDMARK_INDICES:
            lm = landmarks[idx]
            x = lm.x * image_width
            y = lm.y * image_height
            image_points.append([x, y])

        image_points_2d = np.array(image_points, dtype=np.float64)

        # 카메라 내부 파라미터 추정 (초점거리 = 이미지 폭 기준)
        focal_length = image_width
        center = (image_width / 2, image_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, is_valid=True)

        # 회전 벡터 -> 회전 행렬 -> 오일러 각 변환
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        # decomposeProjectionMatrix 반환 각도 (도 단위)
        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        is_valid = (
            abs(yaw) <= self.max_rotation_degrees and
            abs(pitch) <= self.max_rotation_degrees and
            abs(roll) <= self.max_rotation_degrees
        )

        return HeadPose(
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
            is_valid=is_valid
        )

    def analyze(
        self,
        blendshapes_category_list: Any,
        landmarks: Optional[List[Any]] = None,
        image_width: int = 640,
        image_height: int = 480
    ) -> AffectMetrics:
        """
        Face Blendshape 계수를 바탕으로 핵심 정서 바이오마커를 연산합니다.
        """
        shapes = self.parse_blendshapes(blendshapes_category_list)

        # 1. 원본 블렌드셰이프 값 추출 (기본값 0.0)
        mouth_smile_left = shapes.get("mouthSmileLeft", 0.0)
        mouth_smile_right = shapes.get("mouthSmileRight", 0.0)

        mouth_frown_left = shapes.get("mouthFrownLeft", 0.0)
        mouth_frown_right = shapes.get("mouthFrownRight", 0.0)
        brow_down_left = shapes.get("browDownLeft", 0.0)
        brow_down_right = shapes.get("browDownRight", 0.0)

        eye_blink_left = shapes.get("eyeBlinkLeft", 0.0)
        eye_blink_right = shapes.get("eyeBlinkRight", 0.0)

        # 2. 핵심 정서 수식 계산
        # Smile = (mouthSmileLeft + mouthSmileRight) / 2
        smile_score = (mouth_smile_left + mouth_smile_right) / 2.0

        # Frown = (mouthFrownLeft + mouthFrownRight + browDownLeft + browDownRight) / 4
        frown_score = (mouth_frown_left + mouth_frown_right + brow_down_left + brow_down_right) / 4.0

        # Flatness = 1.0 - min(1.0, Smile * 1.5 + Frown * 1.5)
        expressed_affect = (smile_score * 1.5) + (frown_score * 1.5)
        flatness_score = 1.0 - min(1.0, expressed_affect)

        # Blink = (eyeBlinkLeft + eyeBlinkRight) / 2
        blink_score = (eye_blink_left + eye_blink_right) / 2.0

        # 눈 깜빡임 이산 이벤트 판별
        blink_detected = False
        if blink_score >= self.blink_threshold:
            if not self._is_eye_closed:
                self._is_eye_closed = True
                blink_detected = True
        else:
            self._is_eye_closed = False

        # 3. 헤드 포즈 추정
        if landmarks:
            head_pose = self.estimate_head_pose(landmarks, image_width, image_height)
        else:
            head_pose = HeadPose(yaw=0.0, pitch=0.0, roll=0.0, is_valid=True)

        return AffectMetrics(
            smile_score=round(float(smile_score), 4),
            frown_score=round(float(frown_score), 4),
            flatness_score=round(float(flatness_score), 4),
            blink_score=round(float(blink_score), 4),
            blink_detected=blink_detected,
            head_pose=head_pose,
            blendshapes_raw=shapes
        )
