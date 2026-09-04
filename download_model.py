"""
Google MediaPipe Face Landmarker 모델 자동 다운로더
"""

import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"


def download_face_landmarker_model(target_path: Path = MODEL_PATH, force: bool = False) -> Path:
    """MediaPipe face_landmarker.task 모델을 다운로드합니다."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        print(f"[INFO] Face Landmarker 모델이 이미 존재합니다: {target_path}")
        return target_path

    print(f"[INFO] Google Storage에서 Face Landmarker 모델 다운로드 중...")
    print(f"       URL: {MODEL_URL}")
    print(f"       저장 경로: {target_path}")

    def progress_callback(blocks_transferred, block_size, total_size):
        if total_size > 0:
            percent = (blocks_transferred * block_size * 100) / total_size
            downloaded_mb = (blocks_transferred * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r[DOWNLOAD] {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({percent:.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(MODEL_URL, str(target_path), reporthook=progress_callback)
    print("\n[SUCCESS] Face Landmarker 모델 다운로드 완료!")
    return target_path


if __name__ == "__main__":
    download_face_landmarker_model()
