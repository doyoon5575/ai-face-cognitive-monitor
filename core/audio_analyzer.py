"""
실시간 음성 바이오마커 분석 모듈 (core/audio_analyzer.py)
- 노트북 마이크를 통한 실시간 음성 에너지(RMS/성량) 캡처
- 주파수 변동성 및 목소리 활력도(Vitality Score) 측정
- 발화 상태 감지 (활기참, 보통, 가라앉음, 침묵)
"""

import threading
import time
import numpy as np
from typing import Dict, Any, Optional


class RealtimeAudioAnalyzer:
    """실시간 마이크 음성 바이오마커 분석기"""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._stream = None
        self._is_running = False
        self._lock = threading.Lock()

        # 실시간 측정 지표 (스레드 안전)
        self.current_volume_rms = 0.0     # 0.0 ~ 100.0
        self.is_speaking = False
        self.vitality_score = 50.0        # 0.0 ~ 100.0
        self.tone_state = "침묵"
        self._recent_energies = []

        # 백그라운드 캡처 시작
        self.start()

    def _audio_callback(self, indata, frames, time_info, status):
        """마이크 버퍼 콜백 함수"""
        if status:
            pass

        # 모노 오디오 추출
        audio_data = indata[:, 0] if indata.ndim > 1 else indata

        # 1. RMS 성량 (음량 에너지) 계산
        rms = np.sqrt(np.mean(audio_data ** 2))
        # 0.0 ~ 100.0 스케일로 정규화
        norm_volume = float(np.clip(rms * 400.0, 0.0, 100.0))

        # 2. 발화 여부 판정 (노이즈 임계치 8.0 이상)
        is_talking = norm_volume > 12.0

        # 3. 음성 활력도 (에너지 변동성 및 주파수 특성)
        with self._lock:
            self.current_volume_rms = norm_volume
            self.is_speaking = is_talking

            self._recent_energies.append(norm_volume)
            if len(self._recent_energies) > 20:
                self._recent_energies.pop(0)

            # 발화 중일 때 활력도 분석
            if is_talking and len(self._recent_energies) >= 5:
                vol_std = np.std(self._recent_energies)
                vol_mean = np.mean(self._recent_energies)

                # 목소리 톤에 억양과 변화가 있고 성량이 풍부할수록 활력도 증가
                calculated_vitality = float(np.clip(vol_mean * 1.2 + vol_std * 2.0, 30.0, 95.0))
                self.vitality_score = calculated_vitality

                if calculated_vitality >= 70.0:
                    self.tone_state = "활기참"
                elif calculated_vitality >= 45.0:
                    self.tone_state = "보통/안정"
                else:
                    self.tone_state = "가라앉음"
            elif not is_talking:
                self.tone_state = "침묵"

    def start(self) -> None:
        """마이크 캡처 스트림 시작"""
        if self._is_running:
            return

        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="float32",
                callback=self._audio_callback
            )
            self._stream.start()
            self._is_running = True
            print("[AudioAnalyzer] 마이크 실시간 음성 분석 스트림 시작 완료!")
        except Exception as e:
            print(f"[AudioAnalyzer] 마이크 시작 실패 (폴백 모드 동작): {e}")
            self._is_running = False

    def get_metrics(self) -> Dict[str, Any]:
        """현재 실시간 음성 바이오마커 지표 반환"""
        with self._lock:
            return {
                "volume": round(self.current_volume_rms, 1),
                "is_speaking": self.is_speaking,
                "vitality": round(self.vitality_score, 1),
                "tone_state": self.tone_state,
                "is_active": self._is_running
            }

    def stop(self) -> None:
        """마이크 캡처 정지 및 자원 해제"""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._is_running = False
