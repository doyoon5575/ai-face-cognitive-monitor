"""
SQLite 기반 실시간 정서 시계열 데이터베이스 핸들러
- affect_logs.db 테이블 관리
- I/O 최적화를 위한 5초 단위(또는 N개) 배치 버퍼링(Batch Insertion)
- 실시간 및 일별/주별 집계 통계 쿼리
- 정서 둔마 위험 스크리닝 알림 로직
"""

import sys
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass



class AffectDBHandler:
    def __init__(self, db_path: Optional[str] = None, batch_interval_sec: float = 5.0, batch_max_size: int = 150):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.db_path = str(base_dir / "affect_logs.db")
        else:
            self.db_path = str(db_path)

        self.batch_interval_sec = batch_interval_sec
        self.batch_max_size = batch_max_size
        self._buffer: List[Tuple[Any, ...]] = []
        self._lock = threading.Lock()
        self._last_flush_time = time.time()

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """데이터베이스 테이블 및 인덱스 초기화"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS affect_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    smile_score REAL NOT NULL,
                    frown_score REAL NOT NULL,
                    flatness_score REAL NOT NULL,
                    blink_detected INTEGER NOT NULL DEFAULT 0,
                    head_pose_valid INTEGER NOT NULL DEFAULT 1,
                    yaw REAL DEFAULT 0.0,
                    pitch REAL DEFAULT 0.0,
                    roll REAL DEFAULT 0.0
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_affect_logs_timestamp 
                ON affect_logs (timestamp)
            """)
            conn.commit()

    def add_record(
        self,
        smile_score: float,
        frown_score: float,
        flatness_score: float,
        blink_detected: bool,
        head_pose_valid: bool = True,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        timestamp: Optional[str] = None,
        force_flush: bool = False
    ) -> None:
        """
        메모리 버퍼에 레코드를 추가하고 배치 조건 충족 시 SQLite에 일괄 삽입합니다.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        record = (
            timestamp,
            round(float(smile_score), 4),
            round(float(frown_score), 4),
            round(float(flatness_score), 4),
            1 if blink_detected else 0,
            1 if head_pose_valid else 0,
            round(float(yaw), 2),
            round(float(pitch), 2),
            round(float(roll), 2),
        )

        should_flush = False
        with self._lock:
            self._buffer.append(record)
            now = time.time()
            if force_flush or len(self._buffer) >= self.batch_max_size or (now - self._last_flush_time >= self.batch_interval_sec):
                should_flush = True

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """버퍼에 쌓인 데이터를 DB로 일괄 커밋"""
        with self._lock:
            if not self._buffer:
                return
            records_to_insert = list(self._buffer)
            self._buffer.clear()
            self._last_flush_time = time.time()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO affect_logs (
                    timestamp, smile_score, frown_score, flatness_score,
                    blink_detected, head_pose_valid, yaw, pitch, roll
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records_to_insert)
            conn.commit()

    def get_recent_logs(self, limit_seconds: int = 120) -> pd.DataFrame:
        """최근 N초간의 시계열 로그 조회"""
        self.flush()
        cutoff = (datetime.now() - timedelta(seconds=limit_seconds)).isoformat()
        query = """
            SELECT timestamp, smile_score, frown_score, flatness_score, blink_detected, head_pose_valid, yaw, pitch, roll
            FROM affect_logs
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(cutoff,))
        return df

    def get_logs_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """지정 기간 동안의 로그 조회"""
        self.flush()
        query = """
            SELECT timestamp, smile_score, frown_score, flatness_score, blink_detected, head_pose_valid, yaw, pitch, roll
            FROM affect_logs
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(start_date.isoformat(), end_date.isoformat()))
        return df

    def get_daily_summary(self, days: int = 7) -> pd.DataFrame:
        """
        최근 N일간 일별 통계 계산:
        - 일평균 Flatness (평균 정서둔마)
        - 감정 가변성 (Emotional Variability = std dev of smile & frown)
        - 스마일 피크 빈도 (Smile Peak Rate: smile >= 0.45 횟수)
        - 눈 깜빡임 빈도 (Total Blinks)
        - 총 수집 프레임 수
        """
        self.flush()
        start_cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
        query = """
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_frames,
                AVG(flatness_score) as avg_flatness,
                AVG(smile_score) as avg_smile,
                AVG(frown_score) as avg_frown,
                SUM(blink_detected) as total_blinks,
                SUM(CASE WHEN smile_score >= 0.40 THEN 1 ELSE 0 END) as smile_peaks,
                SUM(CASE WHEN head_pose_valid = 1 THEN 1 ELSE 0 END) as valid_frames
            FROM affect_logs
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(start_cutoff,))

        if df.empty:
            return pd.DataFrame(columns=[
                "date", "total_frames", "avg_flatness", "avg_smile", 
                "avg_frown", "total_blinks", "smile_peaks", "emotional_variability", "smile_peak_pct"
            ])

        # 감정 가변성(Emotional Variability) 계산을 위해 전체 데이터를 가져와 날짜별 표준편차 계산
        raw_query = """
            SELECT DATE(timestamp) as date, smile_score, frown_score, flatness_score
            FROM affect_logs
            WHERE timestamp >= ?
        """
        with self._get_connection() as conn:
            raw_df = pd.read_sql_query(raw_query, conn, params=(start_cutoff,))

        if not raw_df.empty:
            var_df = raw_df.groupby("date").agg(
                smile_std=("smile_score", "std"),
                frown_std=("frown_score", "std"),
                flatness_std=("flatness_score", "std")
            ).reset_index()
            # 종합 가변성 = (smile_std + frown_std) / 2
            var_df["emotional_variability"] = (var_df["smile_std"].fillna(0) + var_df["frown_std"].fillna(0)) / 2.0
            df = pd.merge(df, var_df[["date", "emotional_variability", "flatness_std"]], on="date", how="left")
        else:
            df["emotional_variability"] = 0.0
            df["flatness_std"] = 0.0

        df["smile_peak_pct"] = (df["smile_peaks"] / df["total_frames"].replace(0, 1)) * 100.0
        return df

    def check_screening_alert(self, threshold_flatness: float = 0.85, consecutive_days: int = 3) -> Dict[str, Any]:
        """
        스크리닝 알림: 연속 N일 동안 일평균 Flatness가 threshold를 초과했는지 검사
        """
        summary_df = self.get_daily_summary(days=14)
        if len(summary_df) < consecutive_days:
            return {
                "alert_triggered": False,
                "consecutive_count": 0,
                "message": f"데이터 누적 기간 부족 (최소 {consecutive_days}일 필요, 현재 {len(summary_df)}일 수집됨)",
                "recent_daily_flatness": summary_df["avg_flatness"].tolist() if not summary_df.empty else []
            }

        # 최근 날짜부터 역순으로 검사
        recent_records = summary_df.sort_values("date", ascending=False).head(consecutive_days)
        high_flatness_count = 0
        dates_triggered = []

        for _, row in recent_records.iterrows():
            if row["avg_flatness"] >= threshold_flatness:
                high_flatness_count += 1
                dates_triggered.append((row["date"], row["avg_flatness"]))
            else:
                break

        triggered = (high_flatness_count >= consecutive_days)
        return {
            "alert_triggered": triggered,
            "consecutive_count": high_flatness_count,
            "required_consecutive": consecutive_days,
            "threshold": threshold_flatness,
            "dates_triggered": dates_triggered,
            "message": (
                f"⚠️ [주의] 최근 {consecutive_days}일 연속 일평균 정서 둔마 지표({threshold_flatness} 이상)가 감지되었습니다. 전문의 상담 또는 정서 완화 관리를 권장합니다."
                if triggered else "정상 범위 유지 중입니다."
            )
        }

    def seed_mock_data(self, days: int = 7, trigger_alert: bool = False) -> None:
        """대시보드 테스트 및 시뮬레이션을 위한 샘플 데이터 생성"""
        import random
        records = []
        now = datetime.now()

        for day_offset in range(days, -1, -1):
            day_date = now - timedelta(days=day_offset)
            # 하루 120 프레임씩 생성
            is_flat_day = trigger_alert and (day_offset in [0, 1, 2])

            for minute in range(0, 120, 2):
                timestamp = (day_date.replace(hour=9, minute=0, second=0) + timedelta(minutes=minute)).isoformat()
                if is_flat_day:
                    # 정서 둔마 상태 (스마일 거의 없음, 무표정 높음)
                    smile = random.uniform(0.0, 0.05)
                    frown = random.uniform(0.0, 0.05)
                    flatness = round(1.0 - min(1.0, (smile + frown) * 1.5), 4)
                else:
                    # 일반 상태 (자연스러운 표정 변화)
                    rand_state = random.random()
                    if rand_state > 0.8:
                        smile = random.uniform(0.4, 0.85)
                        frown = random.uniform(0.0, 0.1)
                    elif rand_state > 0.65:
                        smile = random.uniform(0.0, 0.1)
                        frown = random.uniform(0.3, 0.7)
                    else:
                        smile = random.uniform(0.05, 0.2)
                        frown = random.uniform(0.05, 0.2)
                    flatness = round(1.0 - min(1.0, (smile * 1.5) + (frown * 1.5)), 4)

                blink = 1 if random.random() < 0.2 else 0
                yaw = random.uniform(-10.0, 10.0)
                pitch = random.uniform(-8.0, 8.0)
                roll = random.uniform(-5.0, 5.0)

                records.append((
                    timestamp,
                    round(smile, 4),
                    round(frown, 4),
                    flatness,
                    blink,
                    1,
                    round(yaw, 2),
                    round(pitch, 2),
                    round(roll, 2)
                ))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO affect_logs (
                    timestamp, smile_score, frown_score, flatness_score,
                    blink_detected, head_pose_valid, yaw, pitch, roll
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
        print(f"[SUCCESS] {len(records)}개의 시뮬레이션 데이터가 DB에 적재되었습니다.")


class SeniorAffectDBHandler:
    """
    시니어 맞춤형 실시간 감정 시계열 데이터베이스 핸들러
    - senior_affect_logs 테이블 관리
    - 7대 감정 확률(%) 및 주 감정, 따뜻한 안부 메시지 저장
    - 오늘의 마음 날씨(맑음/흐림/비) 집계 및 통계 쿼리
    """
    def __init__(self, db_path: Optional[str] = None, batch_interval_sec: float = 3.0, batch_max_size: int = 100):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.db_path = str(base_dir / "affect_logs.db")
        else:
            self.db_path = str(db_path)

        self.batch_interval_sec = batch_interval_sec
        self.batch_max_size = batch_max_size
        self._buffer: List[Tuple[Any, ...]] = []
        self._lock = threading.Lock()
        self._last_flush_time = time.time()

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """시니어 감정 로그 테이블 초기화"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS senior_affect_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dominant_emotion TEXT NOT NULL,
                    emotion_ko TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    happy_score REAL DEFAULT 0.0,
                    neutral_score REAL DEFAULT 0.0,
                    sad_score REAL DEFAULT 0.0,
                    angry_score REAL DEFAULT 0.0,
                    surprise_score REAL DEFAULT 0.0,
                    fear_score REAL DEFAULT 0.0,
                    disgust_score REAL DEFAULT 0.0,
                    care_message TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_senior_logs_timestamp 
                ON senior_affect_logs (timestamp)
            """)
            conn.commit()

    def add_record(
        self,
        dominant_emotion: str,
        emotion_ko: str,
        confidence: float,
        scores: Dict[str, float],
        care_message: str = "",
        timestamp: Optional[str] = None,
        force_flush: bool = False
    ) -> None:
        """메모리 버퍼에 시니어 감정 레코드를 추가하고 배치 주기에 맞춰 SQLite에 저장합니다."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        record = (
            timestamp,
            dominant_emotion,
            emotion_ko,
            round(float(confidence), 2),
            round(float(scores.get("happy", 0.0)), 2),
            round(float(scores.get("neutral", 0.0)), 2),
            round(float(scores.get("sad", 0.0)), 2),
            round(float(scores.get("angry", 0.0)), 2),
            round(float(scores.get("surprise", 0.0)), 2),
            round(float(scores.get("fear", 0.0)), 2),
            round(float(scores.get("disgust", 0.0)), 2),
            care_message
        )

        should_flush = False
        with self._lock:
            self._buffer.append(record)
            now = time.time()
            if force_flush or len(self._buffer) >= self.batch_max_size or (now - self._last_flush_time >= self.batch_interval_sec):
                should_flush = True

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """버퍼에 쌓인 데이터를 일괄 삽입"""
        with self._lock:
            if not self._buffer:
                return
            to_insert = list(self._buffer)
            self._buffer.clear()
            self._last_flush_time = time.time()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO senior_affect_logs (
                        timestamp, dominant_emotion, emotion_ko, confidence,
                        happy_score, neutral_score, sad_score, angry_score,
                        surprise_score, fear_score, disgust_score, care_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, to_insert)
                conn.commit()
        except Exception as e:
            print(f"[SeniorDB Error] Flush failed: {e}")
            with self._lock:
                self._buffer = to_insert + self._buffer

    def get_recent_logs(self, limit: int = 60) -> pd.DataFrame:
        """최근 감정 로그 조회"""
        query = """
            SELECT timestamp, dominant_emotion, emotion_ko, confidence,
                   happy_score, neutral_score, sad_score, angry_score,
                   surprise_score, fear_score, disgust_score, care_message
            FROM senior_affect_logs
            ORDER BY id DESC
            LIMIT ?
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(limit,))
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def get_daily_weather_summary(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        특정 일자의 감정을 집계하여 '오늘의 마음 날씨' 및 요약 통계를 반환합니다.
        - ☀️ 맑음: 행복/웃음 비율이 높거나 긍정 정서 우세
        - 🌤️ 화창: 편안함/평온함이 지배적
        - ⛅ 구름: 복합적이거나 평온+약간의 긴장
        - 🌧️ 비: 슬픔/울적함 비율이 지속적으로 높음
        - ⛈️ 천둥: 화남/답답함 비율이 높음
        """
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")

        query = """
            SELECT dominant_emotion, emotion_ko, happy_score, neutral_score, sad_score, angry_score, surprise_score, fear_score, disgust_score
            FROM senior_affect_logs
            WHERE date(timestamp) = date(?)
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(target_date,))

        if df.empty:
            return {
                "date": target_date,
                "total_records": 0,
                "weather": "자료 없음 🌫️",
                "weather_desc": "측정된 표정 기록이 아직 없습니다.",
                "dominant_state": "기록 없음",
                "happy_ratio": 0.0,
                "neutral_ratio": 0.0,
                "negative_ratio": 0.0,
                "smile_count": 0
            }

        total = len(df)
        happy_count = (df['dominant_emotion'] == 'happy').sum()
        neutral_count = (df['dominant_emotion'] == 'neutral').sum()
        sad_count = (df['dominant_emotion'] == 'sad').sum()
        angry_count = (df['dominant_emotion'] == 'angry').sum()
        fear_count = (df['dominant_emotion'] == 'fear').sum()

        happy_pct = (happy_count / total) * 100
        neutral_pct = (neutral_count / total) * 100
        negative_pct = ((sad_count + angry_count + fear_count) / total) * 100

        # 마음 날씨 판정 로직
        if happy_pct >= 35.0:
            weather = "맑음 ☀️"
            weather_desc = "어르신의 얼굴에 활짝 웃음꽃이 가득한 행복한 하루입니다!"
        elif neutral_pct >= 60.0 or (happy_pct + neutral_pct >= 75.0):
            weather = "화창 🌤️"
            weather_desc = "마음이 안정되고 평온한 하루를 보내셨습니다."
        elif (sad_count / total) >= 0.25:
            weather = "비 🌧️"
            weather_desc = "마음이 조금 울적하거나 적적한 시간이 있었습니다. 따뜻한 관심이 필요합니다."
        elif (angry_count / total) >= 0.20:
            weather = "천둥 ⛈️"
            weather_desc = "마음이 답답하거나 속상한 순간이 자주 감지되었습니다."
        else:
            weather = "구름 조금 ⛅"
            weather_desc = "자연스러운 감정 변화와 함께 하루를 보내셨습니다."

        dominant_state = df['emotion_ko'].mode()[0] if not df['emotion_ko'].empty else "평온"

        return {
            "date": target_date,
            "total_records": total,
            "weather": weather,
            "weather_desc": weather_desc,
            "dominant_state": dominant_state,
            "happy_ratio": round(happy_pct, 1),
            "neutral_ratio": round(neutral_pct, 1),
            "negative_ratio": round(negative_pct, 1),
            "smile_count": int(happy_count)
        }

    def get_weekly_stats(self, days: int = 7) -> pd.DataFrame:
        """최근 N일간 일자별 감정 통계 조회"""
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = """
            SELECT date(timestamp) as log_date,
                   COUNT(*) as total_count,
                   AVG(happy_score) as avg_happy,
                   AVG(neutral_score) as avg_neutral,
                   AVG(sad_score) as avg_sad,
                   SUM(CASE WHEN dominant_emotion = 'happy' THEN 1 ELSE 0 END) as smile_moments
            FROM senior_affect_logs
            WHERE date(timestamp) >= date(?)
            GROUP BY date(timestamp)
            ORDER BY log_date ASC
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(start_date,))
        return df

    def populate_mock_data(self, days: int = 5) -> None:
        """시니어 시연 및 테스트용 모의 데이터 생성"""
        import random
        now = datetime.now()
        records = []
        emotions_pool = [
            ('happy', '활짝 웃음 😊', "어르신의 밝은 미소가 참 보기 좋습니다! ☀️"),
            ('neutral', '편안하고 평온함 😌', "마음이 편안하고 차분해 보이세요. 🌿"),
            ('neutral', '편안하고 평온함 😌', "오늘도 평온하고 기분 좋은 하루 보내세요!"),
            ('sad', '마음이 울적함 😢', "따뜻한 차 한 잔 드시며 좋아하는 음악을 들어보세요 ☕"),
            ('surprise', '깜짝 놀람 😲', "호기심 가득한 눈빛이 활기차 보입니다!"),
        ]

        for d in range(days, -1, -1):
            day_time = now - timedelta(days=d)
            for m in range(0, 100, 2):
                t_str = (day_time.replace(hour=10, minute=0, second=0) + timedelta(minutes=m)).isoformat()
                emo, emo_ko, msg = random.choice(emotions_pool)
                # 점수 생성
                scores = {k: random.uniform(2.0, 15.0) for k in ['happy', 'neutral', 'sad', 'angry', 'surprise', 'fear', 'disgust']}
                scores[emo] = random.uniform(65.0, 95.0)
                conf = scores[emo]

                records.append((
                    t_str, emo, emo_ko, round(conf, 1),
                    round(scores['happy'], 1), round(scores['neutral'], 1),
                    round(scores['sad'], 1), round(scores['angry'], 1),
                    round(scores['surprise'], 1), round(scores['fear'], 1),
                    round(scores['disgust'], 1), msg
                ))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO senior_affect_logs (
                    timestamp, dominant_emotion, emotion_ko, confidence,
                    happy_score, neutral_score, sad_score, angry_score,
                    surprise_score, fear_score, disgust_score, care_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
        print(f"[SeniorDB] {len(records)}개의 모의 시니어 데이터가 성공적으로 생성되었습니다.")

