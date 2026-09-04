"""
시니어(어르신) 두뇌 인지 & 컨디션 종합 자가진단 평가 모듈 (core/senior_assessment.py)
- K-MMSE / 치매안심센터 선별 기준 기반 간이 인지 퀴즈 (지남력, 계산력, 언어/상식, 기억력)
- 실시간 안면 표정(집중도/긴장도)과 결합한 종합 컨디션 판정 알고리즘
- 맞춤형 권장 가이드 (두뇌 학습 권장, 피로 휴식, 전문 상담 안내)
"""

from typing import List, Dict, Any, Tuple
from datetime import datetime


class SeniorCognitiveAssessment:
    """시니어 두뇌 건강 및 컨디션 평가 클래스"""

    # 평가 퀴즈 세트 (친숙하고 직관적인 5문항)
    QUESTIONS = [
        {
            "id": 1,
            "category": "지남력 (시간/날짜)",
            "question": "지금은 일 년 중 어느 계절인가요?",
            "options": ["봄", "여름", "가을", "겨울"],
            "answer": "가을",  # 9월 기준 가을
            "points": 20,
            "tip": "현재 계절을 인지하고 계신지 확인합니다."
        },
        {
            "id": 2,
            "category": "수학적 계산력",
            "question": "100원에서 7원을 빼면 얼마가 남을까요? (100 - 7 = ?)",
            "options": ["91원", "92원", "93원", "94원"],
            "answer": "93원",
            "points": 20,
            "tip": "기본적인 일상 암산 및 뺄셈 집중력을 확인합니다."
        },
        {
            "id": 3,
            "category": "언어 및 범주화",
            "question": "다음 네 가지 중 성격이 다른 하나는 무엇인가요?",
            "options": ["사과", "바나나", "책상", "포도"],
            "answer": "책상",
            "points": 20,
            "tip": "단어의 공통 범주(과일)를 구별하는 어휘 판단력입니다."
        },
        {
            "id": 4,
            "category": "상식 및 속담",
            "question": "‘가는 말이 고와야 (    )이 곱다’에 들어갈 알맞은 말은?",
            "options": ["오는 말", "가는 길", "웃는 낯", "마음씨"],
            "answer": "오는 말",
            "points": 20,
            "tip": "오랜 기억 속에 저장된 친숙한 속담 완성 능력을 봅니다."
        },
        {
            "id": 5,
            "category": "기억력 회상",
            "question": "조금 전 안내해 드린 세 가지 단어 [나무, 자전거, 모자] 중 포함되지 않은 것은?",
            "options": ["나무", "자전거", "비행기", "모자"],
            "answer": "비행기",
            "points": 20,
            "tip": "단기 기억을 유지하고 올바르게 회상하는지 확인합니다."
        }
    ]

    @classmethod
    def evaluate(
        cls,
        user_answers: Dict[int, str],
        dominant_emotion: str = "neutral",
        confidence: float = 80.0
    ) -> Dict[str, Any]:
        """
        사용자 응답과 표정 상태를 종합 평가하여 리포트 생성
        """
        total_score = 0
        details = []

        for q in cls.QUESTIONS:
            qid = q["id"]
            selected = user_answers.get(qid, "")
            is_correct = (selected == q["answer"])
            if is_correct:
                total_score += q["points"]

            details.append({
                "id": qid,
                "category": q["category"],
                "question": q["question"],
                "selected": selected,
                "answer": q["answer"],
                "is_correct": is_correct,
                "points": q["points"] if is_correct else 0
            })

        # 표정 상태 가중치 반영
        is_calm_or_happy = dominant_emotion in ["happy", "neutral"]
        is_stressed = dominant_emotion in ["angry", "fear", "sad"]

        # 종합 상태 등급 판정
        if total_score >= 80:
            level = "총명 & 건강"
            color = "#10b981"
            icon = "🟢"
            title = "두뇌 활력 상태가 매우 훌륭하십니다!"
            desc = (
                f"퀴즈 점수 {total_score}점으로 계산력, 지남력, 언어 능력이 매우 뛰어납니다. "
                "얼굴 표정도 편안하고 안정적이어서 최상의 컨디션을 유지하고 계십니다."
            )
            action_guide = (
                "• 권장 활동: 신문 읽기, 가벼운 산책, 바둑/장기, 친구 및 가족과의 활발한 대화\n"
                "• 학습 제안: 새로운 취미(수채화, 악기 연주, 외국어 한마디 등)에 도전해 보세요!"
            )
        elif total_score >= 60:
            level = "두뇌 트레이닝 권장"
            color = "#f59e0b"
            icon = "🟡"
            title = "가벼운 두뇌 운동(트레이닝)이 권장됩니다."
            desc = (
                f"퀴즈 점수 {total_score}점으로 일상적인 소통은 원활하시나, "
                "간단한 계산이나 순간적인 기억력에서 약간의 연습이 필요할 수 있습니다."
            )
            action_guide = (
                "• 권장 활동: 하루 10분 숫자 퀴즈(구구단 거꾸로 외우기, 100에서 3씩 빼기), 단어 끝말잇기\n"
                "• 생활 습관: 충분한 수분 섭취와 규칙적인 수면 시간을 유지해 주세요."
            )
        else:
            level = "피로 / 상담 권장"
            color = "#ef4444"
            icon = "🟠"
            title = "충분한 휴식과 전문 상담을 권해드립니다."
            desc = (
                f"퀴즈 점수 {total_score}점입니다. "
                + ("오늘 다소 피로하거나 긴장된 표정이 감지되었습니다. " if is_stressed else "")
                + "수면 부족이나 일시적인 피로로 인해 집중이 어려우셨을 수 있습니다."
            )
            action_guide = (
                "• 첫 번째 조치: 따뜻한 차를 드시고 오늘 하루는 편안하게 충분한 휴식을 취하세요.\n"
                "• 안심 상담 안내: 이러한 집중 어려움이 며칠간 지속될 경우, 보건소 치매안심센터(국번없이 ☎ 1899-9988)에서 무료로 제공하는 친절한 인지 건강 상담을 받아보시면 큰 도움이 됩니다."
            )

        return {
            "score": total_score,
            "max_score": 100,
            "level": level,
            "color": color,
            "icon": icon,
            "title": title,
            "desc": desc,
            "action_guide": action_guide,
            "details": details,
            "dominant_emotion": dominant_emotion,
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
