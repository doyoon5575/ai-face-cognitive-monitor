"""
AI 다중 생체신호 컨디션 & 인지 자가진단 평가 모듈 (core/assessment.py)
- 아동, 성인, 직장인, 시니어, 재활 환자 등 누구나 쉽게 참여하는 5문항 인지/컨디션 퀴즈
- 지남력, 집중 계산력, 언어 범주화, 상식 속담, 단기 기억력
- 안면 미세 표정(안정도) + 음성 활력도 + 퀴즈 결과를 종합한 다차원 컨디션 리포트
"""

from typing import List, Dict, Any
from datetime import datetime


class CognitiveConditionAssessment:
    """두뇌 활력 및 컨디션 평가 클래스"""

    QUESTIONS = [
        {
            "id": 1,
            "category": "지남력 (시간/환경)",
            "question": "지금은 일 년 중 어느 계절인가요?",
            "options": ["봄", "여름", "가을", "겨울"],
            "answer": "가을",
            "points": 20,
            "tip": "현재 시간과 계절에 대한 지각 능력을 확인합니다."
        },
        {
            "id": 2,
            "category": "집중 계산력",
            "question": "100원에서 7원을 빼면 얼마가 남을까요? (100 - 7 = ?)",
            "options": ["91원", "92원", "93원", "94원"],
            "answer": "93원",
            "points": 20,
            "tip": "작업 기억 및 기본적인 일상 뺄셈 집중력을 확인합니다."
        },
        {
            "id": 3,
            "category": "언어 및 범주화",
            "question": "다음 네 가지 중 성격이 다른 하나는 무엇인가요?",
            "options": ["사과", "바나나", "책상", "포도"],
            "answer": "책상",
            "points": 20,
            "tip": "단어의 속성과 범주(과일류)를 구별하는 어휘 판단력입니다."
        },
        {
            "id": 4,
            "category": "상식 및 연상력",
            "question": "‘가는 말이 고와야 (    )이 곱다’에 들어갈 알맞은 말은?",
            "options": ["오는 말", "가는 길", "웃는 낯", "마음씨"],
            "answer": "오는 말",
            "points": 20,
            "tip": "장기 기억 속의 친숙한 언어 연상 능력을 평가합니다."
        },
        {
            "id": 5,
            "category": "단기 기억 회상",
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
        eye_fatigue: float = 20.0,
        voice_vitality: float = 60.0
    ) -> Dict[str, Any]:
        """
        퀴즈 결과 + 안면 피로도 + 음성 활력도를 종합 분석하여 컨디션 리포트 도출
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

        # 다중 생체신호 가중치 평가
        is_stressed = dominant_emotion in ["angry", "fear", "sad"]
        is_high_fatigue = eye_fatigue >= 60.0
        is_low_voice = voice_vitality < 40.0

        if total_score >= 80 and not is_high_fatigue:
            level = "최적 컨디션 (매우 우수)"
            color = "#10b981"
            icon = "🟢"
            title = "두뇌 활력과 집중력이 최상입니다!"
            desc = (
                f"퀴즈 점수 {total_score}점이며, 안면 미세 표정과 목소리 활력 모두 안정적입니다. "
                "신체와 두뇌가 맑게 깨어 있는 최적의 컨디션을 유지하고 계십니다."
            )
            action_guide = (
                "• 권장 활동: 중요한 결정, 창의적 학습이나 독서, 활기찬 야외 활동\n"
                "• 컨디션 유지: 지금의 균형 잡힌 수면과 생활 리듬을 유지해 보세요!"
            )
        elif total_score >= 60:
            level = "양호 (가벼운 트레이닝 권장)"
            color = "#f59e0b"
            icon = "🟡"
            title = "원활한 상태이나 가벼운 두뇌 운동이 도움됩니다."
            desc = (
                f"퀴즈 점수 {total_score}점입니다. "
                + ("눈가 피로도가 다소 감지되었습니다. " if is_high_fatigue else "")
                + "기본적인 소통과 일상 집중은 양호하나 가벼운 두뇌 자극 퍼즐을 권장합니다."
            )
            action_guide = (
                "• 권장 활동: 10분 숫자 퀴즈, 단어 끝말잇기, 가벼운 목·어깨 스트레칭\n"
                "• 수분 섭취: 시원한 물 한 잔으로 두뇌 순환을 돕고 잠시 눈을 감고 쉬어가세요."
            )
        else:
            level = "피로 누적 / 휴식 권장"
            color = "#ef4444"
            icon = "🟠"
            title = "충분한 휴식과 재충전이 필요합니다."
            desc = (
                f"퀴즈 점수 {total_score}점입니다. "
                + ("긴장되거나 지친 표정이 감지되었습니다. " if is_stressed else "")
                + ("목소리 톤이 다소 가라앉아 있습니다. " if is_low_voice else "")
                + "일시적인 수면 부족이나 과로로 인해 집중력이 저하되었을 수 있습니다."
            )
            action_guide = (
                "• 첫 번째 조치: 오늘 무리한 일정은 줄이시고 따뜻한 음료와 함께 충분한 수면을 취하세요.\n"
                "• 전문 상담 안내: 이러한 집중 저하나 피로감이 오래 지속될 경우, 가까운 건강상담센터 또는 보건소(1899-9988)의 친절한 인지 건강 체크를 편안하게 받아보시길 권장합니다."
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
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
