from typing import List, Literal, Optional
from pydantic import BaseModel


class Question(BaseModel):
    question: str
    type: Literal["multiple_choice", "true_false", "short_answer"]
    correct_answer: str
    explanation: str
    difficulty: str
    options: Optional[List[str]] = None
    bloom_level: str = "Remember"


class QuizOutput(BaseModel):
    questions: List[Question]
