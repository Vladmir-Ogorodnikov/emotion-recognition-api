from pydantic import BaseModel, field_validator
from typing import Literal

class EmotionResponse(BaseModel):

    status : str = "success"
    predicted_emotion : str = Literal["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]
    confidence : float
    all_probabilities : dict[str, float]
    processing_time_ms : float


    @field_validator("confidence")
    def chech_confidence(cls, value):
        if value > 1.0 or value < 0:
            raise ValueError('Вероятность от 0 до 1')
        return value
         
