from pydantic import BaseModel, Field, ConfigDict
from typing import Literal



class EmotionResponse(BaseModel):

    model_config = ConfigDict(from_attributes = True)

    status : str = "success"
    predicted_emotion : str = Literal["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]
    confidence : float = Field(ge=0.0, le=1.0, description="Уверенность модели")
    all_probabilities : dict[str, float]
    processing_time_ms : float


    
         
