from fastapi import APIRouter, status, UploadFile, File, HTTPException
from app.services.ml_service import ml_model 
from app.models import EmotionResponse
import time

predict_router = APIRouter(prefix = "/api/v1/predict")


@predict_router.post("/emotion", status_code = status.HTTP_200_OK)
async def get_predict(file: UploadFile = File(...)) -> EmotionResponse:

    

    try:
        start = time.time()
        img = await file.read()
        ml_model.load_model()
        emotion_predicted, confidence, all_probabilities = ml_model.predict(img) 
        processing_time_ms = round((time.time() - start) * 1000, 2)


    except ValueError as e:    

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    except RuntimeError as e:

        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = str(e))
    
    return EmotionResponse(
        status = "success",
        predicted_emotion = emotion_predicted,
        confidence = confidence,
        all_probabilities = all_probabilities,
        processing_time_ms = processing_time_ms
    )
    


@predict_router.get("/health", status_code = status.HTTP_200_OK)
async def get_health() -> dict:
    return {"status" : "success"}

        
