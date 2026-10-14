from fastapi import FastAPI 
from app.routers.predict import predict_router
from app.services.ml_service import EmotionModel


app = FastAPI()


app.include_router(predict_router)



