from fastapi import APIRouter, status, UploadFile, File



predict_router = APIRouter(prefix = "/api/v1/predict")


@predict_router.post("/emotion", status_code = status.HTTP_200_OK)
async def get_predict(file: UploadFile = File(...)) -> dict:
    await ...  
