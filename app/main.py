from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse 
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.routers.predict import predict_router
from app.services.ml_service import ml_model

@asynccontextmanager
async def lifespan(app : FastAPI):
    #Код до yield выполняется во время запуска сервера
    print("Запуск приложения")
    try:
        
        ml_model.load_model()


    except Exception as e:
        print(f"Ошибка при загрузке модели {e}")
        raise     
    yield
    print("Приложение закрывается!")

    #Код после yield выполняется при закрытии сервера

app = FastAPI(lifespan = lifespan)

static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Корневой маршрут отдает index.html
@app.get("/")
async def root():
    return FileResponse(static_dir / "index.html")

app.include_router(predict_router)



