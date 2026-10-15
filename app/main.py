from fastapi import FastAPI 
from app.routers.predict import predict_router
from app.services.ml_service import ml_model


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


app.include_router(predict_router)



