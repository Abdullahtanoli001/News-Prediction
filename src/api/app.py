from fastapi import FastAPI
from src.api.Database.database import Base, engine
from src.api.Routes.prediction import prediction_router


Base.metadata.create_all(bind=engine)
app = FastAPI()

@app.get('/')

def home():
    return {'News':'Welcome to News Prediciton'}

app.include_router(prediction_router)