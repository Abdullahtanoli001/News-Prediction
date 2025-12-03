from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.Database.database import SessionLocal
from src.api.schemas.prediction import NewsInput
from src.pipeline.prediction_pipeline import prediction_pipeline
from src.api.Models.models import NewsPrediction

prediction_router = APIRouter(prefix="/predict")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@prediction_router.post("/")
def predict(data: NewsInput, db: Session = Depends(get_db)):

    data_dict = data.dict()

    # Run ML model
    pp = prediction_pipeline()
    prediction = pp.prediction(data_dict)

    # Save prediction in DB
    record = NewsPrediction(
        text=data.text,
        text_rank_summary=data.text_rank_summary,
        lsa_summary=data.lsa_summary,
        prediction=prediction
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return {"Prediction": prediction, "id": record.id}
