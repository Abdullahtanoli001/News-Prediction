from src.api.Database.database import Base
from sqlalchemy import Column, Integer, String

class NewsPrediction(Base):
    __tablename__ = "news_prediction"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    text_rank_summary = Column(String, nullable=False)
    lsa_summary = Column(String, nullable=False)
    prediction = Column(String, nullable=True)
