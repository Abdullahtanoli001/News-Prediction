from pydantic import BaseModel

class NewsInput(BaseModel):
    text: str
    text_rank_summary	: str
    lsa_summary: str

    