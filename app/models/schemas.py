from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(max_length=1000)
    session_id: str
    force_refresh: bool = False

