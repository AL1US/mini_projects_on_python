from datetime import datetime
from  pydantic import BaseModel

class Tasks(BaseModel):
    id: str
    title: str
    description: str
    completed: bool
    created_at: datetime


