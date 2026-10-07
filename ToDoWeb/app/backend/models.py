from datetime import datetime
from pydantic import BaseModel, Field
import uuid

class Tasks(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    title: str
    description: str = ""
    completed: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

