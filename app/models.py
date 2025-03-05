from pydantic import BaseModel

class User(BaseModel):
    id: str 
    name: str

class AttendanceRecord(BaseModel):
    name: str
    user_id: str
    time: str
    event: str
    
class RecognizedFace(BaseModel):
    user_id: str
    name: str
    event: str
    confidence: float