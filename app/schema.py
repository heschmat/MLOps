from pydantic import BaseModel, Field
from typing import Optional, Literal


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: Literal["male", "female"]
    bp: float
    max_hr: Optional[float] = Field(None, gt=0)
    exercise_angina: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: int
    probability: str
