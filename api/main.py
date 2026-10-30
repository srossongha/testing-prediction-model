from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = Path(__file__).parent.parent / "rent_model.joblib"
model =joblib.load(model_path)


class RoomInput(BaseModel):
    size_m2: float
    bedrooms: int
    distance_km: float
    has_aircon: int


@app.post("/predict")
def predict(room: RoomInput):
    new_room = pd.DataFrame({
        "size_m2": [room.size_m2],
        "bedrooms": [room.bedrooms],
        "distance_km": [room.distance_km],
        "has_aircon": [room.has_aircon],
    })
    price = model.predict(new_room)[0]
    return {"predicted_rent": round(float(price), 2)}