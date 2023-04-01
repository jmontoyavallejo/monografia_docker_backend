from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.models.model import predict_cost
from app.models.model import __version__ as model_version


origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://ephemeral-lolly-5a0ff4.netlify.app/",
   'https://ephemeral-lolly-5a0ff4.netlify.app',
    
]
app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInt(BaseModel):
  store_type: str
  store_state: str
  grocery_sqft: str
  frozen_sqft: str
  coffee_bar: str



class PredictionOut(BaseModel):
    answer: str

class TextIn(BaseModel):
  store_type: str
  store_state: str
  grocery_sqft: str
  frozen_sqft: str
  coffee_bar: str
  video_store: str
  prepared_food: str
  florist: str
  media_type: str
  marital_status: str
  gender: str
  total_children: str
  education: str
  member_card: str
  occupation: str
  houseowner: str
  SRP: str
  net_weight: str
  recyclable_package: str
  low_fat: str
  units_per_case: str
  food_family: str
  promotion_name: str
  sales_country: str
  avg_cars_at_home_approx: str
  avg_yearly_income: str
  num_children_at_home: str
  unit_sales_in_millions: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict")
def predict(payload: TextIn):
    cost = predict_cost(dict(payload))
    return {"prediction": cost}
