from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    queixa: str

@app.get("/")
def root():
    return {"status": "ok", "modelo": "TriageGeist v0.1"}

@app.post("/predict")
def predict(data: Input):
    return {"queixa": data.queixa, "acuidade": 3}