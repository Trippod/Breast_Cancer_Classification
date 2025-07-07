# src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, constr
from src.predicted_router import predict, MODEL_PATHS, FEATURE_DIM

class PredictRequest(BaseModel):
    model: constr(strip_whitespace=True, to_lower=True)
    features: conlist(float,min_length=FEATURE_DIM, max_length=FEATURE_DIM)

app = FastAPI(
    title="Breast-Cancer Classifier API",
    version="0.1.0",
    description="Wybierz model i podaj 30 cech – otrzymasz etykietę 0/1 i prawdopodobieństwo.",
)

@app.get("/")
def root():
    return {"status": "ok", "models": list(MODEL_PATHS.keys())}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        result = predict(req.model, req.features)   # z kroku 7
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
