from fastapi import FastAPI, HTTPException
from model_service import score_transaction
from schemas import PredictionResponse, TransactionRequest

app = FastAPI(
    title="Fraud Detection API - PaySim Medallion",
    version="1.0.0",
    description=(
        "API REST para consultar la predicción de fraude financiero sobre transacciones "
        "con variables derivadas del pipeline Bronze -> Silver -> Gold en Databricks."
    ),
)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "fraud-detection-api"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: TransactionRequest):
    try:
        return score_transaction(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Datos inválidos: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error interno al procesar la predicción") from exc
