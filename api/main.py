from typing import List, Literal

from fastapi import FastAPI, HTTPException
from model_service import score_transaction
from schemas import PredictionResponse, TransactionRequest

MODEL_VERSION = "medallion-fraud-v1.1"

app = FastAPI(
    title="Fraud Detection API - PaySim Medallion",
    version=MODEL_VERSION,
    description=(
        "API REST para consultar la predicción de fraude financiero sobre transacciones "
        "con variables derivadas del pipeline Bronze -> Silver -> Gold en Databricks."
    ),
)


def _extract_value(result, attr: str, default=None):
    if isinstance(result, dict):
        return result.get(attr, default)
    return getattr(result, attr, default)


def _build_reasons(request: TransactionRequest, probability: float) -> List[str]:
    reasons: List[str] = []

    tx_type = getattr(request, "type", None)
    amount = getattr(request, "amount", 0.0) or 0.0
    old_orig = getattr(request, "oldbalanceOrg", 0.0) or 0.0
    new_orig = getattr(request, "newbalanceOrig", 0.0) or 0.0
    old_dest = getattr(request, "oldbalanceDest", 0.0) or 0.0
    new_dest = getattr(request, "newbalanceDest", 0.0) or 0.0

    if tx_type in ("TRANSFER", "CASH_OUT"):
        reasons.append(f"Operación tipo '{tx_type}' con mayor exposición a fraude")

    if old_orig > 0 and new_orig == 0:
        reasons.append("Vaciado total del saldo de la cuenta origen (saldo final en 0)")

    expected_orig = max(0.0, old_orig - amount)
    if tx_type != "CASH_IN" and abs(new_orig - expected_orig) > 1e-2:
        reasons.append("Diferencia entre el monto operado y el cambio en el saldo origen")

    if tx_type == "TRANSFER" and old_dest == 0 and new_dest == 0:
        reasons.append("Cuenta destino sin saldo inicial ni cambio en saldo destino")

    if amount >= 200_000:
        reasons.append(f"Monto elevado ({amount:,.2f}) respecto al umbral de referencia")

    if not reasons:
        if probability >= 0.5:
            reasons.append("Probabilidad elevada detectada por el modelo o variables del pipeline")
        else:
            reasons.append("Comportamiento consistente con patrones normales de la transacción")

    return reasons


@app.get("/", tags=["Info"])
def root():
    return {
        "status": "ok",
        "service": "Fraud Detection API",
        "model_version": MODEL_VERSION,
        "health": "/health",
        "docs": "/docs",
        "predict": "/predict",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "service": "fraud-detection-api",
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: TransactionRequest):
    try:
        raw_result = score_transaction(request)

        probability = float(_extract_value(raw_result, "fraud_probability", 0.0))
        is_fraud_raw = _extract_value(raw_result, "is_fraud_prediction", None)
        is_fraud_prediction = (
            bool(is_fraud_raw) if is_fraud_raw is not None else probability >= 0.5
        )

        risk_level_raw = _extract_value(raw_result, "risk_level", None)
        if risk_level_raw in ("LOW", "MEDIUM", "HIGH"):
            risk_level: Literal["LOW", "MEDIUM", "HIGH"] = risk_level_raw
        elif probability >= 0.7:
            risk_level = "HIGH"
        elif probability >= 0.35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        model_version = _extract_value(raw_result, "model_version", None) or MODEL_VERSION
        reasons = _extract_value(raw_result, "reasons", None) or _build_reasons(
            request, probability
        )

        payload = {
            "model_version": model_version,
            "is_fraud_prediction": is_fraud_prediction,
            "fraud_probability": round(probability, 4),
            "risk_level": risk_level,
            "reasons": reasons,
        }

        if isinstance(raw_result, dict):
            return {**raw_result, **payload}

        if hasattr(raw_result, "model_copy"):
            return raw_result.model_copy(update=payload)

        for key, value in payload.items():
            setattr(raw_result, key, value)
        return raw_result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Datos inválidos: {exc}") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Error interno al procesar la predicción"
        ) from exc