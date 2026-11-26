import math
import os
from typing import List

from schemas import PredictionResponse, TransactionRequest

MODEL_VERSION = "medallion-fraud-v1.1"

TYPE_INDEX = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.75"))


def score_transaction(payload: TransactionRequest) -> PredictionResponse:
    delta_orig = payload.newbalanceOrig - payload.oldbalanceOrg
    delta_dest = payload.newbalanceDest - payload.oldbalanceDest
    error_balance_orig = abs((payload.oldbalanceOrg - payload.amount) - payload.newbalanceOrig)
    error_balance_dest = abs((payload.oldbalanceDest + payload.amount) - payload.newbalanceDest)
    log_amount = math.log1p(max(payload.amount, 0.0))
    _type_idx = TYPE_INDEX[payload.type.value]

    reasons: List[str] = []
    prob = 0.08

    if payload.type.value in {"TRANSFER", "CASH_OUT"}:
        prob += 0.32
        reasons.append(f"Operación tipo '{payload.type.value}' con mayor exposición a fraude")
    if error_balance_orig > 1.0:
        prob += 0.35
        reasons.append("Diferencia entre el monto operado y el cambio en el saldo origen")
    if payload.oldbalanceOrg > 0 and payload.newbalanceOrig == 0.0 and payload.amount > 0:
        prob += 0.15
        reasons.append("Vaciado total del saldo de la cuenta origen (saldo final en 0)")
    if log_amount > 12.0:
        prob += 0.07
        reasons.append(f"Monto elevado ({payload.amount:,.2f}) respecto al umbral de referencia")
    if delta_dest == 0.0 and payload.type.value == "TRANSFER" and payload.amount > 0:
        prob += 0.05
        reasons.append("Cuenta destino sin cambio en saldo destino")

    fraud_probability = round(min(0.99, max(0.01, prob)), 4)
    is_fraud = 1 if fraud_probability >= FRAUD_THRESHOLD else 0

    if fraud_probability >= 0.75:
        risk_level = "HIGH"
    elif fraud_probability >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if not reasons:
        reasons.append("Comportamiento consistente con patrones normales de la transacción")

    return PredictionResponse(
        is_fraud_prediction=is_fraud,
        fraud_probability=fraud_probability,
        risk_level=risk_level,
        model_version=MODEL_VERSION,
        reasons=reasons,
    )