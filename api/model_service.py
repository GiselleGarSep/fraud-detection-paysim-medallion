import math
import os
from schemas import PredictionResponse, TransactionRequest

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

    prob = 0.08
    if payload.type.value in {"TRANSFER", "CASH_OUT"}:
        prob += 0.32
    if error_balance_orig > 1.0:
        prob += 0.35
    if payload.oldbalanceOrg > 0 and payload.newbalanceOrig == 0.0 and payload.amount > 0:
        prob += 0.15
    if log_amount > 12.0:
        prob += 0.07
    if delta_dest == 0.0 and payload.type.value == "TRANSFER" and payload.amount > 0:
        prob += 0.05

    fraud_probability = round(min(0.99, max(0.01, prob)), 4)
    is_fraud = 1 if fraud_probability >= FRAUD_THRESHOLD else 0

    if fraud_probability >= 0.75:
        risk_level = "HIGH"
    elif fraud_probability >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictionResponse(
        is_fraud_prediction=is_fraud,
        fraud_probability=fraud_probability,
        risk_level=risk_level,
    )
