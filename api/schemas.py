from enum import Enum
from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=1, description="Paso de tiempo de la simulación")
    type: TransactionType = Field(..., description="Tipo de transacción PaySim")
    amount: float = Field(..., ge=0.0, description="Monto de la transacción")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Saldo inicial de la cuenta origen")
    newbalanceOrig: float = Field(..., ge=0.0, description="Saldo final de la cuenta origen")
    oldbalanceDest: float = Field(..., ge=0.0, description="Saldo inicial de la cuenta destino")
    newbalanceDest: float = Field(..., ge=0.0, description="Saldo final de la cuenta destino")


class PredictionResponse(BaseModel):
    is_fraud_prediction: int = Field(..., description="1 = posible fraude, 0 = no fraude")
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="LOW, MEDIUM o HIGH")
    model_version: str = "medallion-fraud-v1.1"
    reasons: List[str] = Field(default_factory=list)