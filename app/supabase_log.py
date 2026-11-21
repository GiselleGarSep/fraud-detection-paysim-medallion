import os
import pandas as pd
import requests


def get_supabase_config() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_ANON_KEY", "")
    return url, key


def guardar_prediccion(record: dict, timeout: float = 5.0) -> tuple[bool, str]:
    """Inserta una fila en public.fraud_predictions_log."""
    url, key = get_supabase_config()
    if not url or not key:
        return False, "Variables SUPABASE_URL y SUPABASE_ANON_KEY no configuradas."

    try:
        resp = requests.post(
            f"{url}/rest/v1/fraud_predictions_log",
            json=record,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return True, "Registro guardado en Supabase (fraud_predictions_log)."
    except requests.RequestException as exc:
        return False, f"No se pudo guardar en Supabase: {exc}"


def obtener_bitacora(limit: int = 20, timeout: float = 5.0) -> tuple[pd.DataFrame | None, str | None]:
    """Consulta los últimos registros de public.fraud_predictions_log."""
    url, key = get_supabase_config()
    if not url or not key:
        return None, "Configura SUPABASE_URL y SUPABASE_ANON_KEY para ver la bitácora."

    try:
        resp = requests.get(
            f"{url}/rest/v1/fraud_predictions_log",
            params={
                "select": (
                    "id,created_at,step,tx_type,amount,"
                    "oldbalance_org,newbalance_orig,oldbalance_dest,newbalance_dest,"
                    "is_fraud_prediction,fraud_probability,risk_level"
                ),
                "order": "id.desc",
                "limit": limit,
            },
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        rows = resp.json()
        return pd.DataFrame(rows), None
    except requests.RequestException as exc:
        return None, f"Error al consultar Supabase: {exc}"
