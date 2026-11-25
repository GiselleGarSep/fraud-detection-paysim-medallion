import os
from typing import Any
import httpx

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_ANON_KEY = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
SUPABASE_TABLE = (os.getenv("SUPABASE_TABLE") or "fraud_predictions").strip()


def supabase_configurado() -> bool:
    """Indica si existen la URL y la clave de Supabase en el entorno."""
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def _headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _normalizar_filas(raw: Any) -> list[dict[str, Any]]:
    """Convierte respuestas de Supabase/tuplas/objetos en una lista plana de dicts."""
    if raw is None:
        return []
    data = getattr(raw, "data", raw)
    if isinstance(data, tuple):
        data = data[0] if len(data) > 0 else []
    if not isinstance(data, list):
        return []

    filas_limpias: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        fila_limpia: dict[str, Any] = {}
        for k, v in item.items():
            if isinstance(v, (list, dict)):
                fila_limpia[k] = str(v)
            else:
                fila_limpia[k] = v
        filas_limpias.append(fila_limpia)
    return filas_limpias


def guardar_prediccion(registro: dict[str, Any]) -> list[dict[str, Any]]:
    """Inserta una predicción en la tabla de Supabase."""
    if not supabase_configurado():
        raise RuntimeError("Configura SUPABASE_URL y SUPABASE_ANON_KEY para guardar en la bitácora.")

    endpoint = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    with httpx.Client(timeout=10.0) as client:
        response = client.post(endpoint, headers=_headers(), json=registro)
        response.raise_for_status()
        return _normalizar_filas(response.json())


def obtener_bitacora(limit: int = 200) -> list[dict[str, Any]]:
    """Consulta los registros de la bitácora en Supabase y devuelve una lista de dicts."""
    if not supabase_configurado():
        raise RuntimeError("Configura SUPABASE_URL y SUPABASE_ANON_KEY para ver la bitácora.")

    endpoint = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    params = {"select": "*", "limit": str(limit)}
    with httpx.Client(timeout=10.0) as client:
        response = client.get(endpoint, headers=_headers(), params=params)
        response.raise_for_status()
        return _normalizar_filas(response.json())