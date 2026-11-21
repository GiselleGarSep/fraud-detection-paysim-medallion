import os
import httpx
import streamlit as st

st.set_page_config(page_title="Detección de Fraude - Consulta", page_icon="🛡️", layout="centered")

API_BASE_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "5.0"))


def query_fraud_endpoint(base_url: str, payload: dict, timeout_s: float) -> dict:
    url = f"{base_url.rstrip('/')}/predict"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, json=payload)

        if response.status_code in (400, 422):
            return {"ok": False, "error": f"Datos rechazados por la API ({response.status_code}): {response.text}"}
        if response.status_code >= 500:
            return {"ok": False, "error": f"Error interno en el servidor de predicción ({response.status_code})."}
        if response.status_code != 200:
            return {"ok": False, "error": f"Respuesta HTTP inesperada: {response.status_code}"}

        data = response.json()
        required = {"is_fraud_prediction", "fraud_probability", "risk_level"}
        if not isinstance(data, dict) or not required.issubset(data.keys()):
            return {"ok": False, "error": "Respuesta inválida: faltan campos obligatorios en el JSON."}
        if data["is_fraud_prediction"] not in (0, 1):
            return {"ok": False, "error": "Respuesta inválida: 'is_fraud_prediction' fuera de rango."}

        return {"ok": True, "data": data}

    except httpx.TimeoutException:
        return {"ok": False, "error": f"Timeout: el endpoint no respondió en {timeout_s} segundos."}
    except httpx.ConnectError:
        return {"ok": False, "error": f"Error de conexión: no se pudo alcanzar {url}."}
    except ValueError:
        return {"ok": False, "error": "Respuesta inválida: el servicio no devolvió un JSON válido."}
    except Exception as exc:
        return {"ok": False, "error": f"Error inesperado al consultar la API: {exc}"}


def save_to_supabase(payload: dict, prediction: dict) -> tuple[bool, str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False, "Bitácora Supabase omitida (opcional: configura SUPABASE_URL y SUPABASE_ANON_KEY si deseas guardarla)."

    endpoint = f"{SUPABASE_URL.rstrip('/')}/rest/v1/fraud_predictions_log"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    row = {
        "step": payload["step"],
        "tx_type": payload["type"],
        "amount": payload["amount"],
        "oldbalance_org": payload["oldbalanceOrg"],
        "newbalance_orig": payload["newbalanceOrig"],
        "oldbalance_dest": payload["oldbalanceDest"],
        "newbalance_dest": payload["newbalanceDest"],
        "is_fraud_prediction": prediction["is_fraud_prediction"],
        "fraud_probability": prediction["fraud_probability"],
        "risk_level": prediction["risk_level"],
    }
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.post(endpoint, headers=headers, json=row)
        if res.status_code in (200, 201, 204):
            return True, "Registro guardado en Supabase."
        return False, f"Error Supabase ({res.status_code}): {res.text}"
    except Exception as exc:
        return False, f"No se pudo guardar en Supabase: {exc}"


st.title("🛡️ Consulta de Detección de Fraude")
st.caption("Cliente HTTP conectado al servicio FastAPI + Docker con manejo de errores y timeouts.")

with st.sidebar:
    st.subheader("Configuración de Conexión")
    api_url_input = st.text_input("URL del Endpoint API", value=API_BASE_URL)
    timeout_input = st.number_input("Timeout (segundos)", min_value=0.1, max_value=30.0, value=REQUEST_TIMEOUT, step=0.5)

with st.form("fraud_form"):
    col1, col2 = st.columns(2)
    with col1:
        step = st.number_input("Step (hora simulación)", min_value=1, value=42, step=1)
        tx_type = st.selectbox("Tipo de transacción", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Monto (amount)", min_value=0.0, value=250000.0, step=100.0)
        oldbalance_org = st.number_input("Saldo inicial origen", min_value=0.0, value=250000.0, step=100.0)
    with col2:
        newbalance_orig = st.number_input("Saldo final origen", min_value=0.0, value=0.0, step=100.0)
        oldbalance_dest = st.number_input("Saldo inicial destino", min_value=0.0, value=0.0, step=100.0)
        newbalance_dest = st.number_input("Saldo final destino", min_value=0.0, value=0.0, step=100.0)

    submitted = st.form_submit_button("Evaluar transacción")

if submitted:
    tx_payload = {
        "step": int(step),
        "type": tx_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalance_org),
        "newbalanceOrig": float(newbalance_orig),
        "oldbalanceDest": float(oldbalance_dest),
        "newbalanceDest": float(newbalance_dest),
    }

    with st.spinner("Consultando endpoint publicado..."):
        result = query_fraud_endpoint(api_url_input, tx_payload, float(timeout_input))

    if not result["ok"]:
        st.error(result["error"])
    else:
        pred = result["data"]
        if pred["is_fraud_prediction"] == 1:
            st.error(f"🚨 POSIBLE FRAUDE — Probabilidad: {pred['fraud_probability']:.2%} | Riesgo: {pred['risk_level']}")
        else:
            st.success(f"✅ TRANSACCIÓN NORMAL — Probabilidad de fraude: {pred['fraud_probability']:.2%} | Riesgo: {pred['risk_level']}")

        saved, msg = save_to_supabase(tx_payload, pred)
        if saved:
            st.info(msg)
        else:
            st.warning(msg)
