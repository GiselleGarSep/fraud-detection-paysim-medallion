import os
import httpx
import pandas as pd
import streamlit as st
from supabase_log import guardar_prediccion, obtener_bitacora

st.set_page_config(
    page_title="Detección de Fraude - Consulta",
    page_icon="🛡️",
    layout="wide",
)

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
            return {
                "ok": False,
                "error": f"Datos rechazados por la API ({response.status_code}): {response.text}",
            }
        if response.status_code >= 500:
            return {
                "ok": False,
                "error": f"Error interno en el servidor de predicción ({response.status_code}).",
            }
        if response.status_code != 200:
            return {
                "ok": False,
                "error": f"Respuesta HTTP inesperada: {response.status_code}",
            }

        data = response.json()
        required = {"is_fraud_prediction", "fraud_probability", "risk_level"}
        if not isinstance(data, dict) or not required.issubset(data.keys()):
            return {
                "ok": False,
                "error": "Respuesta inválida: faltan campos obligatorios en el JSON.",
            }
        if data["is_fraud_prediction"] not in (0, 1):
            return {
                "ok": False,
                "error": "Respuesta inválida: 'is_fraud_prediction' fuera de rango.",
            }

        return {"ok": True, "data": data}

    except httpx.TimeoutException:
        return {
            "ok": False,
            "error": f"Timeout: el endpoint no respondió en {timeout_s} segundos.",
        }
    except httpx.ConnectError:
        return {
            "ok": False,
            "error": f"Error de conexión: no se pudo alcanzar {url}.",
        }
    except ValueError:
        return {
            "ok": False,
            "error": "Respuesta inválida: el servicio no devolvió un JSON válido.",
        }
    except Exception as exc:
        return {"ok": False, "error": f"Error inesperado al consultar la API: {exc}"}


def build_audit_row(payload: dict, prediction: dict) -> dict:
    return {
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


st.title("🛡️ Consulta de Detección de Fraude")
st.caption(
    "Cliente HTTP conectado al servicio FastAPI + Docker con manejo de errores, timeouts y bitácora en Supabase."
)

with st.sidebar:
    st.subheader("Configuración de Conexión")
    api_url_input = st.text_input("URL del Endpoint API", value=API_BASE_URL)
    timeout_input = st.number_input(
        "Timeout (segundos)",
        min_value=0.1,
        max_value=30.0,
        value=REQUEST_TIMEOUT,
        step=0.5,
    )
    st.divider()
    if SUPABASE_URL and SUPABASE_KEY:
        st.success("Supabase configurado")
    else:
        st.warning(
            "Bitácora Supabase omitida (opcional: configura SUPABASE_URL y SUPABASE_ANON_KEY)."
        )

tab_consulta, tab_bitacora = st.tabs(
    ["🔍 Evaluar transacción", "📋 Bitácora Supabase"]
)

with tab_consulta:
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        with col1:
            step = st.number_input(
                "Step (hora simulación)", min_value=1, value=42, step=1
            )
            tx_type = st.selectbox(
                "Tipo de transacción",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            )
            amount = st.number_input(
                "Monto (amount)", min_value=0.0, value=250000.0, step=100.0
            )
            oldbalance_org = st.number_input(
                "Saldo inicial origen", min_value=0.0, value=250000.0, step=100.0
            )
        with col2:
            newbalance_orig = st.number_input(
                "Saldo final origen", min_value=0.0, value=0.0, step=100.0
            )
            oldbalance_dest = st.number_input(
                "Saldo inicial destino", min_value=0.0, value=0.0, step=100.0
            )
            newbalance_dest = st.number_input(
                "Saldo final destino", min_value=0.0, value=0.0, step=100.0
            )

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
            result = query_fraud_endpoint(
                api_url_input, tx_payload, float(timeout_input)
            )

        if not result["ok"]:
            st.error(result["error"])
        else:
            pred = result["data"]
            if pred["is_fraud_prediction"] == 1:
                st.error(
                    f"🚨 POSIBLE FRAUDE — Probabilidad: {pred['fraud_probability']:.2%} | Riesgo: {pred['risk_level']}"
                )
            else:
                st.success(
                    f"✅ TRANSACCIÓN NORMAL — Probabilidad de fraude: {pred['fraud_probability']:.2%} | Riesgo: {pred['risk_level']}"
                )

            if not SUPABASE_URL or not SUPABASE_KEY:
                st.warning(
                    "Bitácora Supabase omitida (opcional: configura SUPABASE_URL y SUPABASE_ANON_KEY si deseas guardarla)."
                )
            else:
                audit_row = build_audit_row(tx_payload, pred)
                try:
                    saved, msg = guardar_prediccion(audit_row, timeout=float(timeout_input))
                except TypeError:
                    saved, msg = guardar_prediccion(audit_row)

                if saved:
                    st.info(msg)
                else:
                    st.warning(msg)

with tab_bitacora:
    st.subheader("Historial de predicciones (`fraud_predictions_log`)")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        limit_rows = st.selectbox("Últimos registros", [10, 20, 50, 100], index=1)
    with col_ctrl2:
        st.write("")
        st.button("🔄 Actualizar tabla")

    try:
        df_log, err_log = obtener_bitacora(
            limit=int(limit_rows), timeout=float(timeout_input)
        )
    except TypeError:
        df_log, err_log = obtener_bitacora(limit=int(limit_rows))

    if err_log:
        st.warning(err_log)
    elif df_log is not None and not df_log.empty:
        if isinstance(df_log, list):
            df_log = pd.DataFrame(df_log)

        df_filtered = df_log.copy()

        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            if "risk_level" in df_filtered.columns:
                risk_options = sorted(
                    df_filtered["risk_level"].dropna().astype(str).unique().tolist()
                )
                selected_risks = st.multiselect(
                    "Nivel de riesgo",
                    options=risk_options,
                    default=[],
                    placeholder="Todos los niveles",
                )
                if selected_risks:
                    df_filtered = df_filtered[
                        df_filtered["risk_level"].astype(str).isin(selected_risks)
                    ]
            else:
                st.caption("Columna `risk_level` no disponible.")

        with f_col2:
            if "tx_type" in df_filtered.columns:
                tx_options = sorted(
                    df_filtered["tx_type"].dropna().astype(str).unique().tolist()
                )
                selected_types = st.multiselect(
                    "Tipo de transacción",
                    options=tx_options,
                    default=[],
                    placeholder="Todos los tipos",
                )
                if selected_types:
                    df_filtered = df_filtered[
                        df_filtered["tx_type"].astype(str).isin(selected_types)
                    ]
            else:
                st.caption("Columna `tx_type` no disponible.")

        with f_col3:
            if "amount" in df_filtered.columns:
                min_amount = st.number_input(
                    "Monto mínimo",
                    min_value=0.0,
                    value=0.0,
                    step=1000.0,
                )
                if min_amount > 0.0:
                    amounts_num = pd.to_numeric(df_filtered["amount"], errors="coerce")
                    df_filtered = df_filtered[amounts_num >= min_amount]
            else:
                st.caption("Columna `amount` no disponible.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registros mostrados", len(df_filtered))
        if "risk_level" in df_filtered.columns:
            risk_upper = df_filtered["risk_level"].astype(str).str.upper()
            m2.metric("Riesgo HIGH", int((risk_upper == "HIGH").sum()))
            m3.metric("Riesgo MEDIUM", int((risk_upper == "MEDIUM").sum()))
            m4.metric("Riesgo LOW", int((risk_upper == "LOW").sum()))

        priority_cols = ["tx_type", "risk_level", "fraud_probability"]
        existing_priority = [c for c in priority_cols if c in df_filtered.columns]
        remaining_cols = [c for c in df_filtered.columns if c not in existing_priority]
        df_filtered = df_filtered[existing_priority + remaining_cols]
        
        if df_filtered.empty:
            st.info("Ningún registro coincide con los filtros seleccionados.")
        else:
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)
    else:
        st.caption("No hay registros en la bitácora todavía.")