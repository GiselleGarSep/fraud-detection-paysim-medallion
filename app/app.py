import os
from pathlib import Path
import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from supabase_log import guardar_prediccion, obtener_bitacora, supabase_configurado

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo_tarjeta.svg")
PAGE_ICON = LOGO_PATH if os.path.exists(LOGO_PATH) else "💳"

st.set_page_config(
    page_title="Detección de Fraude - Consulta",
    page_icon=PAGE_ICON,
    layout="wide",
)

DEFAULT_API_URL = os.getenv("FRAUD_API_URL") or os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10.0"))


def normalizar_base_url(raw_url: str) -> str:
    """Normaliza la URL base para evitar que termine en /predict o tenga barras extra."""
    cleaned = (raw_url or "http://localhost:8000").strip().rstrip("/")
    if cleaned.endswith("/predict"):
        cleaned = cleaned[: -len("/predict")].rstrip("/")
    return cleaned or "http://localhost:8000"


def query_fraud_endpoint(base_url: str, payload: dict, timeout_s: float = 10.0) -> dict:
    """Envía la petición POST al endpoint /predict de la API FastAPI."""
    clean_base = normalizar_base_url(base_url)
    url = f"{clean_base}/predict"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, json=payload)

        if response.status_code in (400, 422):
            return {
                "ok": False,
                "error": f"Datos rechazados por la API ({response.status_code}): {response.text}",
            }
        if response.status_code == 404:
            return {
                "ok": False,
                "error": f"Ruta no encontrada en {url} (404). Verifica que la API esté corriendo en el puerto 8000.",
            }
        response.raise_for_status()
        return {"ok": True, "data": response.json(), "url": url}
    except httpx.TimeoutException:
        return {"ok": False, "error": f"Tiempo de espera agotado al consultar {url}."}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"Error de conexión con la API ({url}): {exc}"}


def construir_dataframe_bitacora(registros: object) -> pd.DataFrame:
    """Construye un DataFrame limpio evitando secuencias anidadas o formas irregulares."""
    if not isinstance(registros, list) or not registros:
        return pd.DataFrame()

    filas_limpias: list[dict] = []
    for fila in registros:
        if not isinstance(fila, dict):
            continue
        limpia = {}
        for col, val in fila.items():
            if isinstance(val, (list, dict, tuple, set)):
                limpia[col] = str(val)
            else:
                limpia[col] = val
        filas_limpias.append(limpia)

    if not filas_limpias:
        return pd.DataFrame()

    df = pd.DataFrame(filas_limpias)
    prioridad = ["tx_type", "risk_level", "fraud_probability", "amount", "is_fraud_prediction"]
    cols_prioritarias = [c for c in prioridad if c in df.columns]
    cols_restantes = [c for c in df.columns if c not in cols_prioritarias]
    return df[cols_prioritarias + cols_restantes]


# --- Encabezado con branding ---
col_logo, col_title = st.columns([1, 10], vertical_alignment="center")
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=56)
with col_title:
    st.title("Detección de Fraude — PaySim Medallion")
    st.caption("Evaluación en tiempo real vía FastAPI + Bitácora de auditoría en Supabase")

# --- Barra lateral ---
with st.sidebar:
    st.header("Configuración de API")
    api_url_input = st.text_input(
        "URL base de la API",
        value=normalizar_base_url(DEFAULT_API_URL),
        help="Ejemplo: http://localhost:8000 (la app agrega /predict automáticamente)",
    )
    api_base_url = normalizar_base_url(api_url_input)
    st.caption(f"Endpoint activo: `{api_base_url}/predict`")

# --- Pestañas principales ---
tab_eval, tab_bitacora = st.tabs(["🔍 Evaluación", "📋 Bitácora"])

with tab_eval:
    st.subheader("Evaluar transacción")
    with st.form("form_transaccion"):
        c1, c2, c3 = st.columns(3)
        with c1:
            step = st.number_input("Step (hora / paso)", min_value=1, value=1, step=1)
            tx_type = st.selectbox(
                "Tipo de transacción",
                options=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                index=0,
            )
            amount = st.number_input("Monto (amount)", min_value=0.0, value=1000.0, step=100.0)
        with c2:
            oldbalance_org = st.number_input("Saldo origen anterior", min_value=0.0, value=5000.0, step=100.0)
            newbalance_orig = st.number_input("Saldo origen nuevo", min_value=0.0, value=4000.0, step=100.0)
        with c3:
            oldbalance_dest = st.number_input("Saldo destino anterior", min_value=0.0, value=0.0, step=100.0)
            newbalance_dest = st.number_input("Saldo destino nuevo", min_value=0.0, value=1000.0, step=100.0)

        submitted = st.form_submit_button("Evaluar transacción", type="primary")

    if submitted:
        payload = {
            "step": int(step),
            "type": tx_type,
            "amount": float(amount),
            "oldbalanceOrg": float(oldbalance_org),
            "newbalanceOrig": float(newbalance_orig),
            "oldbalanceDest": float(oldbalance_dest),
            "newbalanceDest": float(newbalance_dest),
        }

        with st.spinner("Consultando modelo de fraude..."):
            resultado = query_fraud_endpoint(api_base_url, payload, timeout_s=REQUEST_TIMEOUT)

        if not resultado["ok"]:
            st.error(resultado["error"])
        else:
            data = resultado["data"]
            pred = data.get("is_fraud_prediction", 0)
            prob = data.get("fraud_probability", 0.0)
            risk = data.get("risk_level", "UNKNOWN")

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicción", "POSIBLE FRAUDE" if pred == 1 else "TRANSACCIÓN NORMAL")
            m2.metric(
                "Probabilidad de fraude",
                f"{float(prob):.2%}" if float(prob) <= 1 else f"{float(prob):.4f}",
            )
            m3.metric("Nivel de riesgo", str(risk))

            # --- Tarjeta visual según el nivel de riesgo ---
            riesgo_norm = str(risk).strip().lower()
            img_path = None
            if any(k in riesgo_norm for k in ("alto", "high")):
                img_path = ASSETS_DIR / "RiesgoAlto.jpeg"
            elif any(k in riesgo_norm for k in ("medio", "medium", "mod")):
                img_path = ASSETS_DIR / "RiesgoMedio.jpeg"
            elif any(k in riesgo_norm for k in ("bajo", "low")):
                img_path = ASSETS_DIR / "RiesgoBajo.jpeg"

            if img_path and img_path.exists():
                _, col_img, _ = st.columns([1, 2, 1])
                with col_img:
                    st.image(str(img_path), width=320)

            registro = {
                "step": int(step),
                "tx_type": str(tx_type),
                "amount": float(amount),
                "oldbalance_org": float(oldbalance_org),
                "newbalance_orig": float(newbalance_orig),
                "oldbalance_dest": float(oldbalance_dest),
                "newbalance_dest": float(newbalance_dest),
                "is_fraud_prediction": int(pred if pred is not None else 0),
                "fraud_probability": float(prob if prob is not None else 0.0),
                "risk_level": str(risk if risk is not None else "UNKNOWN"),
            }
            try:
                guardar_prediccion(registro)
                st.success("Predicción registrada correctamente en Supabase.")
            except Exception as exc:
                st.warning(f"La predicción se obtuvo, pero hubo un problema al guardar en Supabase: {exc}")

with tab_bitacora:
    st.subheader("Bitácora de evaluaciones en Supabase")
    if st.button("Actualizar bitácora"):
        st.cache_data.clear()

    if not supabase_configurado():
        st.info("Configura SUPABASE_URL y SUPABASE_ANON_KEY para ver la bitácora.")
    else:
        try:
            registros = obtener_bitacora()
            df_log = construir_dataframe_bitacora(registros)
        except Exception as exc:
            st.error(f"No se pudo cargar la bitácora desde Supabase: {exc}")
            df_log = pd.DataFrame()

        if df_log.empty:
            st.info("Aún no hay registros en la bitácora o la tabla está vacía.")
        else:
            f1, f2, f3 = st.columns(3)
            with f1:
                tipos_disp = (
                    sorted(df_log["tx_type"].dropna().astype(str).unique().tolist())
                    if "tx_type" in df_log.columns
                    else []
                )
                tipos_sel = st.multiselect(
                    "Filtrar por tipo de transacción", options=tipos_disp, default=tipos_disp
                )
            with f2:
                riesgos_disp = (
                    sorted(df_log["risk_level"].dropna().astype(str).unique().tolist())
                    if "risk_level" in df_log.columns
                    else []
                )
                riesgos_sel = st.multiselect(
                    "Filtrar por nivel de riesgo", options=riesgos_disp, default=riesgos_disp
                )
            with f3:
                if "amount" in df_log.columns and pd.api.types.is_numeric_dtype(df_log["amount"]):
                    min_amt = float(df_log["amount"].min())
                    max_amt = float(df_log["amount"].max())
                    if min_amt < max_amt:
                        rango_amt = st.slider(
                            "Rango de monto", min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt)
                        )
                    else:
                        rango_amt = (min_amt, max_amt)
                else:
                    rango_amt = None

            df_filtrado = df_log.copy()
            if "tx_type" in df_filtrado.columns and tipos_sel:
                df_filtrado = df_filtrado[df_filtrado["tx_type"].astype(str).isin(tipos_sel)]
            if "risk_level" in df_filtrado.columns and riesgos_sel:
                df_filtrado = df_filtrado[df_filtrado["risk_level"].astype(str).isin(riesgos_sel)]
            if rango_amt is not None and "amount" in df_filtrado.columns:
                df_filtrado = df_filtrado[
                    (df_filtrado["amount"] >= rango_amt[0]) & (df_filtrado["amount"] <= rango_amt[1])
                ]

            # --- Métricas rápidas de la bitácora filtrada ---
            k1, k2, k3 = st.columns(3)
            total_reg = len(df_filtrado)
            k1.metric("Transacciones evaluadas", total_reg)

            if "risk_level" in df_filtrado.columns:
                alto_riesgo = df_filtrado["risk_level"].astype(str).str.lower().str.contains("alto|high").sum()
            else:
                alto_riesgo = 0
            k2.metric("Riesgo alto", int(alto_riesgo))

            monto_total = float(df_filtrado["amount"].sum()) if "amount" in df_filtrado.columns else 0.0
            k3.metric("Monto acumulado", f"${monto_total:,.2f}")

            st.dataframe(df_filtrado, use_container_width=True, hide_index=True)

            csv_bytes = df_filtrado.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar bitácora (CSV)",
                data=csv_bytes,
                file_name="bitacora_fraude.csv",
                mime="text/csv",
            )