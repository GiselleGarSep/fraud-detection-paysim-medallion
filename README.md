# Financial Fraud Detection Pipeline (PaySim — Medallion Architecture)

An end-to-end financial fraud detection system built on the **PaySim** synthetic mobile money dataset (**6,362,620** records), structured around the **Medallion architecture** (**Bronze**, **Silver**, and **Gold**) using **Databricks** (Apache Spark & Delta Lake), a class-weighted **Random Forest** classifier, a **Dockerized FastAPI** inference service, **Supabase**, and a **Streamlit** query and audit-log application.

---

## 💻 Code Author

- NAME: Giselle Garcia Sepulveda
- ID: 0151990

---

## 1. 🎯 Problem Definition & Dataset Scope

- **Dataset:** PaySim (`PS_20174392719_1491204439457_log.csv`) with **6,362,620** total transactions (exceeding the 4M-record requirement).
- **Task Type:** Supervised **binary classification** predicting `isFraud` (`1` = fraudulent transaction, `0` = legitimate transaction).
- **Class Imbalance:** Out of **6,362,620** transactions, **8,213** (`~0.129%`) are fraud (`1`) and **6,354,407** (`~99.871%`) are non-fraud (`0`).
- **Evaluation Metric Justification (`F1-Score`):** Because accuracy is dominated by the majority class under severe class imbalance, **F1-Score** (the harmonic mean of precision and recall) is used alongside class-specific metrics (`TP`, `FP`, `TN`, `FN`) to evaluate the trade-off between catching fraud (`false negatives`) and generating false alerts (`false positives`).

---

## 2. 🥉🥈🥇 Medallion Architecture in Databricks

All layers are organized under the `transacciones_banco` catalog in Unity Catalog:

1. **🥉 Bronze (`transacciones_banco.Bronze.transacciones_raw`):**
   - Ingests raw data from `/Volumes/transacciones_banco/default/raw_data/PS_20174392719_1491204439457_log.csv` into a Delta table preserving all **6,362,620** records along with ingestion metadata (`_ingested_at`, `_source_file`).
2. **🥈 Silver (`transacciones_banco.Silver.transacciones_limpias`):**
   - Checks null values, removes exact duplicates/critical nulls, and enforces explicit data types (`int`, `double`, `string`).
   - Detects transaction amount anomalies via the **Interquartile Range (IQR)** rule (`is_amount_outlier_iqr`) rather than blindly dropping extreme amounts.
   - Engineers balance and amount features (`delta_orig`, `delta_dest`, `error_balance_orig`, `error_balance_dest`, `log_amount`), encodes transaction `type` (`type_idx` and indicator columns), and applies **Z-score** normalization (`log_amount_zscore`, `err_balance_orig_zscore`).
3. **🥇 Gold (`transacciones_banco.Gold.features_fraude`):**
   - Stores curated, model-ready features and target labels (`isFraud`) in a Delta table for training and analytical queries.

---

## 3. 🤖 Model Training & Evaluation Results

A **Random Forest** classifier was trained with inverse-frequency class weighting (`class_weight`) to account for fraud scarcity on an 80/20 train/test split (`seed=42`).

### Weighted Metrics (Test Set)
- **F1-Score (weighted):** `0.9862`
- **Precision (weighted):** `0.9988`
- **Recall (weighted):** `0.9749`

### Confusion Matrix & Fraud Class (`isFraud = 1`) Breakdown
- **True Positives (`TP` — fraud caught):** `1,562`
- **False Positives (`FP` — false alarms):** `31,925`
- **True Negatives (`TN` — legitimate caught):** `1,240,353`
- **False Negatives (`FN` — fraud missed):** `7`
- **Fraud Class Precision (`1`):** `0.0466`
- **Fraud Class Recall (`1`):** `0.9955`
- **Fraud Class F1-Score (`1`):** `0.0891`

**Key Takeaway:** High class weighting yields very high fraud recall (`0.9955`, missing only `7` fraud cases in test) while producing `31,925` false positives—highlighting the operational trade-off between minimizing missed fraud and reducing alert review volume.

---

## 4. 🏗️ System Architecture (Databricks, FastAPI/Docker & Supabase)

- **Databricks:** Raw ingestion volume, Medallion Delta tables (`Bronze`, `Silver`, `Gold`), and Spark ML training/evaluation.
- **FastAPI + Docker (`api/`):** Exposes the prediction service via a REST API (`POST /predict` and `GET /health`) with **Pydantic** input validation and automatic **OpenAPI / Swagger** documentation at `/docs`.
- **Supabase:** Persists prediction audit logs in PostgreSQL (`fraud_predictions`) including risk classification, model version, and evaluation reasons.
- **Streamlit Query & Log App (`app/app.py`):**
  - Displays live API health status (`GET /health`) in the sidebar and normalizes the base URL for `POST /predict`.
  - **Evaluación tab:** Provides randomized scenario presets (*Pago normal*, *Vaciado total + TRANSFER*, *Descuadre saldo origen*, *Monto muy alto*), base reset, fraud probability progress bar, model version, evaluation reasons, risk visual asset, and a faceted Altair balance chart (`Origen` followed by `Destino`) with an `amount` reference line.
  - **Bitácora tab:** Filters by transaction type, normalized risk level (`BAJO`, `MEDIO`, `ALTO`, `DESCONOCIDO`), and amount range; shows summary KPIs, risk distribution and amount-vs-probability charts with a fixed risk color palette, an evaluations-by-`step` chart, a row-level `reasons` inspector, and CSV export.

---

## 5. 📁 Repository Structure

```text
fraud-detection-project/
├── README.md
├── .env.example
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── model_service.py
│   ├── requirements.txt
│   └── Dockerfile
└── app/
    ├── app.py
    ├── supabase_log.py
    ├── logo_tarjeta.svg
    └── assets/
        ├── RiesgoAlto.jpeg
        ├── RiesgoMedio.jpeg
        └── RiesgoBajo.jpeg
---

## 6. 🐳 Running the API with Docker

Build and run the container on port 8000:

```bash
docker build -t fraud-detection-api ./api
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api
```
- 📄 Swagger / OpenAPI Docs: http://localhost:8000/docs
- ❤️ Health Check: GET http://localhost:8000/health
- 🤖 Predict Endpoint: POST http://localhost:8000/predict

Example request payload:

```json
{
  "step": 42,
  "type": "TRANSFER",
  "amount": 250000.0,
  "oldbalanceOrg": 250000.0,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0
}
```

---

## 7. 📊 Running Streamlit & Logging to Supabase (fraud_predictions_log)

🔑 Export the required environment variables in your terminal session before starting Streamlit (do not commit real secrets to Git):

```bash
export FRAUD_API_URL="http://localhost:8000"
export REQUEST_TIMEOUT="10.0"
export SUPABASE_URL="[https://YOUR-PROJECT.supabase.co](https://YOUR-PROJECT.supabase.co)"
export SUPABASE_ANON_KEY="YOUR_SUPABASE_ANON_OR_PUBLISHABLE_KEY"
```

The Streamlit app inserts and reads records from public.fraud_predictions_log using these columns:

- step (int)
- tx_type (text)
- amount (float)
- oldbalance_org (float)
- newbalance_orig (float)
- oldbalance_dest (float)
- newbalance_dest (float)
- is_fraud_prediction (int)
- fraud_probability (float)
- risk_level (text)
- model_version (text)
- reasons (list/text)

🚀 Start Streamlit (adjust the path if app.py is at the repository root):

```bash
streamlit run app/app.py
```

✅ Evaluate a transaction in the Evaluación tab, open the Bitácora tab, and click Actualizar bitácora to verify that the prediction is persisted, filterable, and inspectable.


---

