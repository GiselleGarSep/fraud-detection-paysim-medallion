cat << 'EOF' > start.sh
#!/usr/bin/env bash
set -e

cd /workspaces/fraud-detection-paysim-medallion
source .venv/bin/activate || true

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ "$1" = "api" ]; then
  cd api && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
elif [ "$1" = "app" ]; then
  python -m streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
else
  echo "Uso: ./start.sh [api|app]"
fi
EOF
chmod +x start.sh
