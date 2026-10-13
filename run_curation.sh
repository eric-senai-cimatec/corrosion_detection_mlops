#!/bin/bash
# Ativa o ambiente virtual caso não esteja ativo
source .venv/bin/activate

echo "🎨 Opening Labelme for Human-in-the-Loop curation..."
QT_QPA_PLATFORM=xcb labelme review_data

echo "🚀 Labelme closed! Launching automated conversion and Airflow Webhook..."
python -m src.data_automation.parse_review_to_train
