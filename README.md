# 🚀 corrosion_detection_mlops

A corrosion detection solution using AI and MLOps.

---

## 🛠️ Virtual Environment & Dedicated Setup (`uv`)

To guarantee total compatibility between Apache Airflow components, graphical tools, and Computer Vision libraries, this project utilizes the high-performance package manager `uv`, locking the environment to a stable Python 3.11 interpreter.

```bash
# 1. Install required Linux system graphics and X11 dependencies for WSL2 UI rendering
sudo apt update && sudo apt install -y libxcb-xinerama0 libqt5gui5 libgles2-mesa-dev

# 2. Ensure you are in the project root directory and create an isolated venv in Python 3.11
uv venv --clear --python 3.11

# 3. Activate the created virtual environment
source .venv/bin/activate

# 4. Update core ecosystem packages and install project dependencies
# Note: opencv-python-headless prevents Qt library distribution conflicts with Labelme
uv pip install -U apache-airflow
uv pip install --upgrade werkzeug
uv pip install opencv-python-headless labelme
```

---

## 📹 Phase 1: Real-Time Data Collection & Dynamic Serving

This initial phase boots up the production application. The local FastAPI server autonomously locates the active `@champion` model weights inside the SQLite database. The OpenCV client captures the webcam stream and uses a background thread pool to dispatch frames, filtering and saving captures into the `online_data/` directory only when detections score above a **0.30 confidence threshold**.

To test inference serving, open **two separate terminals** at the root of the project:

### Terminal 1
*Initializes the FastAPI server and loads the live `@champion` model:*
```bash
python src/serving/app.py
```

### Terminal 2
*Starts the webcam stream capture powered by asynchronous background processing:*
```bash
python src/serving/client_webcam.py
```
*💡 Hint: Point your webcam to focus on metallic corrosion test scenarios. Once the terminal indicates that frames have been generated and flushed to `online_data/`, press the **'q'** key on the video window to quit.*

---

## 🧭 Phase 2: Monitoring & Automated Orchestration (Apache Airflow 3.0+ - Part I)

With live production data saved inside the `online_data/` folder, Apache Airflow steps in as the main orchestrator. The monitoring phase runs statistical evaluations to check for brightness, contrast, and color shifts across new frames using **Evidently AI**.

### Step 1: Initialize the Airflow Physical Structure
Create the execution directory and dispatch your DAG files to the orchestrator's native tracking path:
```bash
# Ensure the standard Airflow home folder and dags sub-folder exist on your home path
mkdir -p ~/airflow/dags

# Copy the structured pipeline files from the repository to the orchestrator's active scan directory
cp dags/*.py ~/airflow/dags/

# Migrate and initialize the metadata backend database
airflow db migrate
```

### Step 2: Initialize Web UI and Motors (3-Terminal Architecture)
Airflow 3.0+ strictly isolates the DAG reading worker loop into a decoupled processor. Open **three additional parallel terminals** with the virtual environment active (`source .venv/bin/activate`):

*   **Terminal A (Processor):** `airflow dag-processor`
*   **Terminal B (Scheduler):** `airflow scheduler`
*   **Terminal C (Web Dashboard):** `airflow api-server --port 8080`

*💡 Access the modern web interface at `http://localhost:8080/dags` using the username `admin` and the temporary password auto-generated inside Terminal C's startup logs.*

### Step 3: Run DAG 1 (Monitoring & Observability)
The first pipeline (`corrosion_phase1_monitoring`) computes data drift metrics and dynamically branches out decisions:
1. Triggers the observability job: `python -m src.observability.monitor` (Injecting the absolute repository path inside the `PYTHONPATH` system environment variable).
2. If the calculated shift breaks past the critical threshold, a `BranchPythonOperator` fires an Active Learning script (`python -m src.data_automation.send_to_review`), creating pseudo-annotations automatically and pushing files to `review_data/`.
3. If data is healthy and under control, the execution pipeline finishes cleanly inside the Airflow board to save compute cycles.

---

## 🎨 Phase 3: Human Curation & Active Learning (Human-in-the-Loop)

If a Data Drift alert was triggered during the previous phase, the flagged flawed images will be sitting inside the `review_data/` directory, enriched with high-quality AI pseudo-boxes pre-labeled by the current champion model.

```bash
# 1. Launch the Labelme graphical UI enforcing the native xcb backend plugin over WSL2
QT_QPA_PLATFORM=xcb labelme review_data

# 2. Once you finish fine-tuning/approving bounding boxes and hit save, run the automation converter:
python -m src.data_automation.parse_review_to_train
```
*💥 MLOps MAGIC:* The `parse_review_to_train.py` parser script translates the updated JSON structures into standard YOLO `.txt` files, merges them with your baseline training set, cleans up old staging directories, and **automatically hits a custom Webhook (REST HTTP API) waking up Airflow's Phase 2 DAG**, completely bypassing manual clicks on the orchestration board.

---

## 🏆 Phase 4: Syncing, Retraining & Live Hot Reloads (Apache Airflow 3.0+ - Part II)

The secondary pipeline (`corrosion_phase2_retrain`) wakes up immediately after receiving the incoming automated HTTP request kicked off by the data curation stage. It automates heavy computing tasks, strict asset governance, and continuous deployment:

### 4.1 Data Version Control (DVC)
*Airflow updates data hashes and pushes immutable datasets to cloud storage automatically:*
```bash
# Computes the MD5 checksum of the expanded data/ directory, updating data.dvc, and uploads data to Google Drive
dvc add data
dvc push

# Tracks the new pointers and snapshots inside Git's file history tree
git add data.dvc
git commit -m "chore: inject new verified web-cam corrosion frames into training pool via pipeline"
```

### 4.2 Training, Governance & Continuous Deployment (MLflow)
*The pipeline triggers a training run, subjects results to multi-metric validation against the active champion, and updates production servers live:*
```bash
# 1. Launches YOLO model training sessions natively by loading hyperparameters from model_config.yaml
python -m src.model_train.yolo

# 2. Processes verification loops over test splits while tracking model registrations via MLflow
# Enforces multi-metric gate criteria checking business metrics (mAP50 > 50% & Recall > 40%) against the current champion
python -m src.model_eval.yolo

# 3. CD (Continuous Deployment): Fires a hot-reload HTTP POST request straight into production endpoints
# The live FastAPI instance safely updates its loaded in-memory weights with ZERO DOWNTIME!
curl -X POST http://127.0.0.1:8000/reload

# 4. Spins up the MLflow Tracking Server UI for model provenance and historical experiment audit runs
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
*💡 Note: Navigate to `http://127.0.0.1:5000` to review comparison charts, track model runs, and see your updated package take over the **`champion`** alias status tag.*
