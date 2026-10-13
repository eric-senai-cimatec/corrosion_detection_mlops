# 🚀 corrosion_detection_mlops

A corrosion detection solution using AI and MLOps.

---

## 🛠️ Virtual Environment & Dedicated Setup (`uv`)

To guarantee total compatibility between Apache Airflow components, graphical tools, and Computer Vision libraries, this project utilizes the high-performance package manager `uv`, locking the environment to a stable Python 3.11 interpreter.

```bash
# 1. Install required Linux system graphics and X11 dependencies for WSL2 UI rendering
sudo apt update && sudo apt install -y libxcb-xinerama0 libqt5gui5 libgles2-mesa-dev

# 2. Ensure the working context is in the project root directory and create an isolated venv in Python 3.11
uv venv --clear --python 3.11

# 3. Activate the created virtual environment
source .venv/bin/activate

# 4. Update core ecosystem packages and install project dependencies
# Note: opencv-python-headless prevents Qt library distribution conflicts with Labelme
uv pip install -U apache-airflow
uv pip install --upgrade werkzeug
uv pip install opencv-python-headless labelme

# 5. Install mandatory Airflow 3.0+ providers for Core and SMTP/Email operations
uv pip install apache-airflow-providers-standard apache-airflow-providers-smtp
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
*💡 Hint: Point the webcam to focus on metallic corrosion test scenarios. Once the terminal indicates that frames have been generated and flushed to `online_data/`, press the **'q'** key on the video window to quit.*

---

## 🧭 Phase 2: Monitoring & Automated Orchestration (Apache Airflow 3.0+)

With live production data saved inside the `online_data/` folder, Apache Airflow steps in as the main orchestrator. The monitoring phase runs statistical evaluations to check for brightness, contrast, and color shifts across new frames using **Evidently AI**.

### Step 1: Sync DAGs Folder Configuration
Ensure your `airflow.cfg` has the `dags_folder` pointing exactly to your absolute repository directory (e.g., `dags_folder = /home/eric/projects/corrosion_detection_mlops/dags`) or dispatch files to your central path. If choosing the central path, copy the pipeline scripts:
```bash
mkdir -p ~/airflow/dags
cp dags/*.py ~/airflow/dags/
```

### Step 2: Initialize the Metadata Database Backend
Before starting the triggers, migrate the database schema to generate all mandatory internal metadata tables (such as the `job` table):
```bash
airflow db migrate
```

### Step 3: Initialize Web UI and Motors (Optimized 2-Terminal Architecture)
Airflow 3.0+ strictly isolates the DAG reading worker loop into a decoupled processor. Instead of opening 4 messy screens, run the ecosystem using only **two parallel terminals** with the virtual environment active (`source .venv/bin/activate`):

#### Terminal A (Infrastructure Hub)
*Boots the webserver, internal scheduler, and API motors simultaneously:*
```bash
airflow standalone
```
*💡 Access the modern web interface at `http://localhost:8080/dags`. Use the username `admin` and the temporary password auto-generated inside this terminal's startup logs.*

#### Terminal B (Code Parsing Engine)
*Launches the decoupled loop that reads and refreshes your Python MLOps DAG scripts in real-time:*
```bash
airflow dag-processor
```

### Step 3.1: Configure SMTP Connection in Airflow UI (Anti-SSL & Sender Rule)
To ensure alert delivery via Gmail and bypass strict Airflow 3.0 `Email/SMTP` hooks that cause connection errors (`[SSL: WRONG_VERSION_NUMBER]`), you must define the connection explicitly using the **Generic** type.

1. Open the Airflow Web Dashboard (`http://localhost:8080`).
2. Navigate to **Admin** -> **Connections** and click the **+** (Add a new record) button.
3. Configure the connection fields exactly as defined below:
   * **Connection Id:** `smtp_default` (or `gmail_connection`, corresponding to the `conn_id` string declared in your DAGs).
   * **Connection Type:** `Generic`
   * **Host:** `smtp.gmail.com`
   * **Login:** `your_email@gmail.com`
   * **Password:** *Your 16-digit Google App Password* (Generated under Google Account Security -> 2-Step Verification -> App Passwords).
   * **Port:** `587`
4. In the **Extra** textbox field, paste the unmapped unified JSON configuration exactly as follows:
   ```json
   {
     "starttls": true,
     "disable_ssl": true,
     "use_ssl": false,
     "from_email": "your_email@gmail.com"
   }
   ```
   *💡 Note: Merging security parameters with `from_email` inside a single JSON dictionary fulfills required server parameters and eliminates 'unset from_email' execution exceptions.*
5. Click **Save**.

### Step 4: Run DAG 1 (Monitoring & Observability)
The first pipeline (`corrosion_phase1_monitoring`) computes data drift metrics and dynamically branches out decisions:
1. Triggers the observability job: `python -m src.observability.monitor` (Injecting the absolute repository path inside the `PYTHONPATH` system environment variable).
2. **Modern SMTP Import:** The email alerting subsystem imports directly from `airflow.providers.smtp.operators.smtp` to fit modern decoupled architecture.
3. If the calculated shift breaks past the critical threshold, a `BranchPythonOperator` fires an Active Learning script (`python -m src.data_automation.send_to_review`), creating pseudo-annotations automatically and pushing files to `review_data/`.
4. If data is healthy and under control, the execution pipeline routes to `stop_pipeline` and finishes cleanly to save compute cycles.

---

## 🎨 Phase 3: Human Curation & Active Learning (Human-in-the-Loop)

If a Data Drift alert was triggered during the previous phase, the flagged flawed images will be sitting inside the `review_data/` directory, enriched with high-quality AI pseudo-boxes pre-labeled by the current champion model.

To streamline the human annotation workflow, a unified execution script is used:
```bash
./run_curation.sh
```
*💥 MLOps MAGIC:* The execution of `./run_curation.sh` launches the Labelme UI over the native XCB platform. Once the operator updates the boxes, saves, and closes the window, the `parse_review_to_train.py` segment automatically triggers. It translates JSON files into normalized YOLO `.txt` files, incorporates them into the local `data/train` split, purges staging queues, and **invokes the local Airflow CLI wrapper (`airflow dags trigger`)**, instantly initiating the Phase 2 training pipeline without needing network HTTP credentials or generating authentication faults.

---

## 🏆 Phase 4: Syncing, Retraining & Live Hot Reloads (Apache Airflow 3.0+ - Part II)

The secondary pipeline (`corrosion_phase2_retrain`) wakes up immediately after receiving the incoming local CLI trigger kicked off by the curation stage. It automates heavy computing tasks, strict asset governance, and continuous deployment:

### 4.1 Corporate Cloud Sincronization & Governance (DVC via SharePoint Mount)
To bypass corporate tenant MFA and Azure AD OAuth2 restrictions, the system utilizes a secure file-system mirror strategy. The host Windows machine syncs the SharePoint document library via the native OneDrive sync engine, which the WSL2 Linux environment mounts under `/mnt/c/`.

```bash
# 1. Link the local DVC repository tracking configuration directly to the mapped corporate SharePoint mount point
dvc remote add -f -d sharepoint_fieb "/mnt/c/Users/eric.santoss/Sistema FIEB/Soft Robots - IA - mlops_for_computervision"

# 2. Re-compute immutable dataset hashes and generate clean MD5 checksum pointers
dvc add data

# 3. Synchronize dataset blobs directly into the Microsoft local mount
dvc push
```
*💡 Data Lineage Rule:* Dataset variants are committed as Content-Addressable Storage hashes under `files/md5/` inside SharePoint to maintain historic immutability, while the small metadata file `data.dvc` is tracked inside Git to maintain lineage records.

### 4.2 Training, Governance & Continuous Deployment (MLflow)
The pipeline automatically triggers training runs, subjects results to multi-metric validation against the active champion using the modern MLflow Alias registry API, and updates live endpoints:

```bash
# 1. Launches YOLO model training sessions natively on the GPU (RTX 3050) loading variables from model_args.yaml
python -m src.model_train.yolo

# 2. Processes verification loops over test splits while tracking model registrations via MLflow
# Enforces multi-metric gate criteria checking business metrics (mAP50 > 50% & Recall > 40%) against the current active champion alias
python -m src.model_eval.yolo

# 3. CD (Continuous Deployment): Fires a hot-reload HTTP POST request straight into production endpoints
# The live FastAPI instance safely updates its loaded in-memory weights with ZERO DOWNTIME!
curl -X POST http://127.0.0.1:8000/reload

# 4. Spins up the MLflow Tracking Server UI for model provenance and historical experiment audit runs
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
*💡 Note: Navigate to `http://127.0.0.1:5000` to review comparison charts, track model runs, and see the updated challenger package take over the active **`champion`** alias status tag.*

---

## 🔄 Dataset Reproducibility & Version Checkout

To reproduce an experiment or roll back the dataset to an exact state tracked by a specific MLflow run, use the DVC tracking file state via Git:

```bash
git checkout <commit-hash> data.dvc
dvc pull
```
