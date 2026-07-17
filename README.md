# Speech-to-Text- Finetuning On Regional Language (Pashto)

An end-to-end, resource-efficient Speech-to-Text (ASR) system built to fine-tune and serve OpenAI's Whisper architecture specifically for regional **Pakistani Pashto**. The project includes a dual-interface architecture: an interactive Streamlit frontend for end-users, and a highly modular, versioned FastAPI backend engineered for high-throughput microservices deployment.

---

# 🚀 STT-Whisper-Pashto

[![Continuous Integration](https://github.com/Sabtain-Dev/STT-Whisper-Pashto/actions/workflows/ci.yml/badge.svg)](https://github.com/Sabtain-Dev/STT-Whisper-Pashto/actions/workflows/ci.yml)

---
## 🎯 Project Overview & Highlights
* **Target Dialect:** Pakistani Regional Pashto language structure.
* **Core Model:** Whisper-Small fine-tuned using Low-Rank Adaptation (LoRA) and merged for optimized execution footprints.
* **Accuracy Performance:** Reached a target Word Error Rate (WER) of **43.83%** (recorded from a baseline error rate of over 100%).
* **Hardware Profile:** Engineered to support runtime environments utilizing standard hardware constraints (CPU execution friendly, low VRAM configurations).

---

## 📊 Dataset Link
The baseline model training and evaluations were driven by a custom, curated regional linguistic corpus containing over 5,000 highly targeted audio tracks across domains like agriculture, health, services, and general communication.
* **Kaggle Dataset:** [Pashto ASR Dataset](https://www.kaggle.com/datasets/itssabtain/pashto-asr-dataset)

## 🤗 Finetuned Model
The baked-in Whisper-LoRA model is open-sourced and hosted directly on the Hugging Face Model Hub for single source of truth accessibility:
* **Hugging Face Repository:** [Sabtain-Dev/STT-Whisper-Pashto](https://huggingface.co/Sabtain-Dev/STT-Whisper-Pashto)

---

## 📁 Project Structure

```text
STT_Whisper_Pashto/
│
├── api/                        # Production Decoupled FastAPI Backend Layer
│   ├── routes/                 # Explicitly versioned routing endpoints
│   │   └── transcription.py
│   ├── schemas/                # Pydantic request/response validation models
│   │   ├── request.py
│   │   └── response.py
│   ├── services/               # Core business logic handlers (disk staging & verification)
│   │   └── transcription_service.py
│   ├── config.py               # Environmental configuration and settings controller
│   ├── dependencies.py         # Thread-safe model singleton dependency injection matrix
│   ├── exceptions.py           # Custom application domain error wrappers
│   ├── logger.py               # Centralized structured application runtime logger
│   └── main.py                 # Core API application engine gatekeeper
│
├── app/                        # Streamlit Interactive User UI Frontend Application
│   └── app.py
│
├── utils/                      # Shared Core Machine Learning Utilities
│   ├── inference.py            # Centralized ML source of truth (PashtoTranscriber)
│   ├── audio_utils.py          # Waveform decoders and audio standardizers
│   └── model_utils.py          # Weights management utilities
│
├── notebooks/                  # Experimental training scripts & model evaluation labs
├── configs/                    # Static configuration parameter specifications
├── workspace_data/             # Local database assets, upload stores, and API temp storage
|
├── Dockerfile.fastapi          # Dockerfile for the FastAPI backend
├── Dockerfile.streamlit        # Dockerfile for the Streamlit frontend
├── docker-compose.yml          # Docker Compose configuration for multi-container orchestration
├── .dockerignore               # Docker ignore file to exclude unnecessary files from the build context
|
├── requirements.txt            # Python dependencies for the project
└── README.md                   # System configuration and documentation handbook
```

---

## Production Architecture: Decoupled API & Frontend

The system has been updated into a decoupled, scalable multi-tier architecture. Machine learning inference tasks are handled exclusively by a robust **FastAPI backend**, while the user interface is served independently by a lightweight **Streamlit client application**.
```text
┌─────────────────┐  HTTP Requests (POST /transcribe)   ┌─────────────────┐
│                 │ ──────────────────────────────────> │                 │
│    Streamlit    │                                     │     FastAPI     │
│  Frontend UI    │ <────────────────────────────────── │  Inference Core │
│                 │       Structured JSON Response      │  (Whisper/LoRA) │
└─────────────────┘                                     └─────────────────┘
```

---

## 🐳 Docker Deployment & Containerization

The system runs on a containerized architecture managed by Docker Compose. This ensures unified environment setups, isolates runtime dependencies, and isolates data flows across production services.

### Architecture Composition
1. **Frontend (`pashto_whisper_frontend`):** Lightweight interactive dashboard stream. Employs internal DNS hooks to securely proxy computational audio files.
2. **Backend (`pashto_whisper_backend`):** High-throughput microservice layer utilizing a dedicated machine-learning processing footprint. Instantiates Whisper weights internally and outputs structured metrics.

### Docker Core Prerequisites
Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your host machine and that the service daemon engine is running.

### Quick Start Commands

* **Build Container Images:**
  Build both services from local configuration parameters:
  ```bash
  docker compose build
  ```

* **Launch the Stack Environment:**
Boot up the interconnected ecosystem nodes:
```bash
docker compose up
```

* **Verify Interface Connectivity Links:**
1. Interactive Frontend UI: Open http://localhost:8501 to access the main user app.
2. Self-Documenting API Portal: Open http://localhost:8000/api/v1/docs to test endpoints via Swagger.

* **Shut Down the Services:**
Stop and remove container instances smoothly:
```bash
docker compose down
```

* **Data Persistence Matrix:**
User account schemas, history metrics, and audio staging blocks are persistently written to a local named Docker volume (pashto_shared_data). Your local transaction logs remain safe even if your containers are destroyed or updated.

---

## 🚀 Getting Started & Execution
### 1. Prerequisites & Environment Setup
Ensure your local environment uses Python 3.10. Install core system utilities like ffmpeg to enable sound track parsing on your machine:
```bash
# Clone the repository
git clone [https://github.com/Sabtain-Dev/STT-Whisper-Pashto.git](https://github.com/Sabtain-Dev/STT-Whisper-Pashto.git)
cd STT-Whisper-Pashto

# Set up and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# Install application dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create an environmental validation configuration file named .env inside your root workspace directory:
```bash
APP_NAME="Pashto Whisper Production API"
DEBUG=False
MERGED_MODEL_PATH="Sabtain-Dev/STT-Whisper-Pashto"
HF_TOKEN=""
```

### 3. Launching the FastAPI Backend Service
```bash
python -m uvicorn api.main:app --reload
```
- Interactive API Documentation Portal (Swagger UI): http://localhost:8000/api/v1/docs
- Alternative Static UI (ReDoc): http://localhost:8000/api/v1/redoc

### 4. Running the Streamlit Frontend UI
#### Features:
* Upload audio (wav, mp3, mp4, etc.)
* Automatic transcription (Pashto)
* Editable output
* WER comparison with reference text
* Translation support (English & Urdu)
* User login system (SQLite database)

#### Run App Locally
```bash
streamlit run app/app.py
```

---

## 🧪 Automated Testing Matrix

This project uses `pytest` to maintain structural integrity across all components before build compilation or deployment.

### Test Architecture Blueprint
* **Unit Tests (`tests/test_audio.py`):** Low-overhead structural assertion sweeps validating individual modules (e.g., extensions, normalization paths).
* **API Route Assertions (`tests/test_api.py`):** Challenges API integrity profiles, network responses, and input boundary exception rules.
* **Integration Pipeline (`tests/test_inference.py`):** Mocks processing bottlenecks to validate the full workflow from initial file consumption down to output parameter mapping.

### Running the Test Suite
1. Ensure development requirements are active in your local virtual environment context.
2. Execute the test command runner from the root directory:
```bash
pytest -v
```

---

## ⚙️ Continuous Integration (CI)

This project uses **GitHub Actions** to guarantee system stability and prevent regression errors.

### What the Pipeline Does
On every `push` or `pull_request` targeting the `main` branch, the runner automatically:
1. Spins up a clean Ubuntu virtual machine.
2. Configures a Python 3.10 execution environment.
3. Installs runtime dependencies (`requirements.txt`) and test-suite utilities (`requirements_dev.txt`).
4. Runs **Ruff** to enforce clean code formatting and catch syntax mistakes.
5. Runs **Pytest** with path mapping injected to evaluate unit, API, and integration test coverage.

### Monitoring Runs
You can inspect active build sequences, execution logs, and detailed step breakdowns under the **Actions** tab of this repository.

---

## ⭐ Acknowledgements

* OpenAI Whisper
* Hugging Face Transformers
* PEFT / LoRA research
* Streamlit community

---