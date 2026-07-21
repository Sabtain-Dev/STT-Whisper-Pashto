# Speech-to-Text- Finetuning On Regional Language (Pashto)

[![Continuous Integration](https://github.com/Sabtain-Dev/STT-Whisper-Pashto/actions/workflows/ci.yml/badge.svg)](https://github.com/Sabtain-Dev/STT-Whisper-Pashto/actions/workflows/ci.yml)

An end-to-end, resource-efficient Speech-to-Text (ASR) system built to fine-tune and serve OpenAI's Whisper architecture specifically for regional **Pakistani Pashto**. The project includes a dual-interface architecture: an interactive Streamlit frontend for end-users, and a highly modular, versioned FastAPI backend engineered for high-throughput microservices deployment.

---
## 🎯 Project Overview & Highlights
* **Target Dialect:** Pakistani Regional Pashto language structure.
* **Core Model:** Whisper-Small fine-tuned using Low-Rank Adaptation (LoRA) and merged for optimized execution footprints.
* **Accuracy Performance:** Reached a target Word Error Rate (WER) of **43.83%** (recorded from a baseline error rate of over 100%).
* **Hardware Profile:** Engineered to support runtime environments utilizing standard hardware constraints (CPU execution friendly, low VRAM configurations).
* **Structured Stream Diagnostics:** Replaced generic console print statements with python's native `logging` stream to provide clean, timestamped performance metrics (`[YYYY-MM-DD HH:MM:SS] [LEVEL] [file.py:line]`).
* **Container Resource Protection:** Integrated localized sliding duration audio pruning caps (30-second sliding processing window) inside the preprocessing pipeline to mitigate memory leaks and out-of-memory (OOM) failures on low-resource standard CPU runtimes.

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
│   ├── utils/                  # Included data validation and utility functions
│   │   └── validation.py
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
├── app_testing/                # App testing notebook for Google Collab
├── notebooks/                  # Experimental training scripts & model evaluation labs
├── configs/                    # Static configuration parameter specifications
├── workspace_data/             # Local database assets, upload stores, and API temp storage
|
├── Dockerfile.fastapi          # Dockerfile for the FastAPI backend
├── Dockerfile.streamlit        # Dockerfile for the Streamlit frontend
├── docker-compose.yml          # Docker Compose configuration for multi-container orchestration
├── .dockerignore               # Docker ignore file to exclude unnecessary files from the build context
├── .env.example                # Example environment variable configuration file  
|
├── LICENSE                     # MIT License for the project
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

### 🔧 Dynamic Runtime Configuration Injection
The container cluster is decoupled from hardcoded configuration strings. When running `docker compose up`, the containers automatically ingest your host machine's `.env` parameters:

* **Backend Environment Mapping:** Spawns a container instance running with the exact logging verbosity dictated by `${ENV_MODE}`.

* **Internal Network Routing:** The frontend container isolates network routing by targeting the backend's internal Docker DNS link (`http://fastapi:8000`) automatically, preventing host port collisions.

* **CPU Multiprocessing Core Optimization:** The initialization engine dynamically samples your host processor layout and maps thread configurations across OpenMP (`OMP_NUM_THREADS`) and Intel MKL libraries. It automatically allocates half of your machine's physical cores (bounded between 1 and 4 cores) to process heavy Whisper matrix calculations without locking up your system.

* **Data Persistence Matrix:**
User account schemas, history metrics, and audio staging blocks are persistently written to a local named Docker volume (pashto_shared_data). Your local transaction logs remain safe even if your containers are destroyed or updated.

---

## ⚡ Performance Profile & System Benchmarks

The system is optimized for low-resource CPU container execution, utilizing thread pooling and singleton model instantiation to maximize throughput.

### Benchmark Metrics (Tested on Standard 2 Core CPU / 8GB RAM)

| Audio Length | Total API Latency |
| :--- | :--- |
| **11 Seconds** | ~164.4s (1st Inference on CPU)|
| **15 Seconds** | ~89.752s (2nd Inference on CPU)|
| **11 Seconds** | ~46.2s (3rd Inference on CPU)|
| **15 Seconds** | ~46.7s (4th Inference on CPU)|
| **10 Seconds** | ~40.7s (1st Inference on GPU) |
| **14 Seconds** | ~8.7s (2nd Inference on GPU) |
| **13 Seconds** | ~8.4s (3rd Inference on GPU) |

> **Optimization Note:** The Whisper model weights are cached in memory upon application startup (`lru_cache` singleton pattern). Temporary files are staged under atomic UUID naming and automatically unlinked from the host disk immediately following inference execution.

---

## 🚀 Getting Started & Execution
### 1. Prerequisites & Environment Setup
Ensure your local environment uses Python 3.10. Install core system utilities like ffmpeg to enable sound track parsing on your machine:
```bash
# Clone the repository
git clone https://github.com/Sabtain-Dev/STT-Whisper-Pashto.git
cd STT-Whisper-Pashto

# Set up and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# Install application dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy the configuration template to create your local environment file:
```bash
cp .env.example .env
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

## 🛡️ Security, Guidelines & Technical Specifications

### 🔒 Security Considerations

* **Local Data Processing:** Audio uploads and temporary processing files are processed within the server environment and automatically cleaned up post-transcription to prevent data leaks or disk overflow.
* **API Key Management:** Sensitive credentials (such as `NGROK_AUTHTOKEN` or API tokens) are strictly managed using environment variables (`.env`) and are excluded from version control via `.gitignore`.
* **CORS & Network Safety:** In production, cross-origin resource sharing (CORS) rules on the FastAPI backend limit requests to trusted origins. When deployed via temporary tunnels (e.g., ngrok), endpoints are protected behind unique URLs.
* **Input Validation:** File upload endpoints validate MIME types and file signatures prior to passing streams to `ffmpeg` or `whisper` model layers to prevent arbitrary file execution.

### 🎙️ Supported File Formats

The backend uses `ffmpeg` for audio decoding, allowing standard audio and video container support.

| Category | Supported Formats | Recommended Format |
| :--- | :--- | :--- |
| **Audio** | `.wav`, `.mp3`, `.mp4`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.webm`, `.aac`, `.wma` | **`.wav` (16kHz, Mono)** |

> 💡 **Optimal Inference Performance:** For highest accuracy, supply uncompressed `.wav` files sampled at **16,000 Hz (16kHz)** with a **single audio channel (mono)**.

### ⚡ Maximum Upload Size

* **Default Single Audio File Limit:** **25 MB** per API request / Streamlit UI upload.
* **Handling Larger Files:** For extended recordings (e.g., long interviews or lectures), audio files should be pre-chunked into 30-second segments before passing through the transcription pipeline to maintain memory stability.

### ⚠️ Known Limitations

* **Dialectal Variations:** Model accuracy may vary across different Pashto regional dialects (e.g., Northern/Yousafzai vs. Southern/Kandahari).
* **Word Error Rate (WER):** The fine-tuned checkpoint currently achieves a **43.83% WER**. Transcriptions may still contain inaccuracies, particularly with non-standard vocabulary, background noise, or overlapping speakers.
* **Domain Context:** Performance is highest in trained domain areas (Agriculture, Food, Services, and General Conversation) and may show reduced performance in highly technical or legal contexts.
* **Hardware Sensitivity:** Running inference on CPU significantly increases execution latency compared to running on CUDA-enabled GPUs.

### 📜 Responsible Use

* **Ethical Usage:** This tool is built to advance low-resource language processing and support accessibility for Pashto speakers. It should **not** be used for unauthorized surveillance, deceptive practices, or malicious voice profiling.
* **Privacy & Consent:** Always obtain explicit consent from speakers before recording and uploading their audio for automated transcription.
* **Verification Required:** Due to the experimental nature of speech recognition fine-tuning, do not rely solely on automated transcriptions for critical legal, medical, or safety-critical decisions without human review.

---

## ⭐ Acknowledgements

* OpenAI Whisper
* Hugging Face Transformers
* PEFT / LoRA research
* Streamlit community

---