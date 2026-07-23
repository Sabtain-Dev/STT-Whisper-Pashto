# Changelog

All notable changes to the **STT-Whisper-Pashto** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v2.6] - 2026-07-23
### Added
- Standardized API reference guide (`docs/API.md`) with cURL and Python client execution examples.
- Pre-configured Postman Collection (`docs/postman/STT_Whisper_Pashto.postman_collection.json`) for endpoint testing.
- Enhanced OpenAPI and Swagger UI metadata with route tags, detailed request parameters, and response schemas.

---

## [v2.5] - 2026-07-23
### Added
- Comprehensive system architecture documentation and Mermaid design diagrams in `docs/`:
  - System Architecture & Data Flow
  - Sequence Request/Response Flow
  - Machine Learning Inference & Fine-Tuning Pipelines

### Fixed
- Updated `docker-compose.yml` healthcheck command to use Python `urllib` module instead of `curl` to prevent false-unhealthy container flags on minimal base images.
- Pointed backend healthcheck probe to active API endpoint path.

---

## [v2.4] - 2026-07-22
### Added
- Dual Docker Compose strategy: `docker-compose.yml` for production image pulls and `docker-compose.dev.yml` for local builds.
- Published production-ready multi-platform container images to Docker Hub (`msabtainkhan/stt-whisper-pashto-fastapi` and `msabtainkhan/stt-whisper-pashto-streamlit`).
- Added Docker Hub badges and streamlined local installation instructions to README.

---

## [v2.2.0] - 2026-07-20
### Added
- Multi-metric timing tracking, exposing both inference time and total file processing time in API payloads.
### Fixed
- Aligned route return unpack signatures and pytest response assertions in `tests/test_inference.py`.

---

## [v2.1] - 2026-07-20
### Added
- Automated temporary disk space cleanup post-transcription.
- Execution timing log metrics.
### Changed
- Optimized model inference lifecycle and resource management.

---

## [v2.0-perf]
### Performance
- Implemented CPU multi-threading optimizations and resource protection boundaries.

---

## [v2.0-rc2]
### CI/CD
- Integrated automated GitHub Actions test pipeline (`pytest`).
- Fixed code style and linting violations.

---

## [v2.0-rc1] / [v2.0-beta]
### Added
- Added GitHub Actions CI workflow config (`.github/workflows/ci.yml`).
- Added automated unit test suite for utilities and API endpoints.
- Parameterized API environment endpoints for container deployment.
- Enhanced README documentation with CI status badges.

---

## [v1.9]
### Added
- Containerized FastAPI backend and Streamlit frontend using `docker-compose.yml`.

---

## [v1.8]
### Changed
- Connected Streamlit user interface directly with FastAPI REST endpoints.

---

## [v1.7]
### Architecture
- Refactored FastAPI code into modular layers (`routes`, `services`, `schemas`, `dependencies`).
- Initialized core FastAPI endpoints for Pashto speech transcription.

---

## [v1.6] & [v1.5]
### Changed
- Refined local application runtime and updated dependency specifications in `requirements.txt`.

---

## [v1.4] & [v1.3]
### Added
- Integrated Pashto speech datasets and fine-tuned model checkpoints yielding lower Word Error Rate (WER).
- Removed sensitive Hugging Face access tokens from notebook checkpoints.

---

## [v1.2] & [v1.1]
### Added
- Restructured repository layout and introduced initial application skeleton.
- Updated documentation with dataset references and proper link formatting.

---

## [v1.0]
### Initial Release
- Initial baseline repository setup for Pashto Speech-to-Text fine-tuning with OpenAI Whisper.
- Configured foundational `requirements.txt` and base training notebooks.