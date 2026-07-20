# Changelog

All notable changes to the **STT-Whisper-Pashto** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v2.2.0] - Unreleased / Current
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
- Refactored FastAPI code into modular modular layers (`routes`, `services`, `schemas`, `dependencies`).
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