# 📡 STT-Whisper-Pashto API Reference

Base URL: `http://localhost:8000/api/v1`

---

## Endpoints Summary

| Endpoint | Method | Description | Auth Required |
| :--- | :---: | :--- | :---: |
| `/health` | `GET` | Service health status check | No |
| `/info` | `GET` | Model & environment metadata | No |
| `/transcribe` | `POST` | Upload audio and receive Pashto transcription | No |

---

## Endpoint Details

### 1. Health Check
* **Method:** `GET`
* **Path:** `/health`
* **Response (200 OK):**
  ```json
  {
    "status": "healthy"
  }

### 2. Audio Transcription
* **Method:** `POST`
* **Path:** `/transcribe`
* **Content-Type:** `multipart/form-data`
* **Payload:** `file` (Binary Audio File — WAV, MP3, FLAC, OGG, M4A up to 25MB)
* **Response (200 OK):**
  ```json
  {
    "transcription": "د پښتو ژبې ماډل ازموینه",
    "processing_time": 2.41,
    "audio_filename": "sample.wav",
  "status": "success"
  }

Error Response (400 Bad Request / 415 Unsupported Media Type):

  ```json
  {
    "detail": "Invalid file format. Supported formats: wav, mp3, flac, ogg, m4a"
  }
  ```

### Client Integration Examples:cURL
```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.wav"
```

#### Python (requests)
  ```text
  import requests

  url = "http://localhost:8000/api/v1/transcribe"
  file_path = "sample.wav"

  with open(file_path, "rb") as f:
      files = {"file": (file_path, f, "audio/wav")}
      response = requests.post(url, files=files)

  print("Status Code:", response.status_code)
  print("Response:", response.json())
  ```
  