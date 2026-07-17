# tests/test_inference.py
# import pytest

def test_end_to_end_transcription_pipeline_mocked(client, mock_pashto_audio, mocker):
    """Verifies system flow from file intake down to structured response payloads."""
    
    # Target your actual operational engine transcription layer
    mocked_transcribe = mocker.patch(
        "utils.inference.PashtoTranscriber.transcribe",
        return_value="په خیر راغلیاست"
    )

    # Trigger full transmission using our fake file data stream
    response = client.post(
        "/api/v1/transcription/transcribe",
        files={"file": (mock_pashto_audio.name, mock_pashto_audio, "audio/wav")}
    )

    # Validate output matches your specific API router configuration structure
    assert response.status_code == 200
    json_data = response.json()
    assert "transcription" in json_data
    assert json_data["transcription"] == "په خیر راغلیاست"
    
    # Confirm our background service was invoked exactly once
    mocked_transcribe.assert_called_once()