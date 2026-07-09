def test_api_health_endpoint(client):
    """Verifies the core heart-beat readiness endpoint responds with 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "Pashto ASR Core"}

def test_transcribe_endpoint_missing_payload(client):
    """Confirms 422 processing validation handles unpopulated form postings."""
    response = client.post("/api/v1/transcription/transcribe", files={})
    assert response.status_code == 422