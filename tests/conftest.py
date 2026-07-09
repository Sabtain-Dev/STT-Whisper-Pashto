import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    """Provides a reusable, synchronous test client for FastAPI routes."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def mock_pashto_audio():
    """Generates a transient, in-memory fake WAV file structure for mocking."""
    import io
    file_stream = io.BytesIO(b"RIFFxxxxWAVEfmt  dataxxxx")
    file_stream.name = "test_peshawari_speech.wav"
    return file_stream