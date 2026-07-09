import pytest
import numpy as np
import io
from utils.audio_utils import preprocess_audio, convert_to_wav

def test_preprocess_audio_matrix_generation(mocker):
    """Ensures preprocess_audio returns a valid 16kHz numpy array."""
    # Mock librosa.load to prevent it from looking for a real file on your disk
    fake_audio_array = np.zeros(32000, dtype=np.float32)  # 2 seconds of silence
    mocker.patch("librosa.load", return_value=(fake_audio_array, 16000))
    
    # Run function using a mock string path
    result = preprocess_audio("fake_path.wav")
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 32000

def test_preprocess_audio_resampling_trigger(mocker):
    """Verifies that librosa.resample is invoked when the input rate is not 16kHz."""
    fake_audio_array = np.zeros(22050, dtype=np.float32)
    mocker.patch("librosa.load", return_value=(fake_audio_array, 22050))
    
    # Mock the resample function to return a clean 16kHz array variant
    mocked_resample = mocker.patch("librosa.resample", return_value=np.zeros(16000, dtype=np.float32))
    
    result = preprocess_audio("fake_path.mp3")
    
    # Confirm resampling was triggered due to the 22.05kHz mismatch
    mocked_resample.assert_called_once_with(fake_audio_array, orig_sr=22050, target_sr=16000)
    assert len(result) == 16000

def test_preprocess_audio_failure_handling(mocker):
    """Ensures failures inside librosa raise a clean RuntimeError."""
    mocker.patch("librosa.load", side_effect=Exception("Corrupted audio headers"))
    
    with pytest.raises(RuntimeError) as exc_info:
        preprocess_audio("broken_file.wav")
        
    assert "Error extracting audio features during preprocessing" in str(exc_info.value)

def test_convert_to_wav_pipeline(mocker):
    """Validates that convert_to_wav reads the target asset and writes PCM_16 data."""
    fake_audio_array = np.zeros(16000, dtype=np.float32)
    mocker.patch("librosa.load", return_value=(fake_audio_array, 16000))
    
    # Mock soundfile.write so it doesn't actually write a file to your E:\ drive
    mocked_sf_write = mocker.patch("soundfile.write")
    
    output = convert_to_wav("input.mp3", "output.wav")
    
    # Verify file-write contract parameters
    mocked_sf_write.assert_called_once_with("output.wav", fake_audio_array, 16000, format='WAV', subtype='PCM_16')
    assert output == "output.wav"