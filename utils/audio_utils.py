# import os
import librosa
import soundfile as sf
# import numpy as np

def preprocess_audio(audio_input, target_sr=16000):
    """
    Loads, resamples, and normalizes any incoming audio input to a 16kHz mono array.
    
    Args:
        audio_input (str, bytes, or file-like): Path to the audio file or a binary stream.
        target_sr (int): Target sampling rate required by Whisper (default: 16000).
        
    Returns:
        np.ndarray: Cleaned audio time-series array ready for feature extraction.
    """
    try:
        # Load audio with librosa (automatically converts multi-channel/stereo to mono)
        speech, sr = librosa.load(audio_input, sr=None)
        
        # Resample explicitly if input sampling rate diverges from 16kHz
        if sr != target_sr:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=target_sr)
            
        return speech
    except Exception as e:
        raise RuntimeError(f"Error extracting audio features during preprocessing: {str(e)}")

def convert_to_wav(input_path, output_path, target_sr=16000):
    """
    Converts alternative audio formats (.mp3, .m4a, etc.) to a unified clean WAV file format.
    """
    speech, _ = librosa.load(input_path, sr=target_sr)
    sf.write(output_path, speech, target_sr, format='WAV', subtype='PCM_16')
    return output_path