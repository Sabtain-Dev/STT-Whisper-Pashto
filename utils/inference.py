# utils/inference.py
# utils/inference.py (The Machine Learning Engine): This houses the actual Hugging Face Transformers model instance setup. The service layer invokes this module to perform the heavy computational tasks (loading weights, processing waveform vectors via audio decoders, and returning text tokens).

import torch

from .audio_utils import preprocess_audio
from .model_utils import load_inference_components


class PashtoTranscriber:
    def __init__(self, model_id_or_path=None, hf_token=None):
        """
        Initializes a stable instance of the Pashto speech processing environment.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = load_inference_components(model_id_or_path, hf_token=hf_token)
        
        # Explicitly configure decoding configurations for Pashto (ps)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="pashto", 
            task="transcribe"
        )

    def transcribe(self, audio_file_path):
        """
        Transforms raw input audio files directly into native Pashto transcription strings.
        
        Args:
            audio_file_path (str): Local file system path pointing to targeted speech segment.
            
        Returns:
            str: Decoded structural Pashto text string.
        """
        # Step 1 & 2: Process audio data stream to unified 16kHz format
        audio_array = preprocess_audio(audio_file_path, target_sr=16000)
        
        # Step 3: Compute log-mel input spectrograms via processor
        input_features = self.processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Step 4: Generate contextual text tokens across multi-head attention blocks
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                max_length=448  # Matches notebook experimental constraints
            )
            
        # Step 5: Convert generated token arrays back to clean human text
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()