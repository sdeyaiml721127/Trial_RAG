# p9_audio_processor.py
"""
Handles Audio Processing: Transcription via MAAS API directly.
"""
import os
import httpx 
import time 
from typing import Callable, TypeVar
import p1_config as config

# Initialize httpx client (used for general purposes)
client = httpx.Client(verify=False)

# Generic type variable for the function being wrapped
T = TypeVar('T')

def _execute_with_retry(func: Callable[..., T], max_retries: int = 3, initial_delay: float = 1.0) -> T:
    """Helper function to retry execution with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                # print(f"   [Retry Warning] Attempt {attempt + 1}/{max_retries} failed ({type(e).__name__}). Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                raise e # Re-raise the exception after the last attempt

class AudioProcessor:
    def __init__(self):
        # Audio transcription endpoint URL
        self.transcribe_url = "https://genailab.tcs.in/v1/audio/transcriptions"
        self.audio_timeout = 60.0 # Increased timeout for audio transcription
        print(f"AudioProcessor initialized with model: {config.AUDIO_MODEL_NAME}")

    def _transcribe_file(self, audio_path: str) -> str:
        """
        Transcribes an audio file using a direct HTTP POST request.
        """
        if not os.path.exists(audio_path):
            return f"[Error] Audio file not found: {audio_path}"

        print(f"   -> Transcribing audio: {audio_path}...")
        
        def attempt_transcription():
            """Function to execute transcription using httpx, compatible with _execute_with_retry."""
            with open(audio_path, "rb") as file:
                files = {
                    'file': (os.path.basename(audio_path), file.read(), 'audio/mp3')
                }
                data = {
                    'model': config.AUDIO_MODEL_NAME,
                    'response_format': 'text',
                    'temperature': 0.0
                }
                headers = {
                    'Authorization': f'Bearer {config.MAAS_API_KEY}'
                }
                
                # Use httpx post request for file upload
                # ADDED: Explicit timeout
                response = httpx.post(
                    self.transcribe_url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self.audio_timeout, # Use the defined timeout
                    verify=False # Match client setting
                )
                response.raise_for_status() # Raise exception for 4xx or 5xx status codes
                
                # The MAAS endpoint should return a JSON object with a 'text' field
                result = response.json()
                return result.get('text', '[Transcription text field not found in response.]')

        try:
            # Execute transcription with retry logic
            return _execute_with_retry(attempt_transcription)
            
        except Exception as e:
            return f"[Audio Transcription Failed: {str(e)}]"


    def get_audio_analysis(self, audio_path: str) -> str:
        """
        Wrapper to format the output.
        """
        transcript = self._transcribe_file(audio_path)
        
        return (
            f"--- AUDIO TRANSCRIPT ---\n"
            f"{transcript}\n"
            f"--------------------------"
        )