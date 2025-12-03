# p7_video_processor.py
"""
Handles Video Processing: Visual Description + Robust Audio Transcription.
"""
import cv2
import base64
import os
import tempfile
import time 
from typing import List, Callable, TypeVar
import httpx 
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage
from moviepy.video.io.VideoFileClip import VideoFileClip 
import p1_config as config

# Initialize httpx client (used for Vision LLM)
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

class VideoProcessor:
    def __init__(self):
        # 1. Vision Model (Using ChatOpenAI)
        self.vision_llm = ChatOpenAI(
            base_url="https://genailab.tcs.in",
            model=config.VISION_MODEL_NAME, 
            api_key=config.MAAS_API_KEY, 
            temperature=0.1,
            http_client=client 
        )
        
        # 2. Audio transcription endpoint URL
        self.transcribe_url = "https://genailab.tcs.in/v1/audio/transcriptions"
        self.audio_timeout = 120.0  # Increased timeout to 120 seconds (2 minutes)
        
        print(f"VideoProcessor initialized.\n - Vision: {config.VISION_MODEL_NAME}\n - Audio: {config.AUDIO_MODEL_NAME}")

    def _extract_frames(self, video_path: str, num_frames: int = 1) -> List[str]: 
        """Extracts 1 representative frame."""
        if not os.path.exists(video_path): return []
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: 
            video.release()
            return []
        
        # Get the middle frame as representative
        middle_frame_index = total_frames // 2
        
        base64_frames = []
        
        video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        success, frame = video.read()
        if success:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        video.release()
        return base64_frames

    def _extract_audio_track(self, video_path: str) -> str:
        """
        Extracts audio from video and saves as a temporary MP3 file.
        Returns the path to the temporary MP3.
        """
        try:
            print(f"   -> Extracting audio track from video...")
            temp_dir = tempfile.gettempdir()
            temp_mp3_path = os.path.join(temp_dir, "temp_audio_extract.mp3")
            
            # Use MoviePy to extract audio
            video = VideoFileClip(video_path)
            if video.audio is None:
                video.close()
                return None
                
            video.audio.write_audiofile(temp_mp3_path, codec='mp3', logger=None)
            video.close()
            
            return temp_mp3_path
        except Exception as e:
            print(f"   [Audio Extraction Error]: {e}")
            return None

    def _transcribe_audio(self, video_path: str) -> str:
        """
        Extracts MP3 first, then sends to MAAS for transcription with retry logic.
        """
        audio_file_path = self._extract_audio_track(video_path)
        
        if not audio_file_path:
            return "[No audio track found in video]"

        print("   -> Transcribing extracted audio...")
        transcription_text = ""
        
        def attempt_transcription():
            """Function to execute transcription using httpx, compatible with _execute_with_retry."""
            # Since the audio track is extracted as MP3, we use audio/mp3 MIME type
            mime_type = 'audio/mp3' 

            with open(audio_file_path, "rb") as file:
                files = {
                    'file': (os.path.basename(audio_file_path), file.read(), mime_type)
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
                response = httpx.post(
                    self.transcribe_url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self.audio_timeout, # Use the defined timeout (120s)
                    verify=False # Match client setting
                )
                response.raise_for_status() # Raise exception for 4xx or 5xx status codes
                
                # The MAAS endpoint should return a JSON object with a 'text' field
                result = response.json()
                return result.get('text', '[Transcription text field not found in response.]')

        try:
            # Execute transcription with retry logic
            transcription_text = _execute_with_retry(attempt_transcription)
            
        except Exception as e:
            # Catch exceptions from httpx (ConnectionError, HTTPStatusError, ReadTimeout)
            transcription_text = f"[Audio Transcription Failed: {str(e)}]"
        
        finally:
            # Cleanup Temp File
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                
        return transcription_text

    def get_video_analysis(self, video_path: str) -> str:
        """
        Performs BOTH Vision and Audio analysis.
        """
        # A. VISUAL ANALYSIS (Only extracts 1 frame now)
        frames = self._extract_frames(video_path)
        visual_desc = "No visual data."
        if frames:
            print(f"   -> Analyzing {len(frames)} visual frame...")
            # We now only send one image URL part, which complies with the Llama model constraint.
            messages = [
                {
                    "type": "text", 
                    "text": "Provide a detailed and objective description of the scene and actions occurring in this single video frame."
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frames[0]}"}} # Only send the first (and only) frame
            ]
            try:
                response = self.vision_llm.invoke([HumanMessage(content=messages)])
                visual_desc = response.content
            except Exception as e:
                visual_desc = f"Visual analysis failed: {e}"

        # B. AUDIO ANALYSIS
        audio_transcript = self._transcribe_audio(video_path)

        # C. COMBINE
        full_report = (
            f"--- VIDEO ANALYSIS REPORT ---\n"
            f"1. VISUAL OBSERVATIONS:\n{visual_desc}\n\n"
            f"2. AUDIO TRANSCRIPT:\n{audio_transcript}\n"
            f"-----------------------------"
        )
        return full_report