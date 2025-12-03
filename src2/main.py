# main.py
"""
Simple script to analyze and describe media files (Video, Image, Audio)
using the MAAS multimedia processors.

Usage:
1. Run: python main.py
2. Enter the absolute path to your file when prompted.
3. Supported file types: .mp4, .avi (Video), .jpg, .png (Image), .mp3, .wav (Audio)
"""
import os
import ssl
import sys

import urllib3

# --- 1. SETUP ENV & IMPORTS ---
try:
    # Import config and set environment variables
    import p1_config as config

    # Use MAAS_API_KEY for all LLM and Groq-based services
    os.environ["GROQ_API_KEY"] = config.MAAS_API_KEY

except ImportError as e:
    print(f"Error: Could not import a required module. Ensure files are present: {e}")
    sys.exit(1)
except AttributeError as e:
    print(f"Error: A required API key is missing from p1_config.py: {e}")
    sys.exit(1)

# Import Media Processors (Now at top level)
from p7_video_processor import VideoProcessor
from p8_image_processor import ImageProcessor
from p9_audio_processor import AudioProcessor

# --- 2. SSL WORKAROUND (Critical for MAAS endpoints) ---
# This is crucial for environments where self-signed certificates might be used.
# We use getattr/setattr here to access protected members ('_') dynamically,
# which prevents Pylint W0212 errors without requiring a disable comment.
create_unverified = getattr(ssl, "_create_unverified_context")
setattr(ssl, "_create_default_https_context", create_unverified)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# --- END SSL WORKAROUND ---


def get_file_type(path: str) -> str:
    """Determines the file type based on extension."""
    ext = os.path.splitext(path.lower())[1]
    if ext in [".mp4", ".avi", ".mov"]:
        return "video"
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return "image"
    if ext in [".mp3", ".wav", ".flac", ".m4a"]:
        return "audio"
    return "unknown"


def main():
    """Main function for the media description script."""

    # Initialize Processors
    video_processor = VideoProcessor()
    image_processor = ImageProcessor()
    audio_processor = AudioProcessor()

    print("=" * 50)
    print("🚀 Media Description Service Started")
    print("=" * 50)
    print("   Supported formats: Video (.mp4), Image (.jpg, .png), Audio (.mp3)")
    print("   Type 'quit' to exit.")

    while True:
        try:
            user_input = input("\n(Input) Enter absolute file path: ")
            file_path = user_input.strip().replace('"', "").replace("'", "")

            if file_path.lower() in ["quit", "exit"]:
                print("Exiting...")
                break

            if not os.path.exists(file_path):
                print(f"[ERROR] File not found: {file_path}")
                continue

            file_type = get_file_type(file_path)
            report = ""

            print(f"   -> Detected file type: {file_type.upper()}")

            if file_type == "video":
                print("   -> Starting Video Analysis (Visual & Audio)...")
                report = video_processor.get_video_analysis(file_path)

            elif file_type == "image":
                print("   -> Starting Image Description (Visual only)...")
                report = image_processor.analyze_image(file_path)

            elif file_type == "audio":
                print("   -> Starting Audio Transcription (Audio only)...")
                report = audio_processor.get_audio_analysis(file_path)

            else:
                report = (
                    f"[ERROR] Unsupported file format: {os.path.basename(file_path)}"
                )

            print("\n" + "=" * 50)
            print("--- GENERATED DESCRIPTION/TRANSCRIPT ---")
            print(report)
            print("=" * 50)

        # pylint: disable=broad-exception-caught
        except Exception as err:
            print(f"\n[CRITICAL ERROR] An unexpected error occurred: {err}")


if __name__ == "__main__":
    main()
