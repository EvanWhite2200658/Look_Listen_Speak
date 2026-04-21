# src/tts/tts_test.py

from src.tts.piper_tts_service import PiperTTSService
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]

tts = PiperTTSService(
    model_path=root_dir / "models" / "tts" / "en_GB-alba-medium.onnx"
)

result = tts.speak("Hello, this is a test of the speech system.")

print(result)