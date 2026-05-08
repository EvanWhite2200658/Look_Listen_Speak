# look_listen_speak_demo.spec
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = []

hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("tokenizers")
hiddenimports += collect_submodules("huggingface_hub")
hiddenimports += collect_submodules("safetensors")

hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("ctranslate2")

hiddenimports += collect_submodules("silero_vad")
hiddenimports += collect_submodules("piper")
hiddenimports += collect_submodules("sounddevice")
hiddenimports += collect_submodules("soundfile")

hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("psutil")

# Monitor / OpenCV
hiddenimports += collect_submodules("cv2")

# Gaze / vision stack
hiddenimports += collect_submodules("mediapipe")
hiddenimports += collect_submodules("gazefollower")

datas = []
datas += collect_data_files("mediapipe")
datas += collect_data_files("gazefollower")
datas += collect_data_files("silero_vad")
datas += collect_data_files("cv2")

datas += [
    ("models/tts/en_GB-alba-medium.onnx", "models/tts"),
    ("models/tts/en_GB-alba-medium.onnx.json", "models/tts"),
    (
        "src/turn_prediction/artifacts/turn_prediction_runtime_compatible/best_model.pt",
        "src/turn_prediction/artifacts/turn_prediction_runtime_compatible",
    ),
]

a = Analysis(
    ["src/runtime/final_runtime_pipeline.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LookListenSpeakDemo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LookListenSpeakDemo",
)