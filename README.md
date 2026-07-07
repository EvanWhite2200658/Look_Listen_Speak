This is the physical product of my Honours Dissertation. 
It is a conversational system that measures the user's gaze direction and uses that to inform the length of silence that it waits for before replying.
It is intended to reduce the latency of turn transitions in human-robot conversation, making the dialogue feel more natural.
The system is not in a production ready state. It was developed as an experimental proof to the current theory around gaze-based timing decisions.
The models used are as follows:
- Gaze tracking and facial measurements: https://github.com/GanchengZhu/GazeFollower
- Voice Activity Detection: https://github.com/snakers4/silero-vad
- Transcription: https://github.com/SYSTRAN/faster-whisper
- Text to speech: https://github.com/OHF-Voice/piper1-gpl
- LLM for system response: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- Custom built model for determining an upcoming turn transition based on gaze activity.

